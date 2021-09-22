# coding=utf-8

from . import utils, layers, layerspp, normalization

from typing import Optional, Sequence, Tuple, Union

import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import (
    ResBlock,
    get_conv_layer,
    get_upsample_layer,
)
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode

from torchinfo import summary

get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
SegResBlockpp = layerspp.SegResBlockpp
MultiSequential = layers.MultiSequential


@utils.register_model(name="ncsnpp3d")
class SegResNetpp(nn.Module):
    """
    Time condioned version of SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module does not include the variational autoencoder (VAE).

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`monai.networks.blocks.SubpixelUpsample`.

    """

    def __init__(
        self,
        config,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 2,
        out_channels: int = 2,
        dropout_prob: Optional[float] = None,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        time_embedding_sz: int = 1024,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        data = config.data

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = data.num_channels
        self.time_embedding_sz = time_embedding_sz
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(
                    f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead."
                )
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(
            spatial_dims, self.in_channels, self.init_filters
        )
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)
        self.time_embed_layer = self._make_time_cond_layers()
        # self.attention_block = SABlock(
        #     init_filters * 2 ** len(blocks_down), num_heads=8
        # )

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_time_cond_layers(self):

        sz = self.time_embedding_sz

        projection = layerspp.GaussianFourierProjection(embedding_size=sz // 4)
        # Projection layer doubles the input_sz
        # Since it concats sin and cos projections
        dense_0 = layerspp.make_dense_layer(sz // 2, sz)
        dense_1 = layerspp.make_dense_layer(sz, sz)

        return nn.Sequential(projection, dense_0, dense_1)

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm, temb_dim = (
            self.blocks_down,
            self.spatial_dims,
            self.init_filters,
            self.norm,
            self.time_embedding_sz,
        )
        for i in range(len(blocks_down)):
            layer_in_channels = filters * 2 ** i
            pre_conv = (  # PUSH THIS INTO THE Res++ Block
                get_conv_layer(
                    spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2
                )
                if i > 0
                else nn.Identity()
            )
            down_layer = MultiSequential(  # First layer needs the preconv
                SegResBlockpp(
                    spatial_dims,
                    layer_in_channels,
                    norm=norm,
                    pre_conv=pre_conv,
                    temb_dim=temb_dim,
                ),
                *[
                    SegResBlockpp(
                        spatial_dims,
                        layer_in_channels,
                        norm=norm,
                        pre_conv=None,
                        temb_dim=temb_dim,
                    )
                    for _ in range(blocks_down[i] - 1)
                ],
            )
            down_layers.append(down_layer)

        # TODO: Add some kind of attention block

        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm, temb_dim = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
            self.time_embedding_sz,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                MultiSequential(
                    *[
                        SegResBlockpp(
                            spatial_dims,
                            sample_in_channels // 2,
                            norm=norm,
                            temb_dim=temb_dim,
                        )
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(
                            spatial_dims,
                            sample_in_channels,
                            sample_in_channels // 2,
                            kernel_size=1,
                        ),
                        get_upsample_layer(
                            spatial_dims,
                            sample_in_channels // 2,
                            upsample_mode=upsample_mode,
                        ),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(
                name=self.norm,
                spatial_dims=self.spatial_dims,
                channels=self.init_filters,
            ),
            self.act,
            get_conv_layer(
                self.spatial_dims,
                self.init_filters,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x, t):
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []
        t = self.time_embed_layer(t)

        for i, down in enumerate(self.down_layers):
            x = down(x, t)
            #             print(f"Computed layer {i}: {x.shape}")
            down_x.append(x)

        down_x.reverse()

        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x, t)
        #             print(f"Computed layer {i}: {x.shape}")

        if self.use_conv_final:
            x = self.conv_final(x)
        return x
