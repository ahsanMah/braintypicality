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
MultiSequential = layers.MultiSequential
AttentionBlock = layerspp.AttentionBlock3d
get_conv_layer_pp = layerspp.get_conv_layer


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
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()
        self.register_buffer("sigmas", torch.tensor(utils.get_sigmas(config)))

        if spatial_dims not in (2, 3):
            raise AssertionError("spatial_dims can only be 2 or 3.")

        self.data = config.data

        self.spatial_dims = spatial_dims
        self.init_filters = config.model.nf
        self.in_channels = self.data.num_channels
        self.out_channels = self.data.num_channels
        self.time_embedding_sz = config.model.time_embedding_sz
        self.fourier_scale = config.model.fourier_scale
        self.blocks_down = config.model.blocks_down
        self.blocks_up = config.model.blocks_up
        self.self_attention = config.model.self_attention
        self.dilation = config.model.dilation
        self.conv_size = config.model.conv_size
        self.scale_by_sigma = config.model.scale_by_sigma
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.compile = config.model.jit
        self.attention_type = config.model.attention_type

        assert resblock_type in ["segresnet", "biggan"], ValueError(
            f"resblock type {resblock_type} unrecognized."
        )
        assert embedding_type in ["fourier", "positional"]

        self.resblock_pp = config.model.resblock_pp
        self.act = config.model.act  # get_act_layer(act)

        if config.model.dropout > 0.0:
            self.dropout_prob = config.model.dropout
        else:
            self.dropout_prob = None

        if self.dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](self.dropout_prob)

        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(
                    f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead."
                )
            norm = ("group", {"num_groups": num_groups})

        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final

        if self.resblock_pp:
            self.conv_layer = functools.partial(
                get_conv_layer_pp,
                init_scale=config.model.init_scale,
                kernel_size=self.conv_size,
            )
        else:
            self.conv_layer = get_conv_layer

        self.convInit = self.conv_layer(
            spatial_dims, self.in_channels, self.init_filters
        )

        if resblock_type == "segresnet":
            ResBlockpp = functools.partial(
                layerspp.SegResBlockpp,
                act=self.act,
                kernel_size=self.conv_size,
                resblock_pp=self.resblock_pp,
                dilation=self.dilation,
                jit=config.model.jit,
                norm=self.norm,
                spatial_dims=self.spatial_dims,
            )
        elif resblock_type == "biggan":
            ResBlockpp = functools.partial(
                layerspp.ResnetBlockBigGANpp,
                act=self.act,
                kernel_size=self.conv_size,
                spatial_dims=self.spatial_dims,
            )

        self.time_embed_layer = self._make_time_cond_layers(embedding_type)
        self.down_layers = self._make_down_layers(ResBlockpp, jit_compile=self.compile)
        # self.down_layers = torch.jit.script(self.down_layers)

        if self.self_attention:
            self.attention_block = AttentionBlock(
                channels=self.init_filters * 2 ** (len(self.blocks_down) - 1)
            )
            if self.compile:
                self.attention_block = torch.jit.script(self.attention_block)

            if self.attention_type == "block":
                self.pre_attention = ResBlockpp(
                    in_channels=self.init_filters * 2 ** (len(self.blocks_down) - 1),
                    temb_dim=self.time_embedding_sz * 4,
                )
                self.post_attention = ResBlockpp(
                    in_channels=self.init_filters * 2 ** (len(self.blocks_down) - 1),
                    temb_dim=self.time_embedding_sz * 4,
                )
        
        self.up_layers, self.up_samples = self._make_up_layers(
            ResBlockpp, jit_compile=self.compile
        )
        self.conv_final = self._make_final_conv(self.out_channels)

    def _make_time_cond_layers(self, embedding_type):
        layer_list = []

        if embedding_type == "fourier":
            # Projection layer doubles the input_sz
            # Since it concats sin and cos projections
            projection = layerspp.GaussianFourierProjection(
                embedding_size=self.time_embedding_sz, scale=self.fourier_scale
            )
            layer_list.append(projection)

        sz = self.time_embedding_sz * 2
        dense_0 = layerspp.make_dense_layer(sz, sz * 2)
        dense_1 = layerspp.make_dense_layer(sz * 2, sz * 2)

        layer_list.append(dense_0)
        layer_list.append(dense_1)

        return nn.Sequential(*layer_list)

    def _make_down_layers(self, ResNetBlock, jit_compile=False):
        down_blocks = nn.ModuleDict()
        blocks_down, spatial_dims, filters, norm, temb_dim = (
            self.blocks_down,
            self.spatial_dims,
            self.init_filters,
            self.norm,
            self.time_embedding_sz * 4,
        )
        for i in range(len(blocks_down)):
            layer_in_channels = filters * 2**i
            final_block_idx = blocks_down[i] - 2

            pre_conv = (  # PUSH THIS INTO THE Res++ Block
                self.conv_layer(
                    spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2
                )
                if i > 0
                else nn.Identity()
            )
            down_layers = [  # First layer needs the preconv
                ResNetBlock(
                    layer_in_channels,
                    pre_conv=pre_conv,
                    temb_dim=temb_dim,
                ),
                *[
                    ResNetBlock(layer_in_channels, temb_dim=temb_dim)
                    for idx in range(blocks_down[i] - 1)
                ],
            ]

            # down_layer = nn.ModuleDict({f"resnet_{i}x{blocks_down[i]}": nn.Sequential(*down_layer)})
            if jit_compile:
                down_layers = MultiSequential(*list(map(torch.jit.script, down_layers)))
            else:
                down_layers = MultiSequential(*down_layers)

            # down_blocks.append(down_layers)
            down_blocks[f"resnet_{i}x{blocks_down[i]}"] = down_layers

        return down_blocks

    # TODO: Add jit compile option
    #      May also bet better to build a ModuleDict
    def _make_up_layers(self, ResNetBlock, jit_compile=False):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm, temb_dim = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
            self.time_embedding_sz * 4,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_conv_block = [
                *[
                    ResNetBlock(
                        sample_in_channels // 2,
                        temb_dim=temb_dim,
                        # attention=self.self_attention,
                    )
                    for _ in range(blocks_up[i])
                ]
            ]

            if jit_compile:
                up_conv_block = MultiSequential(
                    *list(map(torch.jit.script, up_conv_block))
                )
            else:
                up_conv_block = MultiSequential(*up_conv_block)

            upsample_block = nn.Sequential(
                *[
                    self.conv_layer(
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
            if jit_compile:
                upsample_block = torch.jit.script(upsample_block)

            up_layers.append(up_conv_block)
            up_samples.append(upsample_block)

        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(
                name=self.norm,
                spatial_dims=self.spatial_dims,
                channels=self.init_filters,
            ),
            # self.act,
            self.conv_layer(
                self.spatial_dims,
                self.init_filters,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x, time_cond):
        if self.resblock_pp and not self.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.0
        # print("Data shape:", x.shape)
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        if self.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = torch.log(used_sigmas)
        elif self.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            # TODO: Calculate sigmas
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.time_embedding_sz // 2)

        t = self.time_embed_layer(temb)

        for i, down_block in enumerate(self.down_layers.values()):
            # for down_block in blocks:
            x = down_block(x, t)
            # print(f"Computed down-layer {i}: {x.shape}")
            down_x.append(x)

        if self.self_attention:
            if self.attention_type == "block":
                x = self.pre_attention(x, t)

            x = self.attention_block(x)

            if self.attention_type == "block":
                x = self.post_attention(x, t)

        down_x.reverse()

        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x)
            # print(f"Computed up-sample {i}: {x.shape}")
            x = (x + down_x[i + 1]) / np.sqrt(2.0)
            x = upl(x, t)
            # print(f"Computed up-layer {i}: {x.shape}")

        if self.use_conv_final:
            x = self.conv_final(x)

        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            x = x / used_sigmas

        return x
