# coding=utf-8

from . import utils, layers, layerspp, normalization

from typing import Optional, Sequence, Tuple, Union

import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.networks.layers.factories import Act

from torchinfo import summary

get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
MultiSequential = layers.MultiSequential
AttentionBlock = layerspp.AttentionBlock3d
get_conv_layer_pp = layerspp.get_conv_layer
make_dense_layer = layerspp.make_dense_layer
get_upsample_layer = layerspp.get_upsample_layer


class ResnetBlockBigGANpp(nn.Module):
    """
    BigGAN block adapted from song. Notably, the conditioning is done b/w convs
    norm - act - conv --> time_cond --> norm - act - drop - conv --> skip
    """

    def __init__(
        self,
        in_channels,
        out_channels: int = None,
        kernel_size: int = 3,
        spatial_dims: int = 3,
        act: str = "swish",
        temb_dim: int = None,
        dropout: float = 0.0,
        init_scale: float = 0.0,
        downsample: bool = False,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.n_channels = out_channels
        self.downsample = downsample

        if downsample:
            self.pre_conv = layerspp.get_conv_layer(
                spatial_dims,
                in_channels=in_channels,  # Assumes inout is fewer channels
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
            )
            in_channels = out_channels

        # print("IN_CHANNELS:", in_channels)
        self.norm_0 = get_norm_layer(
            name=("GROUP", {"num_groups": min(in_channels // 4, 32), "eps": 1e-6}),
            spatial_dims=spatial_dims,
            channels=in_channels,
        )

        self.conv_0 = layerspp.get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        if temb_dim is not None:
            self.dense = make_dense_layer(temb_dim, out_channels * 2)

        self.norm_1 = get_norm_layer(
            name=("GROUP", {"num_groups": min(out_channels // 4, 32), "eps": 1e-6}),
            spatial_dims=spatial_dims,
            channels=out_channels,
        )
        self.conv_1 = layerspp.get_conv_layer(
            spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            init_scale=init_scale,
            kernel_size=kernel_size,
        )
        self.dropout = nn.Dropout(dropout)
        self.skip_conv = layerspp.get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

        self.act = Act[act]()

    def forward(self, x, temb):
        if self.downsample:
            x = self.pre_conv(x)

        h = self.act(self.norm_0(x))

        h = self.conv_0(h)

        # FiLM-like conditioning for each feature map via time embedding
        cond_info = self.dense(self.act(temb))[:, :, None, None, None]
        gamma, beta = torch.split(cond_info, (self.n_channels, self.n_channels), dim=1)
        h = h * (1 + gamma) + beta

        h = self.act(self.norm_1(h))
        h = self.dropout(h)
        h = self.conv_1(h)

        x = self.skip_conv(x) + h

        return x


@utils.register_model(name="finer_ncsnpp3d")
class FinerNCSN(nn.Module):
    """
    Time condioned version of SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.

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
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        use_conv_final: bool = True,
        upsample_mode: str = "nearest",
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

        assert len(self.blocks_down) == len(self.blocks_up)

        assert upsample_mode in ["nearest", "linear", "pixelshuffle"], ValueError(
            f"upsample_mode type {upsample_mode} not recognized."
        )

        assert embedding_type in ["fourier", "positional"]

        if upsample_mode != "pixelshuffle":
            interp_mode = upsample_mode
            upsample_mode = UpsampleMode.NONTRAINABLE

        self.act = config.model.act

        if config.model.dropout > 0.0:
            self.dropout_prob = config.model.dropout
        else:
            self.dropout_prob = None

        if self.dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](self.dropout_prob)

        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final

        self.conv_layer = functools.partial(
            get_conv_layer_pp,
            init_scale=config.model.init_scale,
            kernel_size=self.conv_size,
        )

        self.upsampler = functools.partial(
            get_upsample_layer,
            spatial_dims=spatial_dims,
            upsample_mode=upsample_mode,
            interp_mode=interp_mode,
        )

        self.convInit = self.conv_layer(
            spatial_dims, self.in_channels, self.init_filters
        )

        ResBlockpp = functools.partial(
            ResnetBlockBigGANpp,
            act=self.act,
            kernel_size=self.conv_size,
            spatial_dims=self.spatial_dims,
            temb_dim=self.time_embedding_sz * 4,
        )

        down_block_channels = [
            self.init_filters * 2 ** (i + 1) for i in range(len(self.blocks_down))
        ]
        up_block_channels = [
            self.init_filters * 2**i for i in range(len(self.blocks_up))
        ][::-1]

        self.down_layers = self._make_down_layers(ResBlockpp, jit_compile=self.compile)
        # self.down_layers = torch.jit.script(self.down_layers)

        self.mid_block = MultiSequential(
            ResBlockpp(
                down_block_channels[-1],
            ),
            ResBlockpp(
                down_block_channels[-1],
            ),
        )

        self.up_blocks = self._make_up_layers(
            ResBlockpp, down_block_channels, up_block_channels, jit_compile=self.compile
        )
        self.conv_final = self._make_final_conv(self.out_channels)
        self.time_embed_layer = self._make_time_cond_layers(embedding_type)

        # if self.self_attention:
        #     self.attention_block = AttentionBlock(
        #         channels=self.init_filters * 2 ** (len(self.blocks_down) - 1)
        #     )
        #     if self.compile:
        #         self.attention_block = torch.jit.script(self.attention_block)

    def _make_time_cond_layers(self, embedding_type):
        layer_list = []

        if embedding_type == "fourier":
            # Projection layer doubles the input_sz
            # Since it concats sin and cos projections
            projection = layerspp.GaussianFourierProjection(
                embedding_size=self.time_embedding_sz,
                scale=self.fourier_scale,
                learnable=True,
            )
            layer_list.append(projection)

        sz = self.time_embedding_sz * 2
        dense_0 = layerspp.make_dense_layer(sz, sz * 2)
        dense_1 = layerspp.make_dense_layer(sz * 2, sz * 2)

        layer_list.append(dense_0)
        layer_list.append(nn.SiLU())
        layer_list.append(dense_1)

        return nn.Sequential(*layer_list)

    def _make_down_layers(self, ResNetBlock, jit_compile=False):
        down_blocks = nn.ModuleDict()
        blocks_down, filters = self.blocks_down, self.init_filters

        for i in range(len(blocks_down)):
            layer_in_channels = filters * 2**i
            layer_out_channels = filters * 2 ** (i + 1)

            down_layers = [  # First layer needs the preconv
                ResNetBlock(
                    in_channels=layer_in_channels,
                    out_channels=layer_out_channels,
                    downsample=True,
                )
            ]

            for idx in range(1, blocks_down[i]):
                down_layers += [  # First layer needs the preconv
                    ResNetBlock(
                        in_channels=layer_out_channels,
                        out_channels=layer_out_channels,
                    )
                ]

            # down_layer = nn.ModuleDict({f"resnet_{i}x{blocks_down[i]}": nn.Sequential(*down_layer)})
            if jit_compile:
                down_layers = MultiSequential(*list(map(torch.jit.script, down_layers)))
            else:
                down_layers = MultiSequential(*down_layers)

            # down_blocks.append(down_layers)
            down_blocks[f"resnet_{i}x{blocks_down[i]}"] = down_layers

        return down_blocks

    def _make_up_layers(
        self, ResNetBlock, down_block_channels, up_block_channels, jit_compile=False
    ):
        blocks_up = self.blocks_up
        up_blocks = nn.ModuleDict()

        # Note that for the first upsampling block
        # The input channels are from the last down block + middle block
        middle_block_channels = down_block_channels[-1]

        for i in range(len(up_block_channels)):
            # First up block input is from middle block
            in_channels = up_block_channels[i - 1] if i > 0 else middle_block_channels
            # As we have incoming skip connections
            in_channels += down_block_channels[-i - 1]

            out_channels = up_block_channels[i]

            res_blocks = [
                ResNetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            ]

            # If there are more than 1 res blocks
            for j in range(1, blocks_up[i]):
                res_blocks += [
                    ResNetBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                    )
                ]

            res_blocks = MultiSequential(*res_blocks)

            upsampler = self.upsampler(in_channels=out_channels)

            up_blocks[f"up_block_{i}x{blocks_up[i]}"] = nn.ModuleList(
                [res_blocks, upsampler]
            )

        return up_blocks

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
                kernel_size=1,  # self.conv_size,
                bias=False,
            ),
        )

    def forward(self, x, time_cond):
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)

        down_x = []

        # Gaussian Fourier features embeddings.
        used_sigmas = time_cond
        temb = torch.log(used_sigmas)

        t = self.time_embed_layer(temb)

        for i, down_block in enumerate(self.down_layers.values()):
            # for down_block in blocks:
            x = down_block(x, t)
            # print(f"Computed down-layer {i}: {x.shape}")
            down_x.append(x)

        if self.mid_block is not None:
            x = self.mid_block(x, t)

        down_x.reverse()

        for i, (res_block, upsample) in enumerate(self.up_blocks.values()):
            x = torch.cat((x, down_x[i] / np.sqrt(2.0)), dim=1)
            x = res_block(x, t)
            # print(f"Computed up-layer {i}: {x.shape}")
            x = upsample(x)
            # print(f"Computed up-sample {i}: {x.shape}")

        if self.use_conv_final:
            x = self.conv_final(x)

        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            x = x / used_sigmas

        return x
