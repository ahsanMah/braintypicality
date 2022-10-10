# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Sequence, Tuple, Union, Any

from monai.networks.blocks.segresnet_block import ResBlock
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.factories import Act
from monai.networks.layers.utils import get_norm_layer
from monai.utils import InterpolateMode, UpsampleMode

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init
AttentionBlock = layers.AttentionBlock


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        # print("x:", x.shape, "x_proj:", x_proj.shape)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


########## Code for 3D brain reconstruction models ##############


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    bias: bool = True,
    dilation: int = 1,
    init_scale=1.0,
):
    conv = Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        bias=bias,
        conv_only=True,
        dilation=dilation,
    )

    conv.conv.weight.data = default_init(init_scale)(conv.conv.weight.data.shape)
    nn.init.zeros_(conv.conv.bias)

    return conv


def get_upsample_layer(
    spatial_dims: int,
    in_channels: int,
    upsample_mode: Union[UpsampleMode, str] = "nontrainable",
    scale_factor: int = 2,
):
    return UpSample(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )


def make_dense_layer(in_sz, out_sz):
    dense = nn.Linear(in_sz, out_sz)
    dense.weight.data = default_init()(dense.weight.shape)
    nn.init.zeros_(dense.bias)
    return dense


class ResBlockpp(nn.Module):
    """
    Modified ResBlock form Monai implementation
    [LINK]
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: Union[Tuple, str],
        act: str = "swish",
        kernel_size: int = 3,
        dilation: int = 1,
        init_scale=0.0,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            dilation: dilation size of kernel for larger receptive fields
            init_scale: variance scale used to initialize kernel params via ddpm initializer
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(
            name=norm, spatial_dims=spatial_dims, channels=in_channels
        )
        self.norm2 = get_norm_layer(
            name=norm, spatial_dims=spatial_dims, channels=in_channels
        )
        # self.act = Act["mish"](inplace=True)
        self.act = Act[act]()

        # Following convention of the BigGAN blocks above
        # first conv uses default init scale, second uses config
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            dilation=dilation,
            kernel_size=kernel_size,
        )
        self.conv2 = get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            dilation=dilation,
            init_scale=init_scale,
            kernel_size=kernel_size,
        )

    def forward(self, x):

        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity
        # x /= torch.sqrt(torch.tensor(2.0, requires_grad=False))

        return x


class SegResBlockpp(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        in_channels: int,
        norm: Union[Tuple, str],
        spatial_dims: int = 3,
        act: str = "swish",
        kernel_size: int = 3,
        temb_dim: int = None,
        pre_conv: Any = None,
        attention_heads: int = 0,
        dilation: int = 1,
        resblock_pp: bool = False,
        jit: bool = False,
    ) -> None:

        super().__init__()

        self.pre_conv = pre_conv
        self.n_channels = in_channels

        if not resblock_pp:
            self.resblock = ResBlock(
                spatial_dims, in_channels, norm, kernel_size=kernel_size
            )
        else:
            self.resblock = ResBlockpp(
                spatial_dims,
                in_channels,
                norm,
                dilation=dilation,
                kernel_size=kernel_size,
                act=act,
            )

        if jit:
            # print("Jitting resblock")
            self.resblock = torch.jit.script(self.resblock)
        self.attention = attention_heads

        if temb_dim is not None:
            self.dense = make_dense_layer(temb_dim, in_channels * 2)

        self.act = nn.SiLU()

        if attention_heads > 0:
            self.attn = AttentionBlock(channels=in_channels, num_heads=attention_heads)

    def forward(self, x, temb=None):

        if self.pre_conv is not None:
            x = self.pre_conv(x)

        x = self.resblock(x)

        # If time embedding provided
        # Conditioning is acheived via time embedding
        if temb is not None:
            cond_info = self.dense(self.act(temb))[:, :, None, None, None]
            gamma, beta = torch.split(
                cond_info, (self.n_channels, self.n_channels), dim=1
            )
            x = x * gamma + beta

        if self.attention:
            x = self.attn(x)

        return x


class ResnetBlockBigGANpp(nn.Module):
    """
    BigGAN block adapted from song. Notably, the conditioning is done b/w convs
    norm - act - conv --> time_cond --> norm - act - drop - conv --> skip
    """

    def __init__(
        self,
        in_channels,
        kernel_size: int = 3,
        spatial_dims: int = 3,
        act: str = "swish",
        temb_dim: int = None,
        dropout: float = 0.0,
        init_scale: float = 0.0,
        pre_conv: Any = None,
        attention_heads: int = None,  # FIXME: Ignored for now, but this should be used in NCSNpp class
    ):
        super().__init__()

        self.n_channels = in_channels
        self.pre_conv = pre_conv

        # print("IN_CHANNELS:", in_channels)
        self.norm_0 = get_norm_layer(
            name=("GROUP", {"num_groups": min(in_channels // 4, 32), "eps": 1e-6}),
            spatial_dims=spatial_dims,
            channels=in_channels,
        )

        self.conv_0 = get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )

        if temb_dim is not None:
            self.dense = make_dense_layer(temb_dim, in_channels * 2)

        self.norm_1 = get_norm_layer(
            name=("GROUP", {"num_groups": min(in_channels // 4, 32), "eps": 1e-6}),
            spatial_dims=spatial_dims,
            channels=in_channels,
        )
        self.conv_1 = get_conv_layer(
            spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            init_scale=init_scale,
            kernel_size=kernel_size,
        )
        self.dropout = nn.Dropout(dropout)

        self.act = Act[act]()

    def forward(self, x, temb=None):

        if self.pre_conv is not None:
            x = self.pre_conv(x)

        h = self.act(self.norm_0(x))

        h = self.conv_0(h)
        # FiLM-like conditioning for each feature map via time embedding
        if temb is not None:
            cond_info = self.dense(self.act(temb))[:, :, None, None, None]
            gamma, beta = torch.split(
                cond_info, (self.n_channels, self.n_channels), dim=1
            )
            h = h * gamma + beta

        h = self.act(self.norm_1(h))
        h = self.dropout(h)
        h = self.conv_1(h)

        return x + h
