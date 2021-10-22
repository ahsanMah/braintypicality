import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layerspp import make_dense_layer, GaussianFourierProjection
from . import utils

MAX_DEPTH = 3


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, padding=1):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(
            in_chan, out_chan, kernel_size=3, padding=padding, bias=True
        )
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == "relu":
            self.activation = nn.ReLU()
        elif act == "prelu":
            self.activation = nn.PReLU(out_chan)
        elif act == "elu":
            self.activation = nn.ELU()
        elif act == "swish":
            self.activation = nn.SiLU()
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):

    if double_chnnel:
        out_channel_1 = 32 * (2 ** (depth + 1))
        layer1 = LUConv(in_channel, out_channel_1, act)

        out_channel_2 = 32 * (2 ** (depth + 1))
        layer2 = LUConv(out_channel_1, out_channel_2, act)
    else:
        out_channel_1 = 32 * (2 ** depth)
        layer1 = LUConv(in_channel, out_channel_1, act)

        out_channel_2 = 32 * (2 ** depth) * 2
        layer2 = LUConv(out_channel_1, out_channel_2, act)

    return nn.Sequential(layer1, layer2), out_channel_2


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act, temb_dim):
        super(DownTransition, self).__init__()
        self.ops, out_channels = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

        if temb_dim is not None:
            self.dense = make_dense_layer(temb_dim, out_channels)

        self.act = nn.SiLU()

    def forward(self, x, t=None):

        if self.current_depth == MAX_DEPTH:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)

        if t is not None:
            b = self.dense(t)
            out += b[:, :, None, None, None]

        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act, temb_dim):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops, out_channels = _make_nConv(
            inChans + outChans // 2, depth, act, double_chnnel=True
        )

        if temb_dim is not None:
            self.dense = make_dense_layer(temb_dim, out_channels)

    def forward(self, x, skip_x, t=None):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)

        if t is not None:
            b = self.dense(t)
            out += b[:, :, None, None, None]

        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv_pp = nn.Conv3d(inChans, n_labels, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # out = self.sigmoid(self.final_conv(x))
        out = self.final_conv_pp(x)
        return out


@utils.register_model(name="models_genesis_pp")
class UNet3Dpp(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, config, n_channels=1, act="relu", temb_dim=None):
        super(UNet3Dpp, self).__init__()

        temb_dim = config.model.time_embedding_sz
        n_channels = config.data.num_channels
        act = config.model.nonlinearity
        fourier_scale = config.model.fourier_scale

        self.init_conv = nn.Conv3d(n_channels, 4, 3, padding="same")
        self.init_pool = nn.Conv3d(4, 1, 1)
        self.time_embed_layer = self._make_time_cond_layers(temb_dim, fourier_scale)

        self.down_tr64 = DownTransition(1, 0, act, temb_dim)
        self.down_tr128 = DownTransition(64, 1, act, temb_dim)
        self.down_tr256 = DownTransition(128, 2, act, temb_dim)
        self.down_tr512 = DownTransition(256, 3, act, temb_dim)

        self.up_tr256 = UpTransition(512, 512, 2, act, temb_dim)
        self.up_tr128 = UpTransition(256, 256, 1, act, temb_dim)
        self.up_tr64 = UpTransition(128, 128, 0, act, temb_dim)
        self.out_tr = OutputTransition(64, n_channels)

    def _make_time_cond_layers(self, sz, scale):

        projection = GaussianFourierProjection(embedding_size=sz // 4, scale=scale)
        # Projection layer doubles the input_sz
        # Since it concats sin and cos projections
        dense_0 = make_dense_layer(sz // 2, sz)
        dense_1 = make_dense_layer(sz, sz)
        return nn.Sequential(projection, dense_0, dense_1)

    def forward(self, x, t):
        # print(x.shape)

        x = self.init_conv(x)
        x = self.init_pool(x)
        if t is not None:
            t = self.time_embed_layer(t)

        self.out64, self.skip_out64 = self.down_tr64(x, t)
        self.out128, self.skip_out128 = self.down_tr128(self.out64, t)
        self.out256, self.skip_out256 = self.down_tr256(self.out128, t)
        self.out512, self.skip_out512 = self.down_tr512(self.out256, t)

        # print(
        #     self.out256.shape, self.out128.shape, self.out64.shape,
        # )

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256, t)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128, t)

        # self.out_up_128 = self.up_tr128(self.out256, self.skip_out128, t)

        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64, t)
        self.out = self.out_tr(self.out_up_64)

        return self.out
