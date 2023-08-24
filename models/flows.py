from functools import partial

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat

from models.layerspp import ResBlockpp, get_conv_layer


def gaussian_logprob(z, ldj):
    _GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))
    return _GCONST_ - 0.5 * torch.sum(z**2, dim=-1) + ldj


def subnet_fc(c_in, c_out, ndim=256, act=nn.LeakyReLU(), input_norm=True):
    return nn.Sequential(
        nn.LayerNorm(c_in) if input_norm else nn.Identity(),
        nn.Linear(c_in, ndim),
        act,
        nn.LayerNorm(ndim),
        nn.Linear(ndim, c_out),
        act,
        # nn.Linear(ndim, c_out),
    )


# https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
class PositionalEncoding3D(nn.Module):
    def __init__(self, embedding_size):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.embedding_size = embedding_size
        channels = int(np.ceil(embedding_size / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, ch, x, y, z)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y, z)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, _, x, y, z = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = self.get_emb(sin_inp_y).unsqueeze(1)
        emb_z = self.get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, : self.embedding_size].repeat(
            batch_size, 1, 1, 1, 1
        )

        # Channels first
        self.cached_penc = self.cached_penc.permute(0, 4, 1, 2, 3)

        return self.cached_penc


class SpatialNorm3D(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size,
            # This is the real trick that ensures each
            # channel dimension is normed separately
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv.weight.data.fill_(1)  # all ones weights
        self.conv.weight.requires_grad = False  # freeze weights

    @torch.no_grad()
    def forward(self, x):
        return self.conv(x.square()).pow_(0.5)


class ScoreAttentionBlock(nn.Module):
    def __init__(self, input_size, embed_dim, outdim=None, num_heads=8, dropout=0.1):
        super().__init__()
        num_sigmas, h, w, d = input_size
        outdim = outdim or embed_dim

        self.spatial_size = (h, w, d)
        self.proj = nn.Linear(num_sigmas, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.conv_res_block = ResBlockpp(
            3, num_sigmas, norm=("layer", {"normalized_shape": self.spatial_size})
        )

        enc = PositionalEncoding3D(embed_dim)(torch.zeros(1, embed_dim, h, w, d))
        enc = rearrange(enc, "b c h w d -> b (h w d) c")
        self.register_buffer("position_encoding", enc)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, outdim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(outdim, outdim, bias=False),
        )
        self.normout = nn.LayerNorm(outdim)

    def forward(self, x, attn_mask=None):
        """
        Returns:
            x: Tensor of shape batch x 1 x out_dim
        """
        x = self.conv_res_block(x)
        x = rearrange(x, "b c h w d -> b (h w d) c")
        x = self.proj(x)
        x = self.norm(x.add_(self.position_encoding))

        h, _ = self.attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=False,
        )

        x = self.normout(self.ffn(x + h))

        # Spatial mean
        x = x.mean(dim=1)

        return x


class ConditionalGaussianMixture(nn.Module):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """

    def __init__(self, n_modes, n_features, context_size):
        """Constructor

        Args:
          n_modes: Number of modes of the mixture model
          n_features: Number of dimensions of each Gaussian
          context_size: Size of the conditioning vector
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = n_features
        self.split_sizes = [
            self.n_modes,
            self.n_modes * self.dim,
            # self.n_modes * self.dim,
        ]
        self.context_encoder = subnet_fc(
            context_size,
            sum(self.split_sizes),
            ndim=context_size,
            act=nn.LeakyReLU(0.2),
            input_norm=False,
        )

        self.log_scale = nn.Parameter(torch.zeros(1, self.n_modes, self.dim))
        # Initialize log scale
        nn.init.xavier_uniform_(self.log_scale.data)

    def sample(self, num_samples=1, context=None):
        encoder_output = self.context_encoder(context)
        w, loc, log_scale = torch.split(encoder_output, self.split_sizes, dim=1)
        loc = loc.reshape(loc.shape[0], self.n_modes, self.dim)
        log_scale = log_scale.reshape(loc.shape[0], self.n_modes, self.dim)

        # Get weights
        weights = torch.softmax(w, 1)

        # Sample mode indices
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]

        # Get samples
        eps_ = torch.randn(num_samples, self.dim, dtype=loc.dtype, device=loc.device)
        scale_sample = torch.sum(torch.exp(log_scale) * mode_1h, 1)
        loc_sample = torch.sum(loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample

        # Compute log probability
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def forward(self, z, context):
        return self.logprob(z, context)

    def logprob(self, z, context=None):
        encoder_output = self.context_encoder(context)
        w, loc = torch.split(encoder_output, self.split_sizes, dim=1)
        loc = loc.reshape(loc.shape[0], self.n_modes, self.dim)
        # log_scale = log_scale.reshape(loc.shape[0], self.n_modes, self.dim)

        # Get weights
        weights = torch.softmax(w, 1)

        # Compute log probability
        eps = (z[:, None, :] - loc) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return log_p


class PatchFlow(torch.nn.Module):
    """
    Contructs a conditional flow model that operates on patches of an image.
    Each patch is fed into the same flow model i.e. parameters are shared across patches.
    The flow models are conditioned on a positional encoding of the patch location.
    The resulting patch-densities can then be recombined into a full image density.
    """

    # Opinionated parameters for computing the spatial norms of the input scores
    patch_config = {
        3: {
            "local": {
                "kernel_size": 3,
                "padding": 1,
                "stride": 1,
            },
            "global": {"kernel_size": 11, "padding": 2, "stride": 4},
        },
        # factor 4 downsampling
        7: {
            "local": {
                "kernel_size": 7,
                "padding": 2,
                "stride": 4,
            },
            "global": {"kernel_size": 17, "padding": 4, "stride": 11},  # factor 8
        },
        # factor 16 downsampling
        17: {
            "local": {"kernel_size": 17, "padding": 4, "stride": 11},
            "global": {"kernel_size": 32, "padding": 4, "stride": 11},
        },
    }

    def __init__(
        self,
        input_size,
        patch_size=3,
        context_embedding_size=128,
        num_blocks=2,
        global_flow=False,
        patch_batch_size=128,
        global_embedding_dim=128,
        gmm_components=-1,
    ):
        super().__init__()
        assert (
            patch_size in PatchFlow.patch_config
        ), f"PatchFlow only support certain patch sizes: {PatchFlow.patch_config.keys()}"
        channels = input_size[0]

        # Used to chunk the input into in fast_forward (vectorized)
        self.patch_batch_size = patch_batch_size
        self.use_global_context = global_flow
        self.gmm = None
        self.context_embedding_size = context_embedding_size

        with torch.no_grad():
            # Pooling for local "patch" flow
            # Each patch-norm is input to the shared conditional flow model
            self.local_pooler = SpatialNorm3D(
                channels, **PatchFlow.patch_config[patch_size]["local"]
            ).requires_grad_(False)

            # Compute the spatial resolution of the patches
            _, self.channels, h, w, d = self.local_pooler(
                torch.empty(1, *input_size)
            ).shape
            self.spatial_res = (h, w, d)
            self.num_patches = h * w * d
            self.position_encoder = PositionalEncoding3D(context_embedding_size)
            print(
                f"Generating {patch_size}x{patch_size}x{patch_size} patches from input size: {input_size}"
            )
            print(f"Pooled spatial resolution: {self.spatial_res}")
            print(f"Number of flows / patches: {self.num_patches}")

        context_dims = context_embedding_size

        if self.use_global_context:
            # Pooling for global "low resolution" flow
            self.norm_pooler = SpatialNorm3D(
                channels, **PatchFlow.patch_config[patch_size]["global"]
            ).requires_grad_(False)
            self.conv_pooler = get_conv_layer(
                3, channels, channels, kernel_size=3, stride=2
            )
            self.global_pooler = nn.Sequential(
                self.norm_pooler,
                self.conv_pooler,
            )
            # Spatial resolution of the global context patches
            _, c, h, w, d = self.global_pooler(torch.empty(1, *input_size)).shape
            print("Global Context Shape: ", (c, h, w))
            self.global_attention = ScoreAttentionBlock(
                input_size=(c, h, w, d),
                embed_dim=global_embedding_dim,
                outdim=context_embedding_size,
            )
            context_dims += context_embedding_size

        num_features = self.channels
        self.flow = self.build_cflow_head(context_dims, num_features, num_blocks)

        if gmm_components > 0:
            self.gmm = ConditionalGaussianMixture(
                gmm_components, num_features, context_embedding_size
            )

    def init_weights(self):
        # Initialize weights with Xavier
        for m in self.flow.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # print(m)
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

        if self.use_global_context:
            for m in self.global_attention.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    # print(m)
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)

    def build_cflow_head(self, n_cond, n_feat, num_blocks=2):
        coder = Ff.SequenceINN(n_feat)
        for k in range(num_blocks):
            # idx = int(k % 2 == 0)
            coder.append(
                Fm.AllInOneBlock,
                cond=0,
                cond_shape=(n_cond,),
                subnet_constructor=partial(subnet_fc, act=nn.GELU()),
                global_affine_type="SOFTPLUS",
                permute_soft=True,
                affine_clamping=1.9,
            )

        return coder

    def forward(self, x, return_attn=False, fast=True):
        B, C = x.shape[0], x.shape[1]
        x_norm = self.local_pooler(x)
        self.position_encoder = self.position_encoder.cpu()
        context = self.position_encoder(x_norm)

        if self.use_global_context:
            global_pooled_image = self.global_pooler(x)
            # Every patch gets the same global context
            global_context = self.global_attention(global_pooled_image)
        if fast:
            zs, log_jac_dets = self.fast_forward(x_norm, context, global_context)

        else:
            # Patches x batch x channels
            local_patches = rearrange(x_norm, "b c h w d -> (h w d) b c")
            context = rearrange(context, "b c h w d -> (h w d) b c")

            zs = []
            log_jac_dets = []

            for patch_feature, context_vector in zip(local_patches, context):
                context_vector = context_vector.cuda()
                c = torch.cat([context_vector, global_context], dim=1)
                z, ldj = self.flow(
                    patch_feature,
                    c=[c],
                )
                if self.gmm is not None:
                    z = self.gmm(z, context_vector)
                zs.append(z)
                log_jac_dets.append(ldj)
                context_vector = context_vector.cpu()

        if self.gmm is not None:
            zs = torch.cat(zs, dim=0).reshape(self.num_patches, B)
        else:
            zs = torch.cat(zs, dim=0).reshape(self.num_patches, B, C)
        log_jac_dets = torch.cat(log_jac_dets, dim=0).reshape(self.num_patches, B)

        if return_attn:
            return zs, log_jac_dets

        return zs, log_jac_dets

    def fast_forward(self, x, local_ctx, global_ctx):
        # assert (
        #     self.num_patches % self.patch_batch_size == 0
        # ), "Need patch batch size to be divisible by total number of patches"

        # (Patches * batch) x channels
        local_ctx = rearrange(local_ctx, "b c h w d -> (h w d) b c")
        patches = rearrange(x, "b c h w d -> (h w d) b c")

        nchunks = self.num_patches // self.patch_batch_size
        nchunks += 1 if self.num_patches % self.patch_batch_size else 0

        patches = patches.chunk(nchunks, dim=0)
        ctx_chunks = local_ctx.chunk(nchunks, dim=0)
        zs, jacs = [], []
        # print(local_ctx.shape, global_ctx.shape)

        for p, ctx in zip(patches, ctx_chunks):
            # Check that patch context is same for all batch elements
            #             assert torch.isclose(c[0, :32], c[B-1, :32]).all()
            #             assert torch.isclose(c[B+1, :32], c[(2*B)-1, :32]).all()
            ctx = ctx.cuda()
            gc = repeat(global_ctx, "b c -> (n b) c", n=ctx.shape[0])
            ctx = rearrange(ctx, "n b c -> (n b) c")
            p = rearrange(p, "n b c -> (n b) c")

            c = torch.cat([ctx, gc], dim=1)
            # print(ctx.shape, gc.shape, c.shape)
            z, ldj = self.flow(p, c=[c])

            if self.gmm is not None:
                z = self.gmm(z, ctx)

            zs.append(z)
            jacs.append(ldj)

            ctx = ctx.cpu()

        return zs, jacs

    def logprob(self, zs, log_jac_dets):
        if self.gmm is not None:
            return zs + log_jac_dets
        return gaussian_logprob(zs, log_jac_dets)

    def nll(self, zs, log_jac_dets):
        return -torch.mean(self.logprob(zs, log_jac_dets))

    @torch.no_grad()
    def log_density(self, x, fast=True):
        self.eval()
        b = x.shape[0]
        h, w, d = self.spatial_res
        zs, jacs = self.forward(x, fast=fast)
        logpx = self.logprob(zs, jacs)
        logpx = rearrange(logpx, "(h w d) b -> b h w d", b=b, h=h, w=w, d=d)

        return logpx

    @staticmethod
    def stochastic_train_step(flow, x, opt, n_patches=1):
        flow.train()
        B, C, _, _, _ = x.shape
        h = flow.local_pooler(x)
        local_patches = rearrange(h, "b c h w d -> (h w d) b c")

        flow.position_encoder = flow.position_encoder.cpu()
        context = rearrange(flow.position_encoder(h), "b c h w d -> (h w d) b c")

        rand_idx = torch.randperm(flow.num_patches)[:n_patches]
        local_loss = 0.0
        for idx in rand_idx:
            patch_feature, context_vector = (
                local_patches[idx],
                context[idx],
            )
            context_vector = context_vector.cuda()
            if flow.use_global_context:
                # Need separate loss for each patch
                global_pooled_image = flow.global_pooler(x)
                global_context = flow.global_attention(global_pooled_image)
                # Concatenate global context to local context
                context_vector = torch.cat([context_vector, global_context], dim=1)

            z, ldj = flow.flow(
                patch_feature,
                c=[context_vector],
            )
            if flow.gmm is not None:
                z = flow.gmm(z, context_vector[:, : flow.context_embedding_size])

            opt.zero_grad(set_to_none=True)
            loss = flow.nll(z, ldj)
            loss.backward()

            opt.step()
            local_loss += loss.item()

        return {"train_loss": local_loss / n_patches}
