import torch
import torch.nn as nn
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from functools import partial
from einops import rearrange, repeat


def gaussian_logprob(z, ldj):
    _GCONST_ = -0.9189385332046727  # ln(sqrt(2*pi))
    return _GCONST_ - 0.5 * torch.sum(z**2, dim=-1) + ldj


def subnet_fc(c_in, c_out, ndim=256, act=nn.LeakyReLU, input_norm=True):
    return nn.Sequential(
        nn.LayerNorm(c_in) if input_norm else nn.Identity(),
        nn.Linear(c_in, ndim),
        act(),
        nn.Linear(ndim, ndim),
        act(),
        nn.LayerNorm(ndim),
        nn.Linear(ndim, c_out),
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
        self.register_buffer("cached_penc", None)

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
          dim: Number of dimensions of each Gaussian
          loc: List of mean values
          scale: List of diagonals of the covariance matrices
          weights: List of mode probabilities
          trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = n_features
        self.split_sizes = [
            self.n_modes,
            self.n_modes * self.dim,
            self.n_modes * self.dim,
        ]
        self.context_encoder = subnet_fc(
            context_size,
            sum(self.split_sizes),
            ndim=context_size * 2,
            act=nn.GELU,
            input_norm=False,
        )

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
        w, loc, log_scale = torch.split(encoder_output, self.split_sizes, dim=1)
        loc = loc.reshape(loc.shape[0], self.n_modes, self.dim)
        log_scale = log_scale.reshape(loc.shape[0], self.n_modes, self.dim)

        # Get weights
        weights = torch.softmax(w, 1)

        # Compute log probability
        eps = (z[:, None, :] - loc) / torch.exp(log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(log_scale, 2)
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
                "stride": 2,
            },
            "global": {"kernel_size": 7, "padding": 2, "stride": 4},
        },
        7: {
            "local": {
                "kernel_size": 7,
                "padding": 2,
                "stride": 4,
            },  # factor 4 downsampling
            "global": {"kernel_size": 17, "padding": 4, "stride": 6},  # factor 8
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
        embed_dim=128,
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
            self.global_pooler = SpatialNorm3D(
                channels, **PatchFlow.patch_config[patch_size]["global"]
            ).requires_grad_(False)
            # Spatial resolution of the global context patches
            _, c, h, w, d = self.global_pooler(torch.empty(1, *input_size)).shape
            print("Global Context Shape: ", (c, h, w))
            self.global_attention = ScoreAttentionBlock(
                input_size=(c, h, w, d),
                embed_dim=embed_dim,
                outdim=context_embedding_size,
            )
            context_dims += context_embedding_size

        num_features = self.channels
        self.flow = self.build_cflow_head(context_dims, num_features, num_blocks)

        if gmm_components > 0:
            self.gmm = ConditionalGaussianMixture(
                gmm_components, num_features, context_dims
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
                subnet_constructor=partial(subnet_fc, act=nn.GELU),
                global_affine_type="SOFTPLUS",
                permute_soft=True,
                affine_clamping=1.9,
            )

        return coder

    def forward(self, x, return_attn=False, fast=True):
        B, C = x.shape[0], x.shape[1]
        x_norm = self.local_pooler(x)
        context = self.position_encoder(x_norm)

        if self.use_global_context:
            global_pooled_image = self.global_pooler(x)
            global_context = self.global_attention(global_pooled_image)
            # Concatenate global context to local context
            # Every patch gets the same global context
            b, c, h, w, d = context.shape
            global_context = repeat(global_context, "b c -> b c n", n=self.num_patches)
            global_context = rearrange(
                global_context, "b c (h w d) -> b c h w d", h=h, w=w, d=d
            )
            context = torch.cat([context, global_context], dim=1)

        if fast:
            local_patches = rearrange(
                x_norm, "b c h w d -> (h w d b) c"
            )  # (Patches * batch) x channels
            context = rearrange(context, "b c h w d -> (h w d b) c")
            zs, log_jac_dets = self.fast_forward(local_patches, context)

        else:
            local_patches = rearrange(
                x_norm, "b c h w d -> (h w d) b c"
            )  # Patches x batch x channels
            context = rearrange(context, "b c h w d -> (h w d) b c")

            zs = []
            log_jac_dets = []

            for patch_feature, context_vector in zip(local_patches, context):

                z, ldj = self.flow(
                    patch_feature,
                    c=[context_vector],
                )
                if self.gmm is not None:
                    z = self.gmm(z, context_vector)
                zs.append(z)
                log_jac_dets.append(ldj)

        if self.gmm is not None:
            zs = torch.cat(zs, dim=0).reshape(self.num_patches, B)
        else:
            zs = torch.cat(zs, dim=0).reshape(self.num_patches, B, C)
        log_jac_dets = torch.cat(log_jac_dets, dim=0).reshape(self.num_patches, B)

        if return_attn:
            return zs, log_jac_dets

        return zs, log_jac_dets

    def fast_forward(self, x, ctx):
        assert (
            self.num_patches % self.patch_batch_size == 0
        ), "Need patch batch size to be divisible by total number of patches"

        nchunks = self.num_patches // self.patch_batch_size
        x, ctx = x.chunk(nchunks, dim=0), ctx.chunk(nchunks, dim=0)
        zs, jacs = [], []

        for p, c in zip(x, ctx):
            # Check that patch context is same for all batch elements
            #             assert torch.isclose(c[0, :32], c[B-1, :32]).all()
            #             assert torch.isclose(c[B+1, :32], c[(2*B)-1, :32]).all()
            z, ldj = self.flow(p, c=[c])

            if self.gmm is not None:
                z = self.gmm(z, c)

            zs.append(z)
            jacs.append(ldj)

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
        logpx = rearrange(logpx, "(h w d b) 1 -> b h w d", b=b, h=h, w=w, d=d)

        return logpx

    @staticmethod
    def stochastic_train_step(flow, x, opt, n_patches=1):
        B, C, _, _, _ = x.shape
        h = flow.local_pooler(x)
        local_patches = rearrange(
            h, "b c h w d -> (h w d) b c"
        )  # Patches x batch x channels
        context = rearrange(flow.position_encoder(h), "b c h w d -> (h w d) b c")

        rand_idx = torch.randperm(flow.num_patches)[:n_patches]
        local_loss = 0.0
        for idx in rand_idx:
            patch_feature, context_vector = (
                local_patches[idx],
                context[idx],
            )

            if flow.use_global_context:
                # Need separate loss for each patch
                global_patches = (
                    flow.global_pooler(x).reshape(B, C, -1).permute(0, 2, 1)
                )
                global_context = flow.global_attention(global_patches)
                context_vector = torch.cat([context_vector, global_context], dim=-1)

            z, ldj = flow.flow(
                patch_feature,
                c=[context_vector],
            )
            if flow.gmm is not None:
                z = flow.gmm(z, context_vector)

            opt.zero_grad(set_to_none=True)
            loss = flow.nll(z, ldj)
            loss.backward()

            opt.step()
            local_loss += loss.item()

        return local_loss
