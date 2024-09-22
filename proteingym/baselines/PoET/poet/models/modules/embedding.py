from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    # rearrange doesn't work with torch.jit
    # x = rearrange(x, '... (d r) -> ... d r', r=2)
    x = x.unflatten(dim=-1, sizes=(-1, 2))
    x1, x2 = x.unbind(dim=-1)
    rotated_x = torch.stack((-x2, x1), dim=-1)
    # return rearrange(rotated_x, '... d r -> ... (d r)')
    return rotated_x.flatten(start_dim=-2)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin):
    # x is (b, s, h, d) or (bs, h, d) if packed
    # cos and sin are (s, d) or (bs, d)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(
        self,
        dim_model: int,
        scale: Optional[int] = None,
        force_fp32: Optional[bool] = None,
        *_,
        **__,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.scale = scale or 10_000
        self.force_fp32 = force_fp32 or False
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._get_inv_freq()
        if not force_fp32:
            self.register_buffer("inv_freq", inv_freq)
        else:
            self.inv_freq = inv_freq

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _get_inv_freq(self):
        r = (
            torch.div(torch.arange(self.dim_model), 2, rounding_mode="floor")
            * 2.0
            / self.dim_model
        )
        return 1.0 / (self.scale**r)

    def _update_cos_sin_tables(self, x, seq_dimension=-2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len
            t = torch.arange(
                x.shape[seq_dimension], device=x.device, dtype=self.inv_freq.dtype
            )
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self._cos_cached = torch.cos(freqs).to(x.dtype)
            self._sin_cached = torch.sin(freqs).to(x.dtype)

        return self._cos_cached, self._sin_cached

    def get_cos_sin_tables(self, t: torch.Tensor, dtype=torch.float32):
        # t is the tensor of indices

        # cast self.inv_freq to force computation in single precision
        # lower precision may not be able to represent all possible values of t
        self.inv_freq = self.inv_freq.to(t.device)
        freqs = torch.outer(t, self.inv_freq.float())
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_positions: Optional[torch.Tensor] = None,
        k_positions: Optional[torch.Tensor] = None,
        transform_q: bool = True,
        transform_k: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q and k are either (b, s, h, d)
        # or they are packed (bs, h, d)

        if transform_q:
            if q_positions is None:
                # in this case, q must be (b, s, ..., d)
                s = q.size(1)
                q_positions = torch.arange(s, device=q.device)
            cos, sin = self.get_cos_sin_tables(q_positions, q.dtype)
            # apply the rotary embedding to q
            q = apply_rotary_pos_emb(q, cos, sin)

        if transform_k:
            if k_positions is not q_positions or not transform_q:
                # need to compute new cos, sin for k positions
                if k_positions is None:
                    s = k.size(1)
                    k_positions = torch.arange(s, device=k.device)
                cos, sin = self.get_cos_sin_tables(k_positions, k.dtype)
            # apply the rotary embedding to k
            k = apply_rotary_pos_emb(k, cos, sin)

        return q, k


"""
class RotaryEmbedding2D(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0
        self.rotary_emb1d = RotaryEmbedding(dim // 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dimension=-2):
        assert seq_dimension in [-2, -3]  # Either (bs, h, s, d) or (bs, s, h, d)
        seqlen = q.shape[seq_dimension]
        seqlen_sqrt = int(math.sqrt(seqlen))
        assert seqlen == seqlen_sqrt ** 2
        if seq_dimension == -3:  # (bs, s, h, d)
            q = rearrange(q, 'b s h d -> b h s d')
            k = rearrange(k, 'b s h d -> b h s d')
        q0, q1 = q.chunk(2, dim=-1)
        k0, k1 = k.chunk(2, dim=-1)
        # (bs, h, s, d)
        q0 = rearrange(q0, 'b nheads (h w) d -> b nheads h w d', h=seqlen_sqrt)
        k0 = rearrange(k0, 'b nheads (h w) d -> b nheads h w d', h=seqlen_sqrt)
        q0_emb, k0_emb = self.rotary_emb1d(q0, k0, seq_dimension=-2)
        q0_emb = rearrange(q0_emb, 'b nheads h w d -> b nheads (h w) d')
        k0_emb = rearrange(k0_emb, 'b nheads h w d -> b nheads (h w) d')
        q1 = rearrange(q1, 'b nheads (h w) d -> b nheads h w d', h=seqlen_sqrt)
        k1 = rearrange(k1, 'b nheads (h w) d -> b nheads h w d', h=seqlen_sqrt)
        q1_emb, k1_emb = self.rotary_emb1d(q1, k1, seq_dimension=-3)
        q1_emb = rearrange(q1_emb, 'b nheads h w d -> b nheads (h w) d')
        k1_emb = rearrange(k1_emb, 'b nheads h w d -> b nheads (h w) d')
        q_emb, k_emb = torch.cat([q0_emb, q1_emb], dim=-1), torch.cat([k0_emb, k1_emb], dim=-1)
        if seq_dimension == -3:
            q_emb = rearrange(q_emb, 'b h s d -> b s h d')
            k_emb = rearrange(k_emb, 'b h s d -> b s h d')
        return q_emb, k_emb
"""


class RelativePositionEmbedding(nn.Module):
    def __init__(self, embedding_dim, window_size):
        super().__init__()
        self.embed = nn.Embedding(2 * window_size + 1, embedding_dim)
        self.window_size = window_size

    def forward(self, x, start_index=None, chain_index=None):
        """
        Input is shape (batch, length, ...)
        """
        index = torch.arange(x.size(1), device=x.device)
        if chain_index is not None:
            index = index + self.window_size * chain_index
        b = x.size(0)

        rel_dist = index.unsqueeze(1) - index.unsqueeze(0)
        rel_dist = torch.clamp(rel_dist, min=-self.window_size, max=self.window_size)
        rel_dist = rel_dist + self.window_size

        z = self.embed(rel_dist)
        z = z.unsqueeze(0).expand(b, z.size(0), z.size(1), z.size(2))

        return z


class SinCosEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_length=2048, linear=False):
        super(SinCosEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        index = torch.arange(embedding_dim).float()
        weight = 1 / (max_length ** (2 * index / embedding_dim))
        bias = torch.zeros(embedding_dim)
        bias[::2] = np.pi / 2
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        if linear:
            self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, start_index=None):
        b, n = x.size()[:2]

        pos = torch.arange(n).to(x.device).float()
        if start_index is not None:
            pos = pos.unsqueeze(0) + start_index.float().unsqueeze(1)
            z = torch.cos(pos.unsqueeze(2) * self.weight + self.bias)  # NxD
        else:
            z = torch.cos(pos.unsqueeze(1) * self.weight + self.bias)  # NxD
            z = z.unsqueeze(0)
        if hasattr(self, "linear"):
            z = self.linear(z)
        z = z.expand(b, n, z.size(2))

        return z


class RandomFourierEmbedding(nn.Module):
    def __init__(self, embedding_dim, sigma=2, linear=False):
        super(RandomFourierEmbedding, self).__init__()

        self.embedding_dim = embedding_dim

        w = torch.randn(embedding_dim) / sigma
        b = torch.rand(embedding_dim) * 2 * np.pi
        self.register_buffer("weight", w)
        self.register_buffer("bias", b)

        if linear:
            self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, start_index=None):
        b, n = x.size()[:2]

        pos = torch.arange(n).to(x.device).float()
        if start_index is not None:
            pos = pos.unsqueeze(0) + start_index.float().unsqueeze(1)
            z = torch.cos(pos.unsqueeze(2) * self.weight + self.bias)  # NxD
        else:
            z = torch.cos(pos.unsqueeze(1) * self.weight + self.bias)  # NxD
            z = z.unsqueeze(0)
        if hasattr(self, "linear"):
            z = self.linear(z)
        z = z.expand(b, n, z.size(2))

        return z


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_length=2048):
        super(PositionalEmbedding, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        index = torch.arange(embedding_dim).float()
        weight = 1 / (max_length ** (2 * index / embedding_dim))
        bias = torch.zeros(embedding_dim)
        bias[::2] = np.pi / 2

        pos = torch.arange(max_length).float()
        z = torch.cos(pos.unsqueeze(1) * weight + bias)  # NxD
        self.embed = nn.Embedding.from_pretrained(z, freeze=False)

    def forward(self, x, start_index=None):
        b, n = x.size()[:2]
        pos = torch.arange(n).to(x.device)
        if start_index is not None:
            pos = pos.unsqueeze(0) + start_index.unsqueeze(1)
            z = self.embed(pos)
        else:
            z = self.embed(pos).unsqueeze(0)  # NxD
        z = z.expand(b, n, z.size(1))
        return z


class GaussianDistEmbedding(nn.Module):
    def __init__(self, n_out, sigma=16):
        super().__init__()
        self.sigma = sigma
        self.linear = nn.Linear(1, n_out, bias=False)

    def forward(self, x):
        if x is None:
            return 0

        dist = torch.cdist(x, x)
        kernel = torch.exp(-(dist**2) / 2 / self.sigma**2)
        isnan = torch.isnan(kernel)
        kernel = torch.where(isnan, torch.zeros_like(kernel), kernel)
        kernel = kernel.unsqueeze(3)
        return self.linear(kernel)


class FourierCoordsEmbed(nn.Module):
    def __init__(self, in_dim, embedding_dim, sigma=12, linear=False):
        super(FourierCoordsEmbed, self).__init__()

        self.embedding_dim = embedding_dim

        w = torch.randn(in_dim, embedding_dim) / sigma / np.sqrt(in_dim)
        b = torch.rand(embedding_dim) * 2 * np.pi
        self.register_buffer("weight", w)
        self.register_buffer("bias", b)

        if linear:
            self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.linear.weight.data.zero_()

    def forward(self, x):
        if x is None:
            return 0
        mask = torch.isnan(x).any(2)

        z = torch.cos(torch.mm(x.view(-1, x.size(2)), self.weight) + self.bias)
        z = z.view(x.size(0), x.size(1), z.size(1))
        z[mask] = 0

        if hasattr(self, "linear"):
            z = self.linear(z)

        return z


class LocalContextEmbed(nn.Module):
    def __init__(self, window_size, embedding_dim, lengthscale=6):
        super(LocalContextEmbed, self).__init__()

        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.lengthscale = lengthscale

        self.linear = nn.Linear(window_size, embedding_dim, bias=False)
        self.linear.weight.data.zero_()

    def forward(self, x):
        if x is None:
            return 0

        # unfold x into windows
        x = x.transpose(1, 2)
        # X is B x 3 x L
        # windows is B x 3*W x L
        k = self.window_size // 2
        x_pad = F.pad(x, (k, k), "constant", np.nan)
        windows = F.unfold(x_pad.unsqueeze(3), (self.window_size, 1))
        # reshape to B x 3 x W x L
        windows = windows.reshape(x.size(0), x.size(1), self.window_size, x.size(2))

        dist = torch.sum((x.unsqueeze(2) - windows) ** 2, 1)  # B x W x L
        kernel = torch.exp(-dist / self.lengthscale**2)

        kernel = kernel.transpose(1, 2)  # B x L x W
        mask = torch.isnan(kernel)
        kernel = torch.where(mask, torch.zeros_like(kernel), kernel)
        return self.linear(kernel)


class AngleEmbed(nn.Module):
    def __init__(self, in_dim, embedding_dim, bias=False):
        super(AngleEmbed, self).__init__()
        self.linear = nn.Linear(2 * in_dim, embedding_dim, bias=bias)
        # self.linear.weight.data.zero_()

    def forward(self, x):
        if x is None:
            return 0
        # expand x into sin and cos of angles
        sin = torch.sin(x)
        cos = torch.cos(x)
        x = torch.cat([sin, cos], axis=2)

        mask = torch.isnan(x)
        x = torch.where(mask, torch.zeros_like(x), x)
        return self.linear(x)
