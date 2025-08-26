import contextlib
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn.layers.rotary import apply_rotary_emb

    FLASH_ROTARY_AVAILABLE = False
except ModuleNotFoundError:
    FLASH_ROTARY_AVAILABLE = False


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


def apply_rotary_pos_emb_flash(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return apply_rotary_emb(
        x=x,
        cos=cos,
        sin=sin,
        interleaved=True,
        cu_seqlens=torch.tensor([0, x.size(0)], device=x.device),
        max_seqlen=x.size(0),
    )


_USE_COS_SIN_CACHE: bool = False
_COS_SIN_CACHE: dict[tuple[int, torch.dtype], torch.Tensor] = {}


@contextlib.contextmanager
def rotary_cos_sin_cache():
    global _USE_COS_SIN_CACHE
    try:
        _USE_COS_SIN_CACHE = True
        yield
    finally:
        _USE_COS_SIN_CACHE = False
        _COS_SIN_CACHE.clear()


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
        # default is true if force_fp32 and false otherwise
        use_flash: Optional[bool] = None,
        *_,
        **__,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.scale = scale or 10_000
        self.force_fp32 = force_fp32 or False
        self._allow_flash = (
            use_flash if use_flash is not None else self.force_fp32
        ) and FLASH_ROTARY_AVAILABLE
        self.use_flash = self._allow_flash

    @property
    def use_flash(self):
        return self._use_flash

    @use_flash.setter
    def use_flash(self, use_flash: bool):
        if hasattr(self, "_use_flash") and use_flash is self._use_flash:
            return
        self._use_flash = use_flash

        # Generate and save the inverse frequency buffer (non trainable)
        original_inv_freq_dtype = (
            None if not hasattr(self, "inv_freq") else self.inv_freq.dtype
        )
        inv_freq = self._get_inv_freq(use_flash)
        if not self.force_fp32:
            self.register_buffer("inv_freq", inv_freq)
        else:
            self.inv_freq = inv_freq
        if original_inv_freq_dtype is not None:
            self.inv_freq = self.inv_freq.to(original_inv_freq_dtype)

    @property
    def apply_rotary_pos_emb(self):
        if not self.use_flash:
            return apply_rotary_pos_emb
        else:
            return apply_rotary_pos_emb_flash

    def _get_inv_freq(self, use_flash: bool):
        if not use_flash:
            r = (
                torch.div(torch.arange(self.dim_model), 2, rounding_mode="floor")
                * 2.0
                / self.dim_model
            )
            return 1.0 / (self.scale**r)
        else:
            return 1.0 / (
                self.scale
                ** (
                    torch.arange(0, self.dim_model, 2, dtype=torch.float32)
                    / self.dim_model
                )
            )

    def get_cos_sin_tables(self, t: torch.Tensor, dtype=torch.float32):
        key = (t, dtype)
        if _USE_COS_SIN_CACHE and key in _COS_SIN_CACHE:
            return _COS_SIN_CACHE[key]
        # t is the tensor of indices

        # cast self.inv_freq to force computation in single precision
        # lower precision may not be able to represent all possible values of t
        self.inv_freq = self.inv_freq.to(t.device)
        freqs = torch.outer(t, self.inv_freq.float())
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        if _USE_COS_SIN_CACHE:
            _COS_SIN_CACHE[key] = (cos, sin)
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
        # NOTE: if any external caller sets use_flash, this may change it...
        device_type = k.device.type if k is not None else q.device.type
        if device_type == "cpu":
            self.use_flash = False
        else:
            self.use_flash = self._allow_flash

        if transform_q:
            if q_positions is None:
                # in this case, q must be (b, s, ..., d)
                s = q.size(1)
                q_positions = torch.arange(s, device=q.device)
            cos, sin = self.get_cos_sin_tables(q_positions, q.dtype)
            # apply the rotary embedding to q
            q = self.apply_rotary_pos_emb(q, cos, sin)

        if transform_k:
            if k_positions is not q_positions or not transform_q:
                # need to compute new cos, sin for k positions
                if k_positions is None:
                    s = k.size(1)
                    k_positions = torch.arange(s, device=k.device)
                cos, sin = self.get_cos_sin_tables(k_positions, k.dtype)
            # apply the rotary embedding to k
            k = self.apply_rotary_pos_emb(k, cos, sin)

        return q, k
