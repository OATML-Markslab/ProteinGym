import copy
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from poet.models.modules.attention import MultiheadAttention, mha_attn
from poet.models.modules.packed_sequence import PackedTensorSequences

# import flash attention module components and flag whether they are available
try:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as attn_func
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    except ImportError:
        from flash_attn.flash_attn_interface import (
            flash_attn_unpadded_func as attn_func,
        )
        from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    FLASH_ATTENTION_MODULE_INSTALLED = True
except ModuleNotFoundError as err:
    warnings.warn(
        "flash_attn module not found. Falling back on standard attention. " + str(err)
    )
    FLASH_ATTENTION_MODULE_INSTALLED = False


class FlashAttentionImpl(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None
    ):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def is_flash_attn_module_avail(self):
        return FLASH_ATTENTION_MODULE_INSTALLED

    def is_flash_attn_avail(
        self,
        qkv: PackedTensorSequences,
        attn_mask=None,
        causal=False,
        need_weights=False,
    ):
        """Check that the inputs, devices, and dtypes are compatible with flash attention and that flash attention is available."""
        # first, check for the flash attention module
        if not FLASH_ATTENTION_MODULE_INSTALLED:
            return False

        # the flash attention module is available, so check for compatability
        if need_weights:
            warnings.warn(
                "Returning attention weights requires standard attention algorithm."
            )
            return False

        if attn_mask is not None:
            warnings.warn("Passing attn_mask requires standard attention algorithm.")
            return False

        if not qkv.is_cuda:
            warnings.warn(
                "Flash attention requires CUDA device, not " + str(qkv.device)
            )
            return False

        return True

    def forward_flash_self_attn(
        self,
        q: PackedTensorSequences,
        k: PackedTensorSequences,
        v: PackedTensorSequences,
        attn_mask=None,
        causal=False,
        need_weights=False,
    ):
        """Use the flash attention implementation of multihead softmax attention."""
        assert not need_weights
        assert attn_mask is None

        # stack qkv
        # qkv = copy.copy(q)
        # qkv.x = torch.stack([q.x, k.x, v.x], dim=1)

        assert q.dtype == torch.float16 or q.dtype == torch.bfloat16
        assert q.is_cuda
        assert k.dtype == torch.float16 or k.dtype == torch.bfloat16
        assert k.is_cuda
        assert v.dtype == torch.float16 or v.dtype == torch.bfloat16
        assert v.is_cuda

        output = attn_func(
            q.x,
            k.x,
            v.x,
            q.cu_seqlens,
            k.cu_seqlens,
            q.max_s,
            k.max_s,
            self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )
        output_packed = copy.copy(q)
        output_packed.x = output
        return output_packed, None

    def forward_flash_attn(
        self,
        q: PackedTensorSequences,
        k: PackedTensorSequences,
        v: PackedTensorSequences,
        attn_mask=None,
        causal=False,
        need_weights=False,
    ):
        """Use the flash attention implementation of multihead softmax attention."""
        assert not need_weights
        assert attn_mask is None
        assert q.dtype == torch.float16 or q.dtype == torch.bfloat16
        assert q.is_cuda

        dropout = self.dropout_p if self.training else 0.0
        output = attn_func(
            q.x,
            k.x,
            v.x,
            q.cu_seqlens,
            k.cu_seqlens,
            q.max_s,
            k.max_s,
            dropout,
            softmax_scale=self.softmax_scale,
            causal=causal,
        )
        output_packed = copy.copy(q)
        output_packed.x = output
        return output_packed, None

    def forward_standard_attn(
        self,
        q: PackedTensorSequences,
        k: PackedTensorSequences,
        v: PackedTensorSequences,
        attn_mask=None,
        causal=False,
        need_weights=False,
    ):
        # need to unpack tensors...
        q, q_mask, positions = q.to_padded(return_mask=True, return_positions=True)
        k, key_padding_mask = k.to_padded(return_mask=True)
        v = v.to_padded()

        dropout = self.dropout_p if self.training else 0.0
        output, attn_weights = mha_attn(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            return_weights=need_weights,
            batch_first=True,
            scaling=self.softmax_scale,
            dropout=dropout,
            causal=causal,
        )
        # repack the output
        output_packed = PackedTensorSequences.pack_input(
            output, positions, key_padding_mask=q_mask
        )
        return output_packed, attn_weights

    def forward(
        self,
        q: PackedTensorSequences,
        k: PackedTensorSequences,
        v: PackedTensorSequences,
        attn_mask=None,
        causal=False,
        need_weights=False,
        self_attention=False,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensors containing the packed query, key, and values. (nnz, h, d)
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            key_padding_mask: An implementation of BaseMask that encodes how
                         many query each sequence in the batch consists of
        """
        # decide on whether we are using FlashAttn or standard Attn implementation
        dtype = q.dtype
        algo = self.forward_standard_attn
        if self.is_flash_attn_avail(
            q, attn_mask=attn_mask, causal=causal, need_weights=need_weights
        ):
            # we meet all the requirements, check the dtype
            if dtype != torch.float16 and dtype != torch.bfloat16:
                warnings.warn(
                    "Flash attention requires float16 or bfloat16 not "
                    + str(q.dtype)
                    + ". Converting to bfloat16."
                )
                q = copy.copy(q)
                k = copy.copy(k)
                v = copy.copy(v)
                q.x = q.x.to(torch.bfloat16)
                k.x = k.x.to(torch.bfloat16)
                v.x = v.x.to(torch.bfloat16)
            if self_attention:
                algo = self.forward_flash_self_attn
            else:
                algo = self.forward_flash_attn
        output, attn_weights = algo(
            q, k, v, attn_mask=attn_mask, causal=causal, need_weights=need_weights
        )
        output.x = output.x.to(dtype)

        return output, attn_weights


class FlashMultiheadAttention(MultiheadAttention):
    def __init__(
        self, *args, device=None, dtype=None, batch_first=True, **kwargs
    ) -> None:
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(*args, batch_first=batch_first, **kwargs)

        assert self.head_dim in [
            16,
            32,
            64,
            128,
        ], "Flash Attention only supports head_dim == 16, 32, 64, or 128"
        self.inner_attn = FlashAttentionImpl(
            attention_dropout=self.dropout, **factory_kwargs
        )

    def _inner_attn(
        self,
        q,
        k,
        v,
        key_padding_mask=None,
        attn_mask=None,
        return_weights=False,
    ):
        """
        Uses the Flash Attention module to calculate attention values efficiently.
        q, k, and v must be PackedTensorSequences.
        """

        # stacking is pointless and innefficient if flash attention module isn't going to be used
        # so push this choice inside the flash attention module
        # qkv = torch.stack([q.x, k.x, v.x], dim=1)
        # qkv_packed = copy.copy(q)
        # qkv_packed.x = qkv
        context_packed, attn_weights = self.inner_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            need_weights=return_weights,
            causal=self.causal,
            self_attention=self.self_attention,
        )
        return context_packed, attn_weights

    # no need to override forward packed
    # def forward_packed(...)

    def forward_padded(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Flash attention requires packed input, so pack and then unpack output
        if self.self_attention:
            query = PackedTensorSequences.pack_input(
                query, key_padding_mask=key_padding_mask
            )
        else:
            assert key is not None
            assert value is not None
            query = PackedTensorSequences.pack_input(query)
            key = PackedTensorSequences.pack_input(
                key, key_padding_mask=key_padding_mask
            )
            value = PackedTensorSequences.pack_input(
                value, key_padding_mask=key_padding_mask
            )

        output_packed, attn_weights = self.forward_packed(
            query, key, value, key_padding_mask, attn_mask, return_weights
        )
        output = output_packed.to_padded()
        return output, attn_weights
