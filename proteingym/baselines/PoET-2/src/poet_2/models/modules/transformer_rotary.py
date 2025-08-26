import copy
import warnings
from typing import TYPE_CHECKING, Optional, Union

import torch

from poet_2.models.modules import FlashMultiheadAttention, RotaryEmbedding
from poet_2.models.modules.attention import Atom3BiasParams, T
from poet_2.models.modules.attention_flash import (
    FLASH_ATTENTION_MODULE_INSTALLED,
    FlashAttentionImpl,
    is_bh_attn_mask,
)
from poet_2.models.modules.embedding import FLASH_ROTARY_AVAILABLE
from poet_2.models.modules.packed_sequence import PackedTensorSequences
from poet_2.models.modules.transformer import (
    TieredTransformerEncoderLayer,
    TransformerDecoderLayer,
)

if FLASH_ATTENTION_MODULE_INSTALLED:
    try:
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    except ImportError:
        flash_attn_with_kvcache = None
else:
    flash_attn_with_kvcache = None

if TYPE_CHECKING:
    # only import if type checking to try to keep flash_attn an optional dependency...
    from flash_attn.utils.generation import InferenceParams


def is_rflash_inference_avail(attn_mask: torch.Tensor | Atom3BiasParams | None) -> bool:
    return attn_mask is None or (
        isinstance(attn_mask, torch.Tensor) and is_bh_attn_mask(attn_mask)
    )


class RotaryFlashAttentionImpl(FlashAttentionImpl):
    def forward_flash_self_attn(
        self,
        q: PackedTensorSequences,
        k: PackedTensorSequences,
        v: PackedTensorSequences,
        attn_mask: torch.Tensor | Atom3BiasParams | None = None,
        causal: bool = False,
        need_weights: bool = False,
        inference_params: Optional["InferenceParams"] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> tuple[PackedTensorSequences, None]:
        if inference_params is None:
            return super().forward_flash_self_attn(
                q=q,
                k=k,
                v=v,
                attn_mask=attn_mask,
                causal=causal,
                need_weights=need_weights,
            )
        assert not need_weights
        assert not isinstance(attn_mask, Atom3BiasParams)
        assert flash_attn_with_kvcache is not None

        assert q.dtype == torch.float16 or q.dtype == torch.bfloat16
        assert q.is_cuda
        assert k.dtype == torch.float16 or k.dtype == torch.bfloat16
        assert k.is_cuda
        assert v.dtype == torch.float16 or v.dtype == torch.bfloat16
        assert v.is_cuda

        assert len(inference_params.key_value_memory_dict) == 1
        cache: torch.Tensor = next(
            iter(inference_params.key_value_memory_dict.values())
        )
        cache_seqlens = (
            inference_params.lengths_per_sample
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        output = flash_attn_with_kvcache(
            q=q.x.view(q.cu_seqlens.numel() - 1, -1, q.x.size(1), q.x.size(2)),
            k_cache=cache[:, :, 0],
            v_cache=cache[:, :, 1],
            k=k.x.view(k.cu_seqlens.numel() - 1, -1, k.x.size(1), k.x.size(2)),
            v=v.x.view(v.cu_seqlens.numel() - 1, -1, v.x.size(1), v.x.size(2)),
            rotary_cos=cos,
            rotary_sin=sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.softmax_scale,
            causal=causal,
            rotary_interleaved=True,
            alibi_slopes=attn_mask.squeeze() if attn_mask is not None else None,
        ).flatten(start_dim=0, end_dim=1)
        output_packed = copy.copy(q)
        output_packed.x = output
        return output_packed, None

    def forward_flash_attn(
        self,
        q: PackedTensorSequences,
        k: PackedTensorSequences,
        v: PackedTensorSequences,
        attn_mask: torch.Tensor | Atom3BiasParams | None = None,
        causal: bool = False,
        need_weights: bool = False,
        inference_params: Optional["InferenceParams"] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> tuple[PackedTensorSequences, None]:
        if (
            self.dropout_p != 0.0
            or inference_params is None
            or k.x.dim() == 3  # means that k is packed, so can't use flash + kvcache
        ):
            return super().forward_flash_attn(
                q=q,
                k=k,
                v=v,
                attn_mask=attn_mask,
                causal=causal,
                need_weights=need_weights,
            )
        assert not need_weights
        assert not isinstance(attn_mask, Atom3BiasParams)
        assert q.dtype == torch.float16 or q.dtype == torch.bfloat16
        assert q.is_cuda
        assert flash_attn_with_kvcache is not None

        # TODO: is this actually faster?
        # TODO: any benefit to fusing rotary?
        output = flash_attn_with_kvcache(
            q=q.x.view(q.cu_seqlens.numel() - 1, -1, q.x.size(1), q.x.size(2)),
            k_cache=k.x,
            v_cache=v.x,
            k=None,
            v=None,
            cache_seqlens=k.cu_seqlens.diff(),
            softmax_scale=self.softmax_scale,
            causal=causal,
            alibi_slopes=attn_mask.squeeze() if attn_mask is not None else None,
        ).flatten(start_dim=0, end_dim=1)
        output_packed = copy.copy(q)
        output_packed.x = output
        return output_packed, None

    def forward(
        self,
        q: PackedTensorSequences,
        k: PackedTensorSequences,
        v: PackedTensorSequences,
        attn_mask: torch.Tensor | Atom3BiasParams | None = None,
        causal: bool = False,
        need_weights: bool = False,
        self_attention: bool = False,
        inference_params: Optional["InferenceParams"] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ):
        if inference_params is None:
            return super().forward(
                q=q,
                k=k,
                v=v,
                attn_mask=attn_mask,
                causal=causal,
                need_weights=need_weights,
                self_attention=self_attention,
            )
        dtype = q.dtype
        assert self.is_flash_attn_avail(
            q, attn_mask=attn_mask, causal=causal, need_weights=need_weights
        )
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
            q,
            k,
            v,
            attn_mask=attn_mask,
            causal=causal,
            need_weights=need_weights,
            inference_params=inference_params,
            cos=cos,
            sin=sin,
        )
        output.x = output.x.to(dtype)
        return output, attn_weights


class RotaryFlashMultiheadAttention(FlashMultiheadAttention):
    _inner_attn_impl_cls = RotaryFlashAttentionImpl

    def __init__(
        self, *args, rotary_scale=None, rotary_force_fp32=None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rotary_emb = RotaryEmbedding(
            dim_model=self.head_dim,
            scale=rotary_scale,
            force_fp32=rotary_force_fp32,
        )
        self._inference_params: Optional["InferenceParams"] = None
        self._cos_sin_key: Optional[tuple[int, torch.dtype, torch.device]] = None
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None

    def _transform_qkv(
        self,
        query,
        key,
        value,
        query_positions=None,
        key_positions=None,
        transform_query=True,
        transform_key=True,
        transform_value=False,
        attn_mask: torch.Tensor | Atom3BiasParams | None = None,
    ):
        if self._inference_params is not None:
            if self.self_attention:  # rotary is fused
                return query, key, value
            if query_positions is not None:
                query_positions = query_positions + self._inference_params.seqlen_offset
            if key_positions is not None:
                key_positions = key_positions + self._inference_params.seqlen_offset
        query, key = self.rotary_emb(
            query,
            key,
            q_positions=query_positions,
            k_positions=key_positions,
            transform_q=transform_query,
            transform_k=transform_key,
        )
        return query, key, value

    def _inner_attn(
        self,
        q: PackedTensorSequences,
        k: PackedTensorSequences,
        v: PackedTensorSequences,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: torch.Tensor | Atom3BiasParams | None = None,
        return_weights: bool = False,
    ):
        context_packed, attn_weights = self.inner_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            need_weights=return_weights,
            causal=self.causal,
            self_attention=self.self_attention,
            inference_params=self._inference_params,
            cos=self._cos,
            sin=self._sin,
        )
        return context_packed, attn_weights

    def _cache_rotary_cos_sin(
        self, max_length: int, dtype: torch.dtype, device: torch.device
    ):
        key = (max_length, dtype, device)
        if key == self._cos_sin_key:
            return
        self._cos_sin_key = key
        cos_sin_t = torch.arange(max_length, device=device)
        self._cos, self._sin = self.rotary_emb.get_cos_sin_tables(
            t=cos_sin_t, dtype=dtype
        )

    def forward(
        self,
        query: T,
        key: Optional[T] = None,
        value: Optional[T] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: torch.Tensor | Atom3BiasParams | None = None,
        return_weights: bool = False,
        return_projs: bool = False,
        transform_query: bool = True,
        transform_key: bool = True,
        transform_value: bool = True,
        inference_params: Optional["InferenceParams"] = None,
    ) -> Union[
        tuple[T, Optional[torch.Tensor]],
        tuple[T, Optional[torch.Tensor], tuple[T, T, T]],
    ]:
        original_inference_params = self._inference_params
        if inference_params is not None:
            assert isinstance(query, PackedTensorSequences)
            assert query.x.device.type == "cuda"
            if not is_rflash_inference_avail(attn_mask):
                raise NotImplementedError(
                    "cuda inference with provided attn_mask is not supported"
                )
            if not FLASH_ROTARY_AVAILABLE:
                raise NotImplementedError(
                    "only flash rotary path supported for cuda inference"
                )
            self._inference_params = inference_params
            self._cache_rotary_cos_sin(
                max_length=inference_params.max_seqlen,
                dtype=query.x.dtype,
                device=query.x.device,
            )
        result = super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            return_weights=return_weights,
            return_projs=return_projs,
            transform_query=transform_query,
            transform_key=transform_key,
            transform_value=transform_value,
        )
        self._inference_params = original_inference_params
        return result


class TieredRotaryTransformerEncoderLayer(TieredTransformerEncoderLayer):
    def __init__(
        self,
        *args,
        rotary_scale=None,
        rotary_force_fp32=None,
        use_multi_rotary=True,
        **kwargs,
    ):
        self.rotary_scale = rotary_scale
        self.rotary_force_fp32 = rotary_force_fp32
        self.use_multi_rotary = use_multi_rotary
        super().__init__(*args, **kwargs)

    def _init_self_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        """
        Initialize the multi-head attention module used for each sequence independently.
        """
        return RotaryFlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )

    def _init_multi_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        """
        Initialize the multi-head attention module used for each sequence-of-sequences.
        """
        Module = FlashMultiheadAttention
        if self.use_multi_rotary:
            Module = RotaryFlashMultiheadAttention
        return Module(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )


class RotaryTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        *args,
        rotary_scale=None,
        rotary_force_fp32=None,
        use_cross_rotary=True,
        **kwargs,
    ):
        self.rotary_scale = rotary_scale
        self.rotary_force_fp32 = rotary_force_fp32
        self.use_cross_rotary = use_cross_rotary
        super().__init__(*args, **kwargs)

    def _init_self_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=True,
    ):
        return RotaryFlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )

    def _init_cross_mha_module(
        self, d_model, nhead, dropout=0, use_qkv_bias=False, batch_first=True
    ):
        Module = FlashMultiheadAttention
        if self.use_cross_rotary:
            Module = RotaryFlashMultiheadAttention
        return Module(
            d_model,
            nhead,
            self_attention=False,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=False,
            rotary_scale=self.rotary_scale,
            rotary_force_fp32=self.rotary_force_fp32,
        )
