import contextlib
import copy
import math
from typing import TYPE_CHECKING, Literal, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

try:
    import xformers

    XFORMERS_MODULE_INSTALLED = True
except ModuleNotFoundError:
    XFORMERS_MODULE_INSTALLED = False


from poet_2.models.modules.attention import Atom3BiasParams
from poet_2.models.modules.attention_flash import FlashMultiheadAttention
from poet_2.models.modules.glu import GLU, ActivatedLinear
from poet_2.models.modules.norm import RMSNorm
from poet_2.models.modules.packed_sequence import PackedTensorSequences

if TYPE_CHECKING:
    # only import if type checking to try to keep flash_attn an optional dependency...
    from flash_attn.utils.generation import InferenceParams


T = TypeVar("T", Tensor, PackedTensorSequences)


def get_glu_d_ff(dim_feedforward: int) -> int:
    multiple_of = 256
    hidden_dim = int(2 * dim_feedforward / 3)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


@torch.compile
def _swiglu_apply_feedforward(
    linear1: Union[ActivatedLinear, GLU],
    dropout: nn.Dropout,
    linear2: nn.Linear,
    x: torch.Tensor,
) -> torch.Tensor:
    return linear2(dropout(linear1(x)))


def _maybe_fused_swiglu_apply_feedforward(
    linear1: Union[ActivatedLinear, GLU],
    dropout: nn.Dropout,
    linear2: nn.Linear,
    x: torch.Tensor,
    use_fused_swiglu: bool,
) -> torch.Tensor:
    from xformers.ops import swiglu, unbind

    # if use_fused_swiglu, must have isinstance(linear1, GLU)
    # linear1.proj may be replaced by a lora; in that case we can't fuse here
    if not use_fused_swiglu or not isinstance(linear1.proj, nn.Linear):
        return _swiglu_apply_feedforward(linear1, dropout, linear2, x)
    else:
        assert isinstance(linear1, GLU)
        assert linear2.bias is None
        w1w2 = linear1.proj.weight
        w1, w2 = unbind(w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]]), dim=0)
        return swiglu(
            x=x.contiguous(),
            w1=w1,
            b1=None,
            w2=w2,
            b2=None,
            w3=linear2.weight,
            b3=None,
            op=None,
        )


class TransformerEncoder(nn.TransformerEncoder):
    def __init__(
        self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False
    ):
        super().__init__(encoder_layer, num_layers, norm, enable_nested_tensor)
        for layer in self.layers:
            layer.reset_parameters()

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward(
        self,
        x,
        src_mask=None,
        src_key_padding_mask=None,
        return_attention=False,
        activation_checkpointing=False,
        **kwargs,
    ):
        attn = []
        for layer in self.layers:
            if not activation_checkpointing:
                x = layer(
                    x,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attention=return_attention,
                    **kwargs,
                )
            else:
                x = checkpoint.checkpoint(
                    layer,
                    x,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    return_attention=return_attention,
                    **kwargs,
                    use_reentrant=False,
                )
            if return_attention:
                x, a = x
                attn.append(a)

        if return_attention:
            return x, attn

        return x


_USE_GET_MAX_S_CACHE: bool = False
_GET_MAX_S_CACHE: dict[torch.Tensor, int] = {}


@contextlib.contextmanager
def get_max_s_cache():
    # can be used to reduce cuda host device syncs
    global _USE_GET_MAX_S_CACHE
    try:
        _USE_GET_MAX_S_CACHE = True
        yield
    finally:
        _USE_GET_MAX_S_CACHE = False
        _GET_MAX_S_CACHE.clear()


def get_max_s(seqs_cu_seqlens: torch.Tensor) -> int:
    key = seqs_cu_seqlens
    if _USE_GET_MAX_S_CACHE and key in _GET_MAX_S_CACHE:
        return _GET_MAX_S_CACHE[key]
    value = seqs_cu_seqlens.diff().max().item()
    if _USE_GET_MAX_S_CACHE:
        _GET_MAX_S_CACHE[key] = value
    return value


class TransformerDecoderLayer(nn.Module):
    """Implements a pre-layer norm transformer decoder layer."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=None,
        activation=nn.GELU(),
        dropout=0,
        use_qkv_bias=False,
        batch_first=False,
        causal=True,
        norm_type: Literal["layer", "rms"] = "layer",
        use_glu: bool = False,
        n_distance_buckets: int | None = None,
        n_angle_buckets: int | None = None,
        use_confidence_bucket: bool = False,
    ):
        super().__init__()

        assert (
            batch_first
        ), "Flash Attention requires batch first, make sure your transformer uses batch_first=True."
        assert norm_type in {"layer", "rms"}
        if use_confidence_bucket:
            assert n_distance_buckets or n_angle_buckets
        n_distance_buckets = n_distance_buckets or 0
        n_angle_buckets = n_angle_buckets or 0

        self.dim = d_model
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.dim_feedforward = dim_feedforward
        self.num_heads = nhead
        self.activation = activation

        self.self_attn = self._init_self_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )
        if n_distance_buckets or n_angle_buckets:
            if n_distance_buckets > 0 and n_angle_buckets > 0:
                n_buckets = n_distance_buckets * n_angle_buckets
            else:
                n_buckets = n_distance_buckets or n_angle_buckets
            n_buckets += 1  # for mask bucket
            if use_confidence_bucket:
                n_buckets += 1
            self.bias_weights = nn.Parameter(
                torch.randn((nhead, n_buckets)) / math.sqrt(16 * d_model // nhead)
            )
        self.multihead_attn = self._init_cross_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
        )

        d_ff = dim_feedforward if not use_glu else get_glu_d_ff(dim_feedforward)
        self.linear1 = (
            ActivatedLinear(d_model, d_ff, activation=activation)
            if not use_glu
            else GLU(d_model, d_ff, activation=activation)
        )
        self.linear2 = nn.Linear(d_ff, d_model, bias=not use_glu)

        nn.init.constant_(self.linear2.weight, 0.0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0.0)

        self.use_fused_swiglu = (
            XFORMERS_MODULE_INSTALLED
            and use_glu
            and activation.__name__ == "silu"
            and dropout == 0
        )

        norm_cls = nn.LayerNorm if norm_type == "layer" else RMSNorm
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.norm3 = norm_cls(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _init_self_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=True,
    ):
        return FlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )

    def _init_cross_mha_module(
        self, d_model, nhead, dropout=0, use_qkv_bias=False, batch_first=True
    ):
        return FlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=False,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
        )

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        if hasattr(self, "bias_weights"):
            nn.init.normal_(
                self.bias_weights, std=1 / math.sqrt(16 * self.dim // self.num_heads)
            )
        self.multihead_attn.reset_parameters()
        self.linear1.reset_parameters()
        nn.init.constant_(self.linear2.weight, 0.0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0.0)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

    def _apply_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        return _maybe_fused_swiglu_apply_feedforward(
            linear1=self.linear1,
            dropout=self.dropout,
            linear2=self.linear2,
            x=x,
            use_fused_swiglu=self.use_fused_swiglu,
        )

    def forward_packed(
        self,
        tgt: PackedTensorSequences,
        memory: PackedTensorSequences,
        seqs_cu_seqlens: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_self_attention: bool = False,
        return_cross_attention: bool = False,
        inference_params: Optional["InferenceParams"] = None,
    ) -> Tuple[PackedTensorSequences, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.forward_packed_from_key_value(
            tgt=tgt,
            key=memory,
            value=memory,
            seqs_cu_seqlens=seqs_cu_seqlens,
            transform_key=True,
            transform_value=True,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_self_attention=return_self_attention,
            return_cross_attention=return_cross_attention,
            inference_params=inference_params,
        )

    def forward_packed_from_key_value(
        self,
        tgt: PackedTensorSequences,
        key: PackedTensorSequences,
        value: PackedTensorSequences,
        seqs_cu_seqlens: Optional[Tensor] = None,
        transform_key: bool = False,
        transform_value: bool = False,
        tgt_mask: Tensor | Atom3BiasParams | None = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_self_attention: bool = False,
        return_cross_attention: bool = False,
        inference_params: Optional["InferenceParams"] = None,
    ) -> Union[
        Tensor, tuple[PackedTensorSequences, tuple[Optional[Tensor], Optional[Tensor]]]
    ]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.
        """
        tgt_norm = copy.copy(tgt)
        tgt_norm.x = self.norm1(tgt.x)
        tgt2, attn_self = self.self_attn(
            tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            return_weights=return_self_attention,
            inference_params=inference_params,
        )
        del tgt_norm
        tgt = copy.copy(tgt)
        tgt.x = tgt.x + self.dropout1(tgt2.x)
        del tgt2

        tgt_norm = copy.copy(tgt)
        tgt_norm.x = self.norm2(tgt.x)
        if seqs_cu_seqlens is not None:
            tgt_norm.cu_seqlens = seqs_cu_seqlens  # "reshape" the packed sequences
            tgt_norm.max_s = get_max_s(tgt_norm.cu_seqlens)
            if tgt_norm.to_paddedable:
                tgt_norm.to_paddedable = False
                tgt_norm.make_to_paddedable()
        tgt2, attn = self.multihead_attn(
            tgt_norm,
            key,
            value,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            return_weights=return_cross_attention,
            transform_query=True,
            transform_key=transform_key,
            transform_value=transform_value,
            inference_params=inference_params,
        )
        del tgt_norm
        tgt = copy.copy(tgt)
        tgt.x = tgt.x + self.dropout2(tgt2.x)
        del tgt2

        tgt.x = tgt.x + self.dropout3(self._apply_feedforward(self.norm3(tgt.x)))
        if return_self_attention or return_cross_attention:
            return tgt, (attn_self, attn)
        else:
            return tgt

    def forward_padded(
        self,
        tgt: Tensor,
        memory: Tensor,
        seqs_cu_seqlens: Optional[Tensor] = None,
        tgt_mask: Tensor | Atom3BiasParams | None = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_self_attention: bool = False,
        return_cross_attention: bool = False,
    ) -> Union[Tensor, tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor]]]]:
        assert not isinstance(tgt_mask, Atom3BiasParams)
        if seqs_cu_seqlens is not None:
            raise NotImplementedError
        tgt_norm = self.norm1(tgt)
        tgt2, attn_self = self.self_attn(
            tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            return_weights=return_self_attention,
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt_norm = self.norm2(tgt)
        tgt2, attn = self.multihead_attn(
            tgt_norm,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            return_weights=return_cross_attention,
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt = tgt + self.dropout3(self._apply_feedforward(self.norm3(tgt)))
        if return_self_attention or return_cross_attention:
            return tgt, (attn_self, attn)
        return tgt

    def forward(
        self,
        tgt: Union[Tensor, PackedTensorSequences],
        memory: Union[Tensor, PackedTensorSequences],
        seqs_cu_seqlens: Optional[Tensor] = None,
        tgt_mask: Tensor | Atom3BiasParams | None = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_self_attention: bool = False,
        return_cross_attention: bool = False,
    ) -> Tuple[
        Union[Tensor, PackedTensorSequences],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        fn = self.forward_padded
        if type(tgt) is PackedTensorSequences:
            assert type(memory) is PackedTensorSequences
            fn = self.forward_packed
        return fn(
            tgt,
            memory,
            seqs_cu_seqlens=seqs_cu_seqlens,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_self_attention=return_self_attention,
            return_cross_attention=return_cross_attention,
        )


class TransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        assert norm is None
        super().__init__(decoder_layer, num_layers, norm)
        for layer in self.layers:
            layer.reset_parameters()

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward(
        self,
        x: T,
        activation_checkpointing: bool = False,
        **kwargs,
    ):
        # TODO: doesn't really handle attention or other return args
        for layer in self.layers:
            if not activation_checkpointing:
                x = layer(x, **kwargs)
            else:
                x = checkpoint.checkpoint(
                    layer,
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
        return x


class TieredTransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer that operates on sequences-of-sequences. Processes sequences
    in two attention blocks analogously to transformer decoder layers. The first attention
    layer only attends within each sequence. The second attention layer also attends to
    other sequences within each sequence-of-sequences.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=None,
        activation=nn.GELU(),
        dropout=0,
        use_qkv_bias=False,
        batch_first=False,
        causal=True,
        self_causal: Optional[bool] = None,  # if None, self_causal = causal
        norm_type: Literal["layer", "rms"] = "layer",
        use_glu: bool = False,
        n_distance_buckets: int | None = None,
        n_angle_buckets: int | None = None,
        use_confidence_bucket: bool = False,
    ):
        super().__init__()
        assert (
            batch_first
        ), "Flash Attention requires batch first, make sure your transformer uses batch_first=True."
        assert norm_type in {"layer", "rms"}
        if use_confidence_bucket:
            assert n_distance_buckets or n_angle_buckets
        n_distance_buckets = n_distance_buckets or 0
        n_angle_buckets = n_angle_buckets or 0

        self.dim = d_model
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.dim_feedforward = dim_feedforward
        self.num_heads = nhead
        self.activation = activation

        self.self_attn = self._init_self_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=self_causal if self_causal is not None else causal,
        )
        if n_distance_buckets or n_angle_buckets:
            if n_distance_buckets > 0 and n_angle_buckets > 0:
                n_buckets = n_distance_buckets * n_angle_buckets
            else:
                n_buckets = n_distance_buckets or n_angle_buckets
            n_buckets += 1  # for mask bucket
            if use_confidence_bucket:
                n_buckets += 1
            self.bias_weights = nn.Parameter(
                torch.randn((nhead, n_buckets)) / math.sqrt(16 * d_model // nhead)
            )
        self.multihead_attn = self._init_multi_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )

        d_ff = dim_feedforward if not use_glu else get_glu_d_ff(dim_feedforward)
        self.linear1 = (
            ActivatedLinear(d_model, d_ff, activation=activation)
            if not use_glu
            else GLU(d_model, d_ff, activation=activation)
        )
        self.linear2 = nn.Linear(d_ff, d_model, bias=not use_glu)

        nn.init.constant_(self.linear2.weight, 0.0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0.0)

        self.use_fused_swiglu = (
            XFORMERS_MODULE_INSTALLED
            and use_glu
            and activation.__name__ == "silu"
            and dropout == 0
        )

        norm_cls = nn.LayerNorm if norm_type == "layer" else RMSNorm
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.norm3 = norm_cls(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

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
        return FlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
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
        return FlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=True,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        if hasattr(self, "bias_weights"):
            nn.init.normal_(
                self.bias_weights, std=1 / math.sqrt(16 * self.dim // self.num_heads)
            )
        self.multihead_attn.reset_parameters()
        self.linear1.reset_parameters()
        nn.init.constant_(self.linear2.weight, 0.0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0.0)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

    def _apply_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        return _maybe_fused_swiglu_apply_feedforward(
            linear1=self.linear1,
            dropout=self.dropout,
            linear2=self.linear2,
            x=x,
            use_fused_swiglu=self.use_fused_swiglu,
        )

    def forward_packed(
        self,
        x: PackedTensorSequences,
        seqs_cu_seqlens: Tensor,
        src_mask: Tensor | Atom3BiasParams | None = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
        return_memory: bool = False,
        patch_size: Optional[int] = None,
        segment_sizes: Optional[torch.Tensor] = None,
        return_patch: bool = False,
    ) -> Union[
        PackedTensorSequences,
        tuple[PackedTensorSequences, tuple[Tensor, Tensor]],
        tuple[
            PackedTensorSequences,
            tuple[Optional[Tensor], Optional[Tensor]],
            tuple[PackedTensorSequences, PackedTensorSequences],
        ],
        tuple[
            PackedTensorSequences,
            tuple[Optional[Tensor], Optional[Tensor]],
            tuple[Optional[PackedTensorSequences], Optional[PackedTensorSequences]],
            PackedTensorSequences,
        ],
    ]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.

        Input is a sequence-of-sequences packed consecutively. This allows sequences to be
        interpreted as individual data points or sequences-of-sequences to be interpreted
        as individual data points by changing the sequence lengths encoded in the packed sequence.

        x: PackedTensorSequences of the individual sequences.
        seqs_cu_seqlens: (B+1,) the cumulative lengths of the sequences-of-sequences.
        src_key_padding_mask: B x N x L x K where B is the batch size, N is the number of sequences-per-sequence,
            L is the length of each sequences, and K is the hidden dim
        """
        if return_multi_attention:
            return_attention = True
        if return_self_attention:
            return_attention = True

        # apply the self attention layer on the sequences independently
        x_norm = copy.copy(x)
        x_norm.x = self.norm1(x.x)
        x2, attn_self = self.self_attn(
            x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_self_attention,
        )
        del x_norm
        x = copy.copy(x)
        x.x = x.x + self.dropout1(x2.x)
        del x2

        # apply the sequence-of-sequence attention layer on the reshaped sequences
        x_norm = copy.copy(x)
        if patch_size is not None:
            assert patch_size == 0
            assert segment_sizes is not None
            x_norm.x = x.x[x.cu_seqlens[:-1].long()]
            x_norm.x = self.norm2(x_norm.x)
            n_seqs = (segment_sizes > 0).long().sum(dim=1)
            x_norm.cu_seqlens = (
                F.pad(n_seqs.cumsum(dim=0), (1, 0))
                .type(torch.int32)
                .to(x_norm.cu_seqlens.device)
            )
            x_norm.max_s = n_seqs.max().item()
            x_norm.positions = None
            nonzero_segment_sizes = (
                segment_sizes[segment_sizes > 0].flatten().to(x_norm.x.device)
            )
            assert not x_norm.to_paddedable
            assert src_key_padding_mask is None
        else:
            x_norm.x = self.norm2(x.x)
            x_norm.cu_seqlens = seqs_cu_seqlens  # "reshape" the packed sequences
            x_norm.max_s = get_max_s(x_norm.cu_seqlens)
            if x_norm.to_paddedable:
                x_norm.to_paddedable = False
                x_norm.make_to_paddedable()
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.view(
                    -1, src_key_padding_mask.size(-1)
                )
        if not return_memory:
            x2, attn = self.multihead_attn(
                x_norm,
                attn_mask=None,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
            )
            key, value = None, None
        else:
            x2, attn, (_, key, value) = self.multihead_attn(
                x_norm,
                attn_mask=None,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
                return_projs=return_memory,
            )
        del x_norm
        patch = copy.copy(x2) if return_patch else None
        if patch_size is not None:
            indices = torch.arange(
                nonzero_segment_sizes.numel(), device=x2.x.device
            ).repeat_interleave(nonzero_segment_sizes)
            x2.x = F.embedding(indices, x2.x)
        x = copy.copy(x)
        x.x = x.x + self.dropout2(x2.x)
        del x2

        x.x = x.x + self.dropout3(self._apply_feedforward(self.norm3(x.x)))

        if return_attention:
            return x, (attn_self, attn)
        if return_memory:
            return x, (attn_self, attn), (key, value)
        if return_patch:
            return x, (attn_self, attn), (key, value), patch
        return x

    def forward_padded(
        self,
        x: Tensor,
        seqs_cu_seqlens: Optional[Tensor] = None,
        src_mask: Tensor | Atom3BiasParams | None = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
        return_memory: bool = False,
    ) -> Union[
        Tensor,
        tuple[Tensor, tuple[Tensor, Tensor]],
        tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor]], Tensor],
    ]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.

        Input is a sequence-of-sequences packed consecutively. This allows sequences to be
        interpreted as individual data points or sequences-of-sequences to be interpreted
        as individual data points by changing the sequence lengths encoded in the packed sequence.

        x: Tensor of the individual sequences. Size B x N x L x K
        src_key_padding_mask: B x N x L where B is the batch size, N is the number of sequences-per-sequence,
            L is the length of each sequences
        """
        assert not isinstance(src_mask, Atom3BiasParams)
        if return_multi_attention:
            return_attention = True
        if return_self_attention:
            return_attention = True

        B, N, L, K = x.size()
        # sequence-independent attention
        x = x.view(B * N, L, K)
        x_norm = self.norm1(x)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(B * N, L)
        x2, attn_self = self.self_attn(
            x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_self_attention,
        )
        x = x + self.dropout1(x2)

        # sequence-of-sequences attention
        x = x.view(B, N * L, K)
        x_norm = self.norm2(x)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(B, N * L)
        if not return_memory:
            x2, attn = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
            )
        else:
            x2, attn, (_, key, value) = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
                return_projs=return_memory,
            )
        x = x + self.dropout2(x2)

        # reshape x back
        x = x.view(B, N, L, K)

        x = x + self.dropout3(self._apply_feedforward(self.norm3(x)))

        if return_attention:
            return x, (attn_self, attn)
        if return_memory:
            return x, (attn_self, attn), (key, value)
        return x

    def forward(
        self,
        x: T,
        seqs_cu_seqlens: Optional[Tensor] = None,
        src_mask: Tensor | Atom3BiasParams | None = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_memory: bool = False,
    ) -> Union[
        T,
        tuple[T, tuple[Tensor, Tensor]],
        tuple[T, tuple[Optional[Tensor], Optional[Tensor]], T],
    ]:
        """
        See self.forward_padded and self.forward_packed for information about x,
        seqs_cu_seqlens, src_mask, and src_key_padding_mask.

        By default, only returns the output of the layer: (out)

        If return_attention=True, additionally returns the self and multi-sequence
        attention matrices: (out, (attn_self, attn))

        If return_memory=True, additionally returns the "memory" (input to multi-
        sequence attention): (out, (attn_self, attn), memory)
        Here, attn_self and attn may be None depending on the value of
        return_attention.
        """
        fn = self.forward_padded
        if type(x) is PackedTensorSequences:
            assert seqs_cu_seqlens is not None
            fn = self.forward_packed
        return fn(
            x,
            seqs_cu_seqlens=seqs_cu_seqlens,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            return_self_attention=return_attention,
            return_multi_attention=return_attention,
            return_memory=return_memory,
        )
