import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from poet_2.models.modules.packed_sequence import PackedTensorSequences


def mha_attn_weights(
    q: Tensor,
    k: Tensor,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    scaling=None,
    batch_first=False,
    causal=False,
) -> Tensor:
    """Input shape: Length x Batch x Channel
    Args:
        query (Tensor): shape `(tgt_len, batch, num_heads, head_dim)`
        key (Tensor): shape `(src_len, batch, num_heads, head_dim)`
        key_padding_mask (ByteTensor, optional): mask to exclude
            keys that are pads, of shape `(batch, src_len)`, where
            padding elements are indicated by 1s.
        attn_mask (FloatTensor, optional): typically used to
            implement causal attention, where the mask prevents the
            attention from looking forward in time (default: None). Can be
            shape `(tgt_len, src_len)`, `(batch, tgt_len, src_len)`,
            or `(batch, num_heads, tgt_len, src_len)`.
            This is added to the attention map, so should be -inf at masked positions.
    Returns:
        attn_weights (Tensor): shape `(batch, num_heads, tgt_len, src_len)`
    """
    # we want q and k to be (batch, num_heads, length, head_dim)
    # input is either (batch, length, num_heads, head_dim) or (length, batch, num_heads, head_dim)
    if batch_first:
        q = q.transpose(1, 2)  # (batch, num_heads, length, head_dim)
        k = k.transpose(1, 2)
    else:
        q = q.permute(1, 2, 0, 3)  # (batch, num_heads, length, head_dim)
        k = k.permute(1, 2, 0, 3)

    bsz, num_heads, tgt_len, head_dim = q.size()
    src_len = k.size(2)

    if scaling is None:
        scaling = head_dim**-0.5
    q *= scaling

    q = q.contiguous().view(bsz * num_heads, tgt_len, head_dim)
    k = k.contiguous().view(bsz * num_heads, src_len, head_dim)

    attn_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len)

    if attn_mask is not None:
        # attn_mask is broadcast over the batch and number of heads
        attn_weights += attn_mask

    if key_padding_mask is not None:
        # don't attend to padding symbols
        # key_padding mask is (batch, src_len)
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

    if causal:
        # apply a triangular mask to the attention to ensure
        # that each token only attends to itself and previous positions
        causal_mask = torch.triu(
            torch.zeros_like(attn_weights[0, 0]) - torch.inf, diagonal=1
        )
        attn_weights += causal_mask

    attn_weights_float = torch.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights_float.type_as(attn_weights)

    return attn_weights


def mha_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    return_weights: bool = False,
    scaling=None,
    batch_first=False,
    dropout=0,
    causal=False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Input shape: Batch x Length x Num_Heads x Head_Dim or Length x Batch x Num_Heads x Head_Dim
    Args:
        key_padding_mask (ByteTensor, optional): mask to exclude
            keys that are pads, of shape `(batch, src_len)`, where
            padding elements are indicated by 1s.
        attn_mask (FloatTensor, optional): typically used to
            implement causal attention, where the mask prevents the
            attention from looking forward in time (default: None). Can be
            shape `(tgt_len, src_len)`, `(batch, tgt_len, src_len)`,
            or `(batch, num_heads, tgt_len, src_len)`.
            This is added to the attention map, so should be -inf at masked positions.
        return_weights (bool, optional): return the log attention weights,
            (default: False).
        scaling (float, optional): scale the query matrix by this (default: sqrt(head_dim))
        batch_first (bool, optional): set batch as first dimension (default: False)
        dropout (float, optional): apply dropout with this rate
        causal (bool, optional): whether to apply a causal mask in additiona to
            the attn_mask. For compatability with the flash attention module.
            (default: False)
    """
    # first, calculate the attention weights
    attn_weights = mha_attn_weights(
        q,
        k,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        scaling=scaling,
        batch_first=batch_first,
        causal=causal,
    )
    # attn_weights is (batch x num_heads x tgt_len x src_len)
    # needs v to be (batch x num_heads x src_len x head_dim)
    if batch_first:  # v is (batch x src_len x num_heads x head_dim)
        v = v.transpose(1, 2)
    else:  # v is (src_len x batch x num_heads x head_dim)
        v = v.permute(1, 2, 0, 3)
    assert v.size(2) == attn_weights.size(3)

    bsz, num_heads, src_len, head_dim = v.size()
    v = v.contiguous().view(bsz * num_heads, src_len, head_dim)
    tgt_len = attn_weights.size(2)
    attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_probs = attn_weights
    if dropout > 0:
        attn_probs = F.dropout(attn_probs, p=dropout)

    attn = torch.bmm(attn_probs, v)  # (bsz*num_heads, tgt_len, head_dim)
    assert list(attn.size()) == [bsz * num_heads, tgt_len, head_dim]
    if batch_first:
        # return should be (batch, length, num_heads*head_dim)
        attn = attn.view(bsz, num_heads, tgt_len, head_dim)
        attn = (
            attn.transpose(1, 2).contiguous().view(bsz, tgt_len, num_heads * head_dim)
        )
    else:
        # return should be (length, batch, num_heads*head_dim)
        attn = (
            attn.transpose(0, 1).contiguous().view(tgt_len, bsz, num_heads * head_dim)
        )

    if return_weights:
        attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn, attn_weights

    return attn, None


T = TypeVar("T", torch.Tensor, PackedTensorSequences)


@dataclass
class Atom3BiasParams:
    coords: Tensor | None
    planes: Tensor | None
    confidences: Tensor | None
    weights: Tensor | None
    n_distance_buckets: int = 128
    n_angle_buckets: int = 128
    use_confidence_bucket: bool = True
    max_distance: float = 48.0
    confidence_threshold: float = 0.7
    pad_input: Callable[[Tensor], Tensor] | None = None

    @cached_property
    def attn_mask(self) -> torch.Tensor:
        assert self.pad_input is not None
        qd = self.pad_input(self.coords) if self.coords is not None else None
        qa = self.pad_input(self.planes) if self.planes is not None else None
        qc = (
            self.pad_input(self.confidences.unsqueeze(1)).squeeze(2)
            if self.confidences is not None
            else None
        )
        kd, ka, kc = qd, qa, qc
        assert qd is not None or qa is not None
        assert self.weights is not None

        if qd is not None:
            device = qd.device
            qdx, qdy, qdz = qd[:, :, 0], qd[:, :, 1], qd[:, :, 2]
            kdx, kdy, kdz = kd[:, :, 0], kd[:, :, 1], kd[:, :, 2]
        if qa is not None:
            device = qa.device
            qax, qay, qaz = qa[:, :, 0], qa[:, :, 1], qa[:, :, 2]
            kax, kay, kaz = ka[:, :, 0], ka[:, :, 1], ka[:, :, 2]

        if qd is not None:
            MIN_DISTANCE = 2.5
            distances = (
                torch.pow(qdx[:, :, None] - kdx[:, None, :], 2)
                + torch.pow(qdy[:, :, None] - kdy[:, None, :], 2)
                + torch.pow(qdz[:, :, None] - kdz[:, None, :], 2)
            ).sqrt()
            distance_buckets = torch.minimum(
                torch.maximum(
                    (distances - MIN_DISTANCE)
                    / (self.max_distance - MIN_DISTANCE)
                    * (self.n_distance_buckets - 1),
                    torch.tensor(0, dtype=distances.dtype, device=device),
                ).to(torch.int32),
                torch.tensor(
                    self.n_distance_buckets - 1, dtype=torch.int32, device=device
                ),
            )

        if qa is not None:
            cos = (
                qax[:, :, None] * kax[:, None, :]
                + qay[:, :, None] * kay[:, None, :]
                + qaz[:, :, None] * kaz[:, None, :]
            )
            cx = qay[:, :, None] * kaz[:, None, :] - qaz[:, :, None] * kay[:, None, :]
            cy = qaz[:, :, None] * kax[:, None, :] - qax[:, :, None] * kaz[:, None, :]
            cz = qax[:, :, None] * kay[:, None, :] - qay[:, :, None] * kax[:, None, :]
            sin = torch.sqrt(torch.pow(cx, 2) + torch.pow(cy, 2) + torch.pow(cz, 2))
            sin = torch.where(cz >= 0, sin, -sin)
            angles = torch.atan2(sin, cos)
            angles = torch.where(angles < 0, angles + 2 * torch.pi, angles)
            # Discretize angles
            angle_buckets = torch.minimum(
                torch.maximum(
                    angles / (2 * torch.pi) * self.n_angle_buckets,
                    torch.tensor(0, dtype=angles.dtype, device=device),
                ).to(torch.int32),
                torch.tensor(
                    self.n_angle_buckets - 1, dtype=torch.int32, device=device
                ),
            )

        if qd is not None and qa is not None:
            offsets = distance_buckets * self.n_angle_buckets + angle_buckets
            combined_n = self.n_distance_buckets * self.n_angle_buckets
        elif qd is not None:
            offsets = distance_buckets
            combined_n = self.n_distance_buckets
        else:  # USE_ANGLE_BIAS only
            offsets = angle_buckets
            combined_n = self.n_angle_buckets

        if qd is not None:
            is_nan_q = torch.isnan(qdx)
            is_nan_k = torch.isnan(kdx)
        else:  # USE_ANGLE_BIAS only
            is_nan_q = torch.isnan(qax)
            is_nan_k = torch.isnan(kax)
        mask = is_nan_q[:, :, None] | is_nan_k[:, None, :]
        offsets = torch.where(mask, combined_n, offsets)
        if qc is not None and self.confidence_threshold > 0:
            low_conf_q = qc < self.confidence_threshold
            low_conf_k = kc < self.confidence_threshold
            conf_mask = low_conf_q[:, :, None] | low_conf_k[:, None, :]
            offsets = torch.where(conf_mask & ~mask, combined_n + 1, offsets)

        # bw: (H, N_BUCKETS)
        # offsets: (B, L, L)
        # bias: (H, B, L, L) -> (batch_size, nheads, seqlen_q, seqlen_k)
        bias = self.weights[:, offsets].transpose(0, 1)
        return bias.diagonal_scatter(
            torch.zeros(
                (bias.size(0), bias.size(1), bias.size(2)),
                dtype=bias.dtype,
                device=device,
            ),
            dim1=2,
            dim2=3,
        )


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=False,
        batch_first=True,
        dropout=0.0,
        init_scaling=1 / math.sqrt(2),
        self_attention=False,
        causal=False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert batch_first
        self.batch_first = batch_first
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.causal = causal

        self.init_scaling = init_scaling
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        nn.init.xavier_uniform_(self.k_proj.weight, gain=self.init_scaling)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=self.init_scaling)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=self.init_scaling)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)

        # nn.init.xavier_uniform_(self.out_proj.weight, gain=self.init_scaling)
        nn.init.constant_(self.out_proj.weight, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def _transform_qkv(
        self,
        query,
        key,
        value,
        query_positions=None,
        key_positions=None,
        transform_query=True,
        transform_key=True,
        transform_value=True,
        attn_mask: Tensor | Atom3BiasParams | None = None,
    ):
        return query, key, value

    def _inner_attn(
        self,
        q,
        k,
        v,
        key_padding_mask=None,
        attn_mask=None,
        return_weights=False,
    ):
        assert not isinstance(attn_mask, Atom3BiasParams)
        # need to unpack inputs for usual mha attention...
        is_packed = False
        query_packed = q
        if type(q) is PackedTensorSequences:
            q = q.to_padded()
            is_packed = True
        if type(k) is PackedTensorSequences:
            # key padding mask is stored as the padding indices in the PackedTensor
            k, key_padding_mask = k.to_padded(return_mask=True)
        if type(v) is PackedTensorSequences:
            v = v.to_padded()

        dropout = self.dropout if self.training else 0
        attn, attn_weights = mha_attn(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            return_weights=return_weights,
            scaling=self.scaling,
            batch_first=self.batch_first,
            dropout=dropout,
            causal=self.causal,
        )

        # repack the output if the inputs were packed
        if is_packed:
            attn_packed = copy.copy(query_packed)
            attn_packed.x = attn
            attn = attn_packed

        return attn, attn_weights

    def forward_packed(
        self,
        query: PackedTensorSequences,
        key: Optional[PackedTensorSequences] = None,
        value: Optional[PackedTensorSequences] = None,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | Atom3BiasParams | None = None,
        return_weights: bool = False,
        return_projs: bool = False,
        transform_query: bool = True,
        transform_key: bool = True,
        transform_value: bool = True,
    ) -> Union[
        tuple[PackedTensorSequences, Optional[torch.Tensor]],
        tuple[
            PackedTensorSequences,
            Optional[torch.Tensor],
            tuple[PackedTensorSequences, PackedTensorSequences, PackedTensorSequences],
        ],
    ]:
        """
        When the input is packed, we can apply the projections efficiently to only the non-padding entries.
        """
        if self.self_attention:
            assert key is None and value is None
            key = value = query
        assert key is not None and value is not None

        query_positions = query.positions
        key_positions = key.positions

        if transform_query:
            qm = self.q_proj(query.x)
            qm = qm.view(-1, self.num_heads, self.head_dim)
        else:
            qm = None
        if transform_key:
            km = self.k_proj(key.x)
            km = km.view(-1, self.num_heads, self.head_dim)
        else:
            km = None
        if transform_value:
            vm = self.v_proj(value.x)
            vm = vm.view(-1, self.num_heads, self.head_dim)
        else:
            vm = None

        qm, km, vm = self._transform_qkv(
            qm,
            km,
            vm,
            query_positions=query_positions,
            key_positions=key_positions,
            transform_query=transform_query,
            transform_key=transform_key,
            transform_value=transform_value,
            attn_mask=attn_mask,
        )

        if transform_query:
            query = copy.copy(query)
            query.x = qm

        if transform_key:
            key = copy.copy(key)
            key.x = km

        if transform_value:
            value = copy.copy(value)
            value.x = vm

        # now calculate the attention values
        context_packed, attn_weights = self._inner_attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            return_weights=return_weights,
        )

        # handle packing again...
        context = context_packed.x
        context = context.view(context.size(0), self.embed_dim)

        output = self.out_proj(context)

        # repack ...
        output_packed = copy.copy(context_packed)
        output_packed.x = output
        output = output_packed

        if return_projs:
            return (output, attn_weights, (query, key, value))
        else:
            return output, attn_weights

    def forward_padded(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: torch.Tensor | Atom3BiasParams | None = None,
        return_weights: bool = False,
        return_projs: bool = False,
        transform_query: bool = True,
        transform_key: bool = True,
        transform_value: bool = True,
    ) -> Union[
        tuple[torch.Tensor, Optional[torch.Tensor]],
        tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ]:
        """
        Normal MHA approach for padded inputs.
        """
        assert not isinstance(attn_mask, Atom3BiasParams)
        if self.self_attention:
            assert key is None and value is None
            key = value = query
        assert key is not None and value is not None

        if transform_query:
            query = self.q_proj(query).view(
                query.size(0), query.size(1), self.num_heads, self.head_dim
            )
        if transform_key:
            key = self.k_proj(key).view(
                key.size(0), key.size(1), self.num_heads, self.head_dim
            )
        if transform_value:
            value = self.v_proj(value).view(
                value.size(0), value.size(1), self.num_heads, self.head_dim
            )

        query, key, value = self._transform_qkv(
            query,
            key,
            value,
            transform_query=transform_query,
            transform_key=transform_key,
            transform_value=transform_value,
        )

        # now calculate the attention values
        context, attn_weights = self._inner_attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        context = context.view(context.size(0), context.size(1), self.embed_dim)
        output = self.out_proj(context)

        if return_projs:
            return (output, attn_weights, (query, key, value))
        else:
            return output, attn_weights

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
    ) -> Union[
        tuple[T, Optional[torch.Tensor]],
        tuple[T, Optional[torch.Tensor], tuple[T, T, T]],
    ]:
        # dispatch depending on whether input is Packed or unpacked
        packed_input = type(query) is PackedTensorSequences
        fn = self.forward_padded
        if packed_input:
            fn = self.forward_packed

        return fn(
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
