import copy
import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from poet.models.modules.packed_sequence import PackedTensorSequences


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
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        return_projs: bool = False,
        transform_query: bool = True,
        transform_key: bool = True,
        transform_value: bool = True,
    ) -> Tuple[PackedTensorSequences, Optional[torch.Tensor]]:
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
        attn_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        return_projs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Normal MHA approach for padded inputs.
        """
        if self.self_attention:
            assert key is None and value is None
            key = value = query
        assert key is not None and value is not None

        query = self.q_proj(query).view(
            query.size(0), query.size(1), self.num_heads, self.head_dim
        )
        key = self.k_proj(key).view(
            key.size(0), key.size(1), self.num_heads, self.head_dim
        )
        value = self.v_proj(value).view(
            value.size(0), value.size(1), self.num_heads, self.head_dim
        )

        query, key, value = self._transform_qkv(query, key, value)

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
        query: Union[torch.Tensor, PackedTensorSequences],
        key: Optional[Union[torch.Tensor, PackedTensorSequences]] = None,
        value: Optional[Union[torch.Tensor, PackedTensorSequences]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
        return_projs: bool = False,
    ) -> Tuple[Union[torch.Tensor, PackedTensorSequences], Optional[torch.Tensor]]:
        # dispatch depending on whether input is Packed or unpacked
        packed_input = type(query) is PackedTensorSequences
        fn = self.forward_padded
        if packed_input:
            fn = self.forward_packed

        return fn(
            query, key, value, key_padding_mask, attn_mask, return_weights, return_projs
        )


class SelfAttention2d(MultiheadAttention):
    """Compute self-attention over 2D input."""

    def __init__(self, embed_dim, num_heads, axis=None, dropout=0, max_size=67108864):
        super(SelfAttention2d, self).__init__(
            embed_dim, num_heads, dropout=dropout, self_attention=True
        )
        self.axis = axis
        self.max_size = max_size

    def forward(self, x, padding_mask=None):
        """
        x : num_rows X num_cols X batch_size X embed_dim
        padding_mask : batch_size X num_rows X num_cols
        """

        N, M, B, H = x.size()

        # reshape X depending on axis attention mode!
        axis = self.axis
        if axis is None:  # flatten over rows and cols for full N*M*N*M attention
            x = x.view(N * M, B, H)
            if padding_mask is not None:
                padding_mask = padding_mask.view(B, N * M)
        else:
            assert axis == 0 or axis == 1

            if axis == 0:  # attend along the row dimension
                x = x.view(N, M * B, H)
                if padding_mask is not None:
                    padding_mask = padding_mask.permute(2, 0, 1)
                    padding_mask = padding_mask.view(M * B, N)
            else:  # axis == 1
                x = x.transpose(0, 1)  # M,N,B,H
                x = x.view(M, N * B, H)
                if padding_mask is not None:
                    padding_mask = padding_mask.permute(1, 0, 2)
                    padding_mask = padding_mask.view(N * B, M)

        if self.max_size > 0 and x.size(0) ** 2 * x.size(1) > self.max_size:
            # attention matrix size times batch size will exceed maximum allowable entries
            # split into batches to make attention matrix RAM workable
            # calculating attention over batches helps reduce RAM when N or M are large

            # calculate the maximum batch size that ensures <= max_size entries
            batch_size = x.size(0) ** 2 // self.max_size
            if batch_size < 1:
                batch_size = 1  # might run out of RAM, but batch size can't be < 1

            h = []
            for i in range(0, x.size(1), batch_size):
                xi = x[:, i : i + batch_size]
                mask = None
                if padding_mask is not None:
                    mask = padding_mask[i : i + batch_size]
                h.append(
                    super(SelfAttention2d, self).forward(xi, key_padding_mask=mask)
                )
            h = torch.cat(h, 1)
        else:
            h = super(SelfAttention2d, self).forward(x, key_padding_mask=padding_mask)

        # transpose h back to input shape
        if axis is None:
            h = h.view(N, M, B, H)
        elif axis == 0:
            h = h.view(N, M, B, H)
        else:  # axis == 1
            h = h.view(M, N, B, H)
            h = h.transpose(0, 1)

        return h


class PairWeightedSelfAttention(nn.Module):
    """
    Self attention with edge feature attention for encoding relative positions.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        pair_dim=None,
        init_scaling=1 / math.sqrt(2),
        bias=False,
    ):
        super().__init__()
        self.embed_dim = self.kdim = self.vdim = embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.init_scaling = init_scaling

        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.pair_proj = None
        self.pair_dim = embed_dim
        if pair_dim is not None:
            self.pair_dim = pair_dim
            self.pair_proj = nn.Linear(self.pair_dim, embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=self.init_scaling)

        nn.init.constant_(self.out_proj.weight, 0.0)
        # nn.init.xavier_uniform_(self.out_proj.weight, gain=self.init_scaling)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.pair_proj is not None:
            nn.init.xavier_uniform_(self.pair_proj.weight, gain=self.init_scaling)

    def forward(
        self,
        query,
        pairs,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Length x Batch x Channel
        Pair shape: Batch x Length x Length x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        # attention weights on the nodes
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # attention weights on the edge representations
        pair_keys = pairs  # (B x L x L x C), q is (B*H x L x Hdim)
        if self.pair_proj is not None:
            pair_keys = self.pair_proj(pairs)

        pair_keys = pair_keys.permute(0, 3, 1, 2)  # (B x C x L x L)
        pair_keys = pair_keys.reshape(
            bsz * self.num_heads, self.head_dim, tgt_len, src_len
        )  # (B*H x Hdim x L x L)
        pair_keys = pair_keys.transpose(1, 2)  # (B*H x L x Hdim x L)

        pair_keys = pair_keys.reshape(
            bsz * self.num_heads * tgt_len, self.head_dim, src_len
        )  # (B*H*L x Hdim x L)
        q = q.reshape(
            bsz * self.num_heads * tgt_len, 1, self.head_dim
        )  # (B*H*L x 1 x Hdim)

        pair_attn_weights = torch.bmm(q, pair_keys)  # (B*H*L x 1 x L)
        pair_attn_weights = pair_attn_weights.view(
            bsz * self.num_heads, tgt_len, src_len
        )

        attn_weights = attn_weights + pair_attn_weights

        # mask and softmax then calculate values
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = attn_weights

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=1)

            return attn, attn_weights

        return attn


class ReducingAttention(nn.Module):
    """
    Attention mechanism for reducing sequence of vectors to a fixed sized vector.
    """

    def __init__(
        self,
        embed_dim,
        output_dim,
        num_heads,
        head_dim=64,
        init_scaling=1 / math.sqrt(2),
        include_null=False,
        activation=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = output_dim

        self.norm = nn.LayerNorm(embed_dim)

        self.include_null = include_null
        self.activation = activation

        self.scaling = self.head_dim**-0.5
        self.init_scaling = init_scaling

        self.v_proj = nn.Linear(embed_dim, head_dim * num_heads)
        self.query_module = nn.Linear(embed_dim, head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, output_dim)

        if self.include_null:
            self.null_weights = nn.Parameter(torch.zeros(num_heads))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.query_module.weight, gain=self.init_scaling)

        nn.init.constant_(self.out_proj.weight, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.include_null:
            nn.init.constant_(self.null_weights, 0.0)

    def forward(
        self,
        x,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Length x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        src_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        tgt_len = 1

        attn_weights = self.query_module(self.norm(x))  # L x B x H
        attn_weights = (
            attn_weights.view(-1, bsz * self.num_heads).transpose(0, 1).unsqueeze(1)
        )  # B*NHEAD x 1 x L

        # apply padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # apply padding mask
        if self.include_null:
            null_weights = (
                self.null_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            )  # 1 x H x 1 x 1
            null_weights = null_weights.expand(bsz, self.num_heads, 1, 1)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = torch.cat([null_weights, attn_weights], dim=3)
            src_len += 1

        # calculate value projection
        v = self.v_proj(x)  # L x B x VDIM
        if self.activation is not None:
            v = self.activation(v)
        if self.include_null:
            null_values = torch.zeros(1, bsz, v.size(2), device=v.device, dtype=v.dtype)
            v = torch.cat([null_values, v], dim=0)
        v = (
            v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        )  # B*NHEAD x L x HDIM

        # attention weights and aggregate
        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = attn_weights

        attn = torch.bmm(attn_probs, v)  # B*NHEAD x 1 x HDIM
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        # reshape back and drop the tgt_len dim
        attn = attn.transpose(0, 1).contiguous().view(bsz, -1)  # B x VDIM

        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, src_len)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=1)

            return attn, attn_weights

        return attn
