import copy
from typing import Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor

from poet.models.modules.activation import gelu
from poet.models.modules.attention import MultiheadAttention
from poet.models.modules.attention_flash import FlashMultiheadAttention
from poet.models.modules.packed_sequence import PackedTensorSequences

T = TypeVar("T", Tensor, PackedTensorSequences)


class TransformerEncoderLayer(nn.Module):
    """Implements a pre-layer norm transformer encoder layer."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        activation=nn.GELU(),
        dropout=0,
        use_qkv_bias=False,
        batch_first=False,
        causal=False,
    ):
        super().__init__()

        assert (
            batch_first
        ), "Flash Attention requires batch first, make sure your transformer uses batch_first=True."

        self.dim = d_model
        self.dim_feedforward = dim_feedforward
        self.num_heads = nhead
        self.activation = activation

        self.self_attn = self._init_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _init_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
        causal=False,
    ):
        # return MultiheadAttention(d_model, nhead, self_attention=True, dropout=dropout, bias=use_qkv_bias, batch_first=batch_first)
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
        self.linear1.reset_parameters()
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def forward_packed(
        self,
        x: PackedTensorSequences,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_memory: bool = False,
    ) -> Union[
        Tuple[PackedTensorSequences, Optional[torch.Tensor]],
        Tuple[
            PackedTensorSequences,
            Optional[torch.Tensor],
            Tuple[PackedTensorSequences, PackedTensorSequences],
        ],
    ]:
        """
        When the input is packed, we can apply token-wise operations to only non-padding tokens.
        """
        x_packed = x
        x_flat = x_packed.x

        x_norm = copy.copy(x_packed)
        x_norm.x = self.norm1(x_flat)

        if not return_memory:
            h_packed, attn_weights = self.self_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_attention,
            )
        else:
            h_packed, attn_weights, (_, key, value) = self.self_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_attention,
                return_projs=return_memory,
            )

        h = h_packed.x
        h = self.dropout1(h) + x_flat

        h_norm = self.norm2(h)
        h = h + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(h_norm))))
        )

        h_packed.x = h

        if return_attention:
            return h_packed, attn_weights
        if return_memory:
            return h_packed, attn_weights, (key, value)

        return h_packed

    def forward_padded(
        self,
        x: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x_norm = self.norm1(x)
        h, attn_weights = self.self_attn(
            x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_attention,
        )

        h = self.dropout1(h) + x

        h_norm = self.norm2(h)
        h = h + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(h_norm))))
        )

        if return_attention:
            return h, attn_weights

        return h

    def forward(
        self,
        x: Union[Tensor, PackedTensorSequences],
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[Union[Tensor, PackedTensorSequences], Optional[torch.Tensor]]:
        fn = self.forward_padded
        if type(x) is PackedTensorSequences:
            fn = self.forward_packed
        return fn(x, src_mask, src_key_padding_mask, return_attention)


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
    ):
        super().__init__()

        assert (
            batch_first
        ), "Flash Attention requires batch first, make sure your transformer uses batch_first=True."

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
        self.multihead_attn = self._init_cross_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

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
        self.multihead_attn.reset_parameters()
        self.linear1.reset_parameters()
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

    def forward_packed(
        self,
        tgt: PackedTensorSequences,
        memory: PackedTensorSequences,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_self_attention: bool = False,
        return_cross_attention: bool = False,
    ) -> Tuple[PackedTensorSequences, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        )
        tgt = copy.copy(tgt)
        tgt.x = tgt.x + self.dropout1(tgt2.x)

        tgt_norm = copy.copy(tgt)
        tgt_norm.x = self.norm2(tgt.x)
        tgt2, attn = self.multihead_attn(
            tgt_norm,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            return_weights=return_cross_attention,
        )
        tgt = copy.copy(tgt)
        tgt.x = tgt.x + self.dropout2(tgt2.x)

        tgt2 = self.linear2(self.dropout(gelu(self.linear1(self.norm3(tgt.x)))))
        tgt.x = tgt.x + self.dropout3(tgt2)
        return tgt, attn, attn_self

    def forward_packed_from_key_value(
        self,
        tgt: PackedTensorSequences,
        key: PackedTensorSequences,
        value: PackedTensorSequences,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_self_attention: bool = False,
        return_cross_attention: bool = False,
    ) -> Tuple[PackedTensorSequences, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
        )
        tgt = copy.copy(tgt)
        tgt.x = tgt.x + self.dropout1(tgt2.x)

        tgt_norm = copy.copy(tgt)
        tgt_norm.x = self.norm2(tgt.x)
        tgt2, attn = self.multihead_attn.forward_packed(
            tgt_norm,
            key,
            value,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            return_weights=return_cross_attention,
            transform_query=True,
            transform_key=False,
            transform_value=False,
        )
        tgt = copy.copy(tgt)
        tgt.x = tgt.x + self.dropout2(tgt2.x)

        tgt2 = self.linear2(self.dropout(gelu(self.linear1(self.norm3(tgt.x)))))
        tgt.x = tgt.x + self.dropout3(tgt2)
        return tgt, attn, attn_self

    def forward_padded(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_self_attention: bool = False,
        return_cross_attention: bool = False,
    ) -> Tuple[Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        tgt2 = self.linear2(self.dropout(gelu(self.linear1(self.norm3(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn, attn_self

    def forward(
        self,
        tgt: Union[Tensor, PackedTensorSequences],
        memory: Union[Tensor, PackedTensorSequences],
        tgt_mask: Optional[Tensor] = None,
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
        return_attention: bool = False,
        activation_checkpointing: bool = False,
        **kwargs,
    ):
        assert not return_attention
        for layer in self.layers:
            if not activation_checkpointing:
                x = layer(
                    x,
                    return_attention=return_attention,
                    **kwargs,
                )
            else:
                x = checkpoint.checkpoint(
                    layer,
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
    ):
        super().__init__()

        assert (
            batch_first
        ), "Flash Attention requires batch first, make sure your transformer uses batch_first=True."

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
        self.multihead_attn = self._init_multi_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

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
        self.multihead_attn.reset_parameters()
        self.linear1.reset_parameters()
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm3.reset_parameters()

    def forward_packed(
        self,
        x: PackedTensorSequences,
        seqs_cu_seqlens: Tensor,
        seqs_cu_seqlens_cpu: Optional[Tensor],
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
        return_memory: bool = False,
        patch_size: Optional[int] = None,
        segment_sizes_cpu: Optional[torch.Tensor] = None,
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
        x = copy.copy(x)
        x.x = x.x + self.dropout1(x2.x)

        # apply the sequence-of-sequence attention layer on the reshaped sequences
        x_norm = copy.copy(x)
        if patch_size is not None:
            assert patch_size == 0
            assert segment_sizes_cpu is not None
            x_norm.x = x.x[x.cu_seqlens_cpu[:-1].long()]
            x_norm.x = self.norm2(x_norm.x)
            n_seqs = (segment_sizes_cpu > 0).long().sum(dim=1)
            x_norm.cu_seqlens = (
                F.pad(n_seqs.cumsum(dim=0), (1, 0))
                .type(torch.int32)
                .to(x_norm.cu_seqlens.device)
            )
            x_norm.cu_seqlens_cpu = F.pad(n_seqs.cumsum(dim=0), (1, 0)).type(
                torch.int32
            )
            x_norm.max_s = n_seqs.max()
            x_norm.positions = None
            nonzero_segment_sizes = (
                segment_sizes_cpu[segment_sizes_cpu > 0].flatten().to(x_norm.x.device)
            )
            assert not x_norm.to_paddedable
            assert src_key_padding_mask is None
        else:
            x_norm.x = self.norm2(x.x)
            x_norm.cu_seqlens = seqs_cu_seqlens  # "reshape" the packed sequences
            if seqs_cu_seqlens_cpu is not None:
                x_norm.cu_seqlens_cpu = seqs_cu_seqlens_cpu
            else:
                x_norm.cu_seqlens_cpu = seqs_cu_seqlens.cpu()
            x_norm.max_s = x_norm.cu_seqlens_cpu.max()
            if x_norm.to_paddedable:
                seqs_seqlens = seqs_cu_seqlens.diff()
                x_norm.indices = x_norm.compute_indices(seqs_seqlens)
                x_norm.batch_size = seqs_seqlens.numel()
            if src_key_padding_mask is not None:
                src_key_padding_mask = src_key_padding_mask.view(
                    -1, src_key_padding_mask.size(-1)
                )
        if not return_memory:
            x2, attn = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
            )
            key, value = None, None
        else:
            x2, attn, (_, key, value) = self.multihead_attn(
                x_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                return_weights=return_multi_attention,
                return_projs=return_memory,
            )
        patch = copy.copy(x2) if return_patch else None
        if patch_size is not None:
            indices = torch.arange(
                nonzero_segment_sizes.numel(), device=x2.x.device
            ).repeat_interleave(nonzero_segment_sizes)
            x2.x = F.embedding(indices, x2.x)
        x = copy.copy(x)
        x.x = x.x + self.dropout2(x2.x)

        x2 = self.linear2(self.dropout(gelu(self.linear1(self.norm3(x.x)))))
        x.x = x.x + self.dropout3(x2)

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
        seqs_cu_seqlens_cpu: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
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

        x2 = self.linear2(self.dropout(gelu(self.linear1(self.norm3(x)))))
        x = x + self.dropout3(x2)

        if return_attention:
            return x, (attn_self, attn)
        if return_memory:
            return x, (attn_self, attn), (key, value)
        return x

    def forward(
        self,
        x: T,
        seqs_cu_seqlens: Optional[Tensor] = None,
        seqs_cu_seqlens_cpu: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
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
            seqs_cu_seqlens_cpu=seqs_cu_seqlens_cpu,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            return_self_attention=return_attention,
            return_multi_attention=return_attention,
            return_memory=return_memory,
        )


class TieredTransformerDecoderLayer(nn.Module):
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
    ):
        super().__init__()

        assert (
            batch_first
        ), "Flash Attention requires batch first, make sure your transformer uses batch_first=True."

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
        self.multihead_attn = self._init_multi_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
            causal=causal,
        )
        self.cross_attn = self._init_cross_mha_module(
            d_model,
            nhead,
            dropout=dropout,
            use_qkv_bias=use_qkv_bias,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
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

    def _init_cross_mha_module(
        self,
        d_model,
        nhead,
        dropout=0,
        use_qkv_bias=False,
        batch_first=True,
    ):
        """
        Initialize the cross attention module.
        """
        return FlashMultiheadAttention(
            d_model,
            nhead,
            self_attention=False,
            dropout=dropout,
            bias=use_qkv_bias,
            batch_first=batch_first,
            causal=False,
        )

    def reset_parameters(self):
        self.self_attn.reset_parameters()
        self.multihead_attn.reset_parameters()
        self.cross_attn.reset_parameters()
        self.linear1.reset_parameters()
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.norm_cross.reset_parameters()
        self.norm3.reset_parameters()

    def forward_packed(
        self,
        x: PackedTensorSequences,
        memory: PackedTensorSequences,
        seqs_cu_seqlens: Tensor,
        src_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
    ) -> Union[
        PackedTensorSequences,
        tuple[PackedTensorSequences, tuple[Tensor, Tensor]],
        tuple[
            PackedTensorSequences,
            tuple[Optional[Tensor], Optional[Tensor]],
            PackedTensorSequences,
        ],
    ]:
        return self.forward_packed_from_key_value(
            x=x,
            key=memory,
            value=memory,
            seqs_cu_seqlens=seqs_cu_seqlens,
            src_mask=src_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            transform_key_value=True,
            return_attention=return_attention,
            return_self_attention=return_self_attention,
            return_multi_attention=return_multi_attention,
        )

    def forward_packed_from_key_value(
        self,
        x: PackedTensorSequences,
        key: PackedTensorSequences,
        value: PackedTensorSequences,
        seqs_cu_seqlens: Tensor,
        src_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        transform_key_value: bool = False,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
    ) -> Union[
        PackedTensorSequences,
        tuple[PackedTensorSequences, tuple[Tensor, Tensor]],
        tuple[
            PackedTensorSequences,
            tuple[Optional[Tensor], Optional[Tensor]],
            PackedTensorSequences,
        ],
    ]:
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
        x = copy.copy(x)
        x.x = x.x + self.dropout1(x2.x)

        # apply the sequence-of-sequence attention layer on the reshaped sequences
        x_norm = copy.copy(x)
        x_norm.x = self.norm2(x.x)
        x_norm.cu_seqlens = seqs_cu_seqlens  # "reshape" the packed sequences
        x_norm.cu_seqlens_cpu = seqs_cu_seqlens.cpu()
        x_norm.max_s = x_norm.cu_seqlens_cpu.max()
        if x_norm.to_paddedable:
            seqs_seqlens = seqs_cu_seqlens.diff()
            x_norm.indices = x_norm.compute_indices(seqs_seqlens)
            x_norm.batch_size = seqs_seqlens.numel()
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(
                -1, src_key_padding_mask.size(-1)
            )
        x2, attn = self.multihead_attn(
            query=x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            return_weights=return_multi_attention,
        )
        x = copy.copy(x)
        x.x = x.x + self.dropout2(x2.x)

        # apply cross attention
        x_norm = copy.copy(x_norm)
        x_norm.x = self.norm_cross(x.x)
        x2, _ = self.cross_attn.forward_packed(
            query=x_norm,
            key=key,
            value=value,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            return_weights=False,
            return_projs=False,
            transform_query=True,
            transform_key=transform_key_value,
            transform_value=transform_key_value,
        )
        x = copy.copy(x)
        x.x = x.x + self.dropout_cross(x2.x)

        x2 = self.linear2(self.dropout(gelu(self.linear1(self.norm3(x.x)))))
        x.x = x.x + self.dropout3(x2)

        if return_attention:
            return x, (attn_self, attn)
        return x

    def forward_padded(
        self,
        x: Tensor,
        memory: Tensor,
        seqs_cu_seqlens: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
        return_self_attention: bool = False,
        return_multi_attention: bool = False,
    ) -> Union[
        Tensor,
        tuple[Tensor, tuple[Tensor, Tensor]],
        tuple[Tensor, tuple[Optional[Tensor], Optional[Tensor]], Tensor],
    ]:
        raise NotImplementedError

    def forward(
        self,
        x: T,
        memory: T,
        seqs_cu_seqlens: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[
        T,
        tuple[T, tuple[Tensor, Tensor]],
        tuple[T, tuple[Optional[Tensor], Optional[Tensor]], T],
    ]:
        fn = self.forward_padded
        if type(x) is PackedTensorSequences:
            assert seqs_cu_seqlens is not None
            fn = self.forward_packed
        return fn(
            x,
            memory,
            seqs_cu_seqlens=seqs_cu_seqlens,
            src_mask=src_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_self_attention=return_attention,
            return_multi_attention=return_attention,
        )


class PreTransformerSummaryLayer(nn.Module):
    """TransformerSummaryLayer uses multi-head-attn and feedforward network."""

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0, use_feedforward=True
    ):
        super(PreTransformerSummaryLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # Implementation of Feedforward model
        if use_feedforward:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        key: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt_norm = self.norm1(key)
        tgt2, attn = self.multihead_attn(
            tgt_norm,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = key + self.dropout1(tgt2)

        if hasattr(self, "linear1"):
            tgt2 = self.linear2(self.dropout(gelu(self.linear1(self.norm2(tgt)))))
            tgt = tgt + self.dropout2(tgt2)
        return tgt, attn
