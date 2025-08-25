# This file includes code adapted from:
# https://github.com/Dao-AILab/flash-attention/blob/23e8fa5a263d1c7122bc46a86ef32030ee7130f9/flash_attn/utils/generation.py
# Copyright (c) 2023, Tri Dao
# Licensed under the BSD 3-Clause License
# Modifications made by OpenProtein.AI

import copy
import os
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Literal, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.bert_padding import index_first_axis, rearrange
from flash_attn.bert_padding import pad_input as pad_input_flash
from flash_attn.bert_padding import unpad_input as unpad_input_flash
from flash_attn.utils.generation import InferenceParams, sample
from tqdm import tqdm

from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet
from poet_2.models.modules.attention import Atom3BiasParams
from poet_2.models.modules.norm import RMSNorm
from poet_2.models.modules.packed_sequence import PackedTensorSequences, get_mask
from poet_2.models.modules.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
)
from poet_2.models.modules.transformer_rotary import (
    RotaryTransformerDecoderLayer,
    TieredRotaryTransformerEncoderLayer,
)


# Whether or not to force attention computations with atom3 bias to use flash attn,
#   which does mostly the right thing if the inputs contain no structure (and thus no
#   atom3).
# This only makes sense to enable for sequence only inputs, for which the atom3 bias
#   is just a constant per head, which our custom flash attn implementation can handle
#   (constant here means that every pair of residues gets the same bias rather than a
#   bias that depends on the atom3 input; it's constant for sequence only inputs
#   because there is no atom3 input).
# There is still one other discrepency though - when using flash attn with sequence
#   only inputs, the gradient does not propagate to the constant per head bias i.e.
#   these biases become frozen rather than trainable parameters. This is probably not
#   important though, so it probably makes sense to leverage flash attn for its
#   performance and battle tested code.
POET_2_ATOM3_BIAS_FORCE_FLASH_ATTN = (
    os.environ.get("POET_2_ATOM3_BIAS_FORCE_FLASH_ATTN", "0") == "1"
)
N_ATOMB = 36


def pad_input(hidden_states, indices, batch, seqlen) -> tuple[torch.Tensor, None]:
    # mimic interface of pad_input in packed_sequence
    return pad_input_flash(hidden_states, indices, batch, seqlen), None  # type: ignore


def unpad_input(
    hidden_states, attention_mask
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    assert hidden_states.size()[:2] == attention_mask.size()
    return unpad_input_flash(hidden_states, attention_mask)  # type: ignore


def tie_module_weights(
    src_module: nn.Module, target_module: nn.Module, remove_duplicate: bool = True
):
    src_named_params = list(
        src_module.named_parameters(remove_duplicate=remove_duplicate)
    )
    target_named_params = list(
        target_module.named_parameters(remove_duplicate=remove_duplicate)
    )
    if remove_duplicate:
        assert len(src_named_params) == len(list(src_module.parameters()))
        assert len(target_named_params) == len(list(target_module.parameters()))
    assert len(src_named_params) == len(target_named_params)
    for sk, sv in src_named_params:
        module = target_module
        parts, parameter_name = sk.rsplit(".", 1)
        for part in parts.split("."):
            module = getattr(module, part)
        old_param = getattr(module, parameter_name)
        assert isinstance(old_param, nn.Parameter)
        assert old_param.size() == sv.size()
        setattr(module, parameter_name, sv)


def tie_linear_weights(src_module: nn.Linear, target_module: nn.Linear):
    target_module.weight = src_module.weight
    target_module.bias = src_module.bias


def tie_cross_attn_kv_weights(
    src_module: TransformerDecoderLayer, target_module: TransformerDecoderLayer
):
    tie_linear_weights(
        src_module=src_module.multihead_attn.k_proj,
        target_module=target_module.multihead_attn.k_proj,
    )
    tie_linear_weights(
        src_module=src_module.multihead_attn.v_proj,
        target_module=target_module.multihead_attn.v_proj,
    )


def unpad_seqofseqs(
    xs: torch.Tensor, segment_sizes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs_seqlens = segment_sizes.sum(dim=1).type(torch.int32)
    attention_mask = ~get_mask(seqs_seqlens)
    xs, indices, _, _ = unpad_input(xs.unsqueeze(2), attention_mask)
    return seqs_seqlens, attention_mask, xs.squeeze(1), indices


def unpad_ys_refs(
    ys_refs: torch.Tensor, segment_sizes: torch.Tensor, src_segment_sizes: torch.Tensor
) -> torch.Tensor:
    ys_refs_in = ys_refs
    ys_refs = ys_refs + F.pad(
        src_segment_sizes[:-1].sum(dim=1).cumsum(dim=0), (1, 0)
    ).unsqueeze(1)
    ys_refs[ys_refs_in == -100] = -100
    return unpad_seqofseqs(ys_refs, segment_sizes)[2]


class PoET2Output(NamedTuple):
    self_attns: dict[int, torch.Tensor]
    cross_attns: dict[int, torch.Tensor]
    reprs: dict[int, PackedTensorSequences]
    logits: Optional[torch.Tensor]


def unpad_from_indices(
    hidden_states: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    return index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)  # type: ignore


class PoET2(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        n_out: int | None = None,
        hidden_dim: int = 768,
        ff_dim: Optional[int] = None,
        n_layers: int = 6,
        nhead: int = 12,
        dropout: float = 0,
        use_multi_rotary: bool = True,
        use_cross_rotary: bool = True,
        activation: Literal["gelu", "silu"] = "silu",
        use_glu: bool = True,
        tied_cross_attn_kv: bool = True,
        norm_type: Literal["layer", "rms"] = "rms",
        final_norm: bool = True,
        in_seqid: bool = True,
        in_plddt: bool = True,
        in_s3di: int | None = None,  # vocab size
        atomx_layer_idxs: tuple[int, ...] | None = None,
        n_distance_buckets: int | None = 128,
        n_angle_buckets: int | None = 128,
        use_confidence_bucket: bool = True,
        confidence_threshold: float = 0.7,  # NOTE: 0-1 range, only relevant if use_confidence_bucket
        in_atomb: bool = False,  # atomic local backbone distances
        version: int = 0,  # version 1: rename atom3_layer_idxs -> atomx_layer_idxs
        tied_heads: bool = True,
        # for backwards compatibility only
        atom3_layer_idxs: tuple[int, ...] | None = None,
    ):
        assert n_distance_buckets or n_angle_buckets
        super().__init__()
        if version == 0:
            atomx_layer_idxs = atom3_layer_idxs
        else:
            assert atom3_layer_idxs is None
        self.n_vocab = n_vocab
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.token_embed = nn.Embedding(n_vocab, hidden_dim)
        self.seqid_embed = nn.Linear(3, hidden_dim) if in_seqid else None
        self.plddt_embed = nn.Linear(2, hidden_dim) if in_plddt else None
        self.s3di_embed = nn.Embedding(in_s3di, hidden_dim) if in_s3di else None
        self.atomx_layer_idxs = atomx_layer_idxs
        self.n_distance_buckets = n_distance_buckets
        self.n_angle_buckets = n_angle_buckets
        self.use_confidence_bucket = use_confidence_bucket
        self.confidence_threshold = confidence_threshold
        self.atomb_embed = nn.Linear(N_ATOMB * 2, hidden_dim) if in_atomb else None

        ff_dim = ff_dim or 4 * hidden_dim
        self.encoder = TransformerEncoder(
            encoder_layer=TieredRotaryTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                activation=getattr(F, activation),
                dropout=dropout,
                use_multi_rotary=use_multi_rotary,
                batch_first=True,
                causal=False,
                norm_type=norm_type,
                use_glu=use_glu,
                rotary_force_fp32=True,
                n_distance_buckets=n_distance_buckets,
                n_angle_buckets=n_angle_buckets,
                use_confidence_bucket=use_confidence_bucket,
            ),
            num_layers=n_layers,
        )
        self.tied_cross_attn_kv = tied_cross_attn_kv
        if tied_cross_attn_kv:
            # always necessary b/c we assume encoder and decoders are tied
            first_layer = self.encoder.layers[0]
            for layer in self.encoder.layers[1:]:
                tie_cross_attn_kv_weights(
                    src_module=first_layer,
                    target_module=layer,
                )
            del first_layer, layer

        def make_decoder(causal: bool, num_layers: int):
            decoder = TransformerDecoder(
                decoder_layer=RotaryTransformerDecoderLayer(  # type: ignore
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=ff_dim,
                    activation=getattr(F, activation),
                    dropout=dropout,
                    use_cross_rotary=use_cross_rotary,
                    batch_first=True,
                    causal=causal,
                    norm_type=norm_type,
                    use_glu=use_glu,
                    rotary_force_fp32=True,
                    n_distance_buckets=n_distance_buckets,
                    n_angle_buckets=n_angle_buckets,
                    use_confidence_bucket=use_confidence_bucket,
                ),
                num_layers=num_layers,
            )
            if tied_cross_attn_kv:
                first_layer = decoder.layers[0]
                for layer in decoder.layers[1:]:
                    tie_cross_attn_kv_weights(
                        src_module=first_layer,
                        target_module=layer,
                    )
            return decoder

        self.mlm_decoder = make_decoder(causal=False, num_layers=n_layers)
        tie_module_weights(
            src_module=self.encoder,
            target_module=self.mlm_decoder,
            remove_duplicate=False,
        )
        self.clm_decoder = make_decoder(causal=True, num_layers=n_layers)
        tie_module_weights(
            src_module=self.mlm_decoder,
            target_module=self.clm_decoder,
            remove_duplicate=False,
        )

        norm_cls = nn.LayerNorm if norm_type == "layer" else RMSNorm
        self.norm = norm_cls(hidden_dim) if final_norm else nn.Identity()

        self.tied_heads = tied_heads
        if tied_heads:
            self.head = nn.Linear(hidden_dim, n_out or n_vocab)
        else:
            self.mlm_head = nn.Linear(hidden_dim, n_out or n_vocab)
            self.clm_head = nn.Linear(hidden_dim, n_out or n_vocab)

        self.mask_token_s3di: int | None = None

        # # NOTE: this means the model will be saved with these weights having
        # #       requires_grad False... so only uncomment if necessary...
        # for layer in (
        #     list(self.encoder.layers)
        #     + list(self.mlm_decoder.layers)
        #     + list(self.clm_decoder.layers)
        # ):
        #     if not hasattr(layer, "bias_weights"):
        #         continue
        #     layer.bias_weights.requires_grad = False

    def _unpad_helper(
        self,
        xs: torch.Tensor,
        indices: torch.Tensor,
        xs_seqids: torch.Tensor | None,
        xs_plddts: torch.Tensor | None,
        xs_atomxs: torch.Tensor | None,
        xs_atombs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        dtype = next(iter(self.parameters())).dtype
        if (
            xs_seqids is not None
            and xs_plddts is not None
            and xs_atomxs is not None
            and xs_atombs is not None
        ):
            # combine, unpad all at once, and then separate
            combined = torch.cat(
                (
                    xs_seqids,
                    xs_plddts.unsqueeze(2),
                    xs_atomxs.flatten(start_dim=-2, end_dim=-1),
                    xs_atombs,
                ),
                dim=2,
            )
            combined = unpad_from_indices(combined, indices)
            assert xs_seqids.size(2) == 2
            assert xs_plddts.ndim == 2
            assert xs_atombs.size(2) == N_ATOMB
            xs_seqids = combined[:, :2]
            xs_plddts = combined[:, [2]]
            xs_atomxs = combined[:, 3:-N_ATOMB].reshape(-1, 3, 3).float()
            xs_atombs = combined[:, -N_ATOMB:]
        else:
            if xs_seqids is not None:
                xs_seqids = unpad_from_indices(xs_seqids, indices)
            if xs_plddts is not None:
                xs_plddts = unpad_from_indices(xs_plddts.unsqueeze(2), indices)
            if xs_atomxs is not None:
                xs_atomxs = unpad_from_indices(xs_atomxs, indices).float()
            if xs_atombs is not None:  # (B, L, N_ATOMB)
                xs_atombs = unpad_from_indices(xs_atombs, indices)
        if xs_seqids is not None:
            mask = torch.isnan(xs_seqids).any(dim=1)
            xs_seqids[mask] = 0
            xs_seqids = torch.cat((xs_seqids.to(dtype), ~mask.unsqueeze(1)), dim=1)
        elif xs_seqids is None and self.seqid_embed is not None:
            xs_seqids = torch.zeros((xs.size(0), 3), dtype=dtype, device=xs.device)
        if xs_plddts is not None:
            mask = torch.isnan(xs_plddts)
            xs_plddts /= 100  # compress plddts to 0-1 range
            xs_plddts[mask] = 0
            xs_plddts = torch.cat((xs_plddts.to(dtype), ~mask), dim=1)
        elif xs_plddts is None and self.plddt_embed is not None:
            xs_plddts = torch.zeros((xs.size(0), 2), dtype=dtype, device=xs.device)
        if xs_atombs is not None:  # (B, L, N_ATOMB)
            mask = torch.isnan(xs_atombs)
            xs_atombs /= 10  # normalize to ~0-1 range
            xs_atombs[mask] = 0
            xs_atombs = torch.cat((xs_atombs.to(dtype), ~mask), dim=1)
        elif xs_atombs is None and self.atomb_embed is not None:
            xs_atombs = torch.zeros(
                (xs.size(0), N_ATOMB * 2), dtype=dtype, device=xs.device
            )
        return xs_seqids, xs_plddts, xs_atomxs, xs_atombs

    def _embed_and_pack(
        self,
        xs: torch.Tensor,
        xs_seqids: torch.Tensor | None,
        xs_plddts: torch.Tensor | None,
        xs_s3dis: torch.Tensor | None,
        xs_atomxs: torch.Tensor | None,
        xs_atombs: torch.Tensor | None,
        segment_sizes: torch.Tensor,
        bias_need_pad_input: bool = False,
    ) -> tuple[PackedTensorSequences, torch.Tensor, torch.Tensor, Atom3BiasParams]:
        """
        Returns

        h: xs embedded as PackedTensorSequences of sequences
        seqs_cu_seqlens: can be used to convert to seq of seqs
        indices: can be used to pad into seq of seqs
        """
        seqs_seqlens, _, xs, indices = unpad_seqofseqs(xs, segment_sizes)
        if xs_s3dis is not None:
            xs_s3dis = unpad_from_indices(xs_s3dis.unsqueeze(2), indices).squeeze(1)
        else:
            if self.mask_token_s3di is not None:
                xs_s3dis = torch.full_like(xs, self.mask_token_s3di)
            else:
                assert self.s3di_embed is None
        xs_seqids, xs_plddts, xs_atomxs, xs_atombs = self._unpad_helper(
            xs=xs,
            indices=indices,
            xs_seqids=xs_seqids,
            xs_plddts=xs_plddts,
            xs_atomxs=xs_atomxs,
            xs_atombs=xs_atombs,
        )
        h = self.token_embed.forward(xs)
        if self.seqid_embed is not None:
            h += self.seqid_embed(xs_seqids)
        if self.plddt_embed is not None:
            h += self.plddt_embed(xs_plddts)
        if self.s3di_embed is not None:
            h += self.s3di_embed(xs_s3dis)
        if self.atomb_embed is not None:
            h += self.atomb_embed(xs_atombs)

        nonzero_segment_sizes = (
            segment_sizes[segment_sizes > 0].flatten().type(torch.int32)
        )
        cu_seqlens = F.pad(
            nonzero_segment_sizes.cumsum(dim=0, dtype=nonzero_segment_sizes.dtype),
            (1, 0),
        )
        h = PackedTensorSequences(
            packed_tensor=h,
            positions=torch.cat(
                [
                    torch.arange(segment_size, dtype=xs.dtype, device=xs.device)
                    for segment_size in nonzero_segment_sizes
                ]
            ),
            cu_seqlens=cu_seqlens,
            max_s=nonzero_segment_sizes.max().item(),
            # only needed for unpadding (used in standard attn)
            to_paddedable=False,
            indices=None,
            batch_size=None,
        )
        seqs_cu_seqlens = F.pad(
            seqs_seqlens.cumsum(dim=0, dtype=seqs_seqlens.dtype), (1, 0)
        )

        if xs_atomxs is not None:
            coords = xs_atomxs[:, 1]
            planes = torch.linalg.cross(
                xs_atomxs[:, 0] - xs_atomxs[:, 1], xs_atomxs[:, 2] - xs_atomxs[:, 1]
            )
            planes /= torch.linalg.norm(planes, dim=1, keepdim=True)
            confidences = xs_plddts[:, 0].float()
        else:
            coords = torch.full(
                (xs.size(0), 3), torch.nan, dtype=torch.float32, device=xs.device
            )
            planes = torch.full(
                (xs.size(0), 3), torch.nan, dtype=torch.float32, device=xs.device
            )
            confidences = torch.full(
                (xs.size(0),), torch.nan, dtype=torch.float32, device=xs.device
            )

        def attn_mask_pad_input_fn() -> Callable[[torch.Tensor], torch.Tensor]:
            indices = PackedTensorSequences.compute_indices(nonzero_segment_sizes)
            indices = indices.to(xs.device)
            return lambda x_unpadded: pad_input(
                x_unpadded, indices, len(cu_seqlens) - 1, h.max_s
            )[0]

        attn_mask = Atom3BiasParams(
            coords=coords.contiguous() if self.n_distance_buckets else None,
            planes=planes.contiguous() if self.n_angle_buckets else None,
            confidences=(
                confidences.contiguous() if self.use_confidence_bucket else None
            ),
            weights=None,
            n_distance_buckets=self.n_distance_buckets or 0,
            n_angle_buckets=self.n_angle_buckets or 0,
            use_confidence_bucket=self.use_confidence_bucket,
            confidence_threshold=self.confidence_threshold,
            pad_input=attn_mask_pad_input_fn() if bias_need_pad_input else None,
        )
        return h, seqs_cu_seqlens, indices, attn_mask

    def to_seq_of_seqs(
        self, x: PackedTensorSequences, seqs_cu_seqlens: torch.Tensor
    ) -> PackedTensorSequences:
        x.cu_seqlens = seqs_cu_seqlens  # "reshape" the packed sequences
        xs_seqs_seqlens = seqs_cu_seqlens.diff()
        x.max_s = xs_seqs_seqlens.max().item()  # type: ignore
        if x.to_paddedable:
            x.to_paddedable = False
            x.make_to_paddedable()
        return x

    def encoder_outputs(
        self,
        *,
        xs: torch.Tensor,
        segment_sizes: torch.Tensor,
        xs_plddts: torch.Tensor | None = None,
        xs_s3dis: torch.Tensor | None = None,
        xs_atomxs: torch.Tensor | None = None,
        xs_atombs: torch.Tensor | None = None,
        self_attn_layers: set[int] | tuple[int, ...] = (),
        cross_attn_layers: set[int] | tuple[int, ...] = (),
        repr_layers: set[int] | tuple[int, ...] = (-1,),
        return_logits: bool = False,
    ) -> PoET2Output:
        if len(self_attn_layers) > 0 or len(cross_attn_layers) > 0:
            warnings.warn("API for attn is untested and not stable")
        xs_h, xs_seqs_cu_seqlens, xs_indices, attn_mask = self._embed_and_pack(
            xs=xs,
            xs_seqids=None,
            xs_plddts=xs_plddts,
            xs_s3dis=xs_s3dis,
            xs_atomxs=xs_atomxs,
            xs_atombs=xs_atombs,
            segment_sizes=segment_sizes,
            bias_need_pad_input=xs.device.type == "cpu"
            or len(self_attn_layers) > 0
            or len(cross_attn_layers) > 0,
        )
        if (
            xs.device.type == "cpu"
            or len(self_attn_layers) > 0
            or len(cross_attn_layers) > 0
        ):
            xs_h.make_to_paddedable()
        self_attns: dict[int, torch.Tensor] = {}
        cross_attns: dict[int, torch.Tensor] = {}
        hidden_representations: dict[int, PackedTensorSequences] = {}
        if 0 in repr_layers:
            hidden_representations[0] = xs_h
        layer: TieredRotaryTransformerEncoderLayer
        for i, layer in enumerate(self.encoder.layers):
            last_layer = i == len(self.encoder.layers) - 1
            return_self_attention = i in self_attn_layers or (
                -1 in self_attn_layers and last_layer
            )
            return_cross_attention = i in cross_attn_layers or (
                -1 in cross_attn_layers and last_layer
            )
            if self.atomx_layer_idxs is not None and i not in self.atomx_layer_idxs:
                layer_attn_mask = None
            elif POET_2_ATOM3_BIAS_FORCE_FLASH_ATTN:
                layer_attn_mask = (
                    layer.bias_weights[:, -2 if self.use_confidence_bucket else -1]
                    .float()
                    .contiguous()
                    .view(1, -1, 1, 1)
                    .detach()
                )
            else:
                layer_attn_mask = replace(attn_mask, weights=layer.bias_weights)
            result = layer.forward_packed(
                xs_h,
                src_mask=layer_attn_mask,
                seqs_cu_seqlens=xs_seqs_cu_seqlens,
                return_self_attention=return_self_attention,
                return_multi_attention=return_cross_attention,
            )
            if return_self_attention or return_cross_attention:
                xs_h, (self_attn, cross_attn) = result
            else:
                xs_h = result
            if return_self_attention:
                self_attns[i] = self_attn
            if return_cross_attention:
                cross_attns[i] = cross_attn
            if i + 1 in repr_layers:
                hidden_representations[i + 1] = xs_h
        if -1 in self_attn_layers:
            self_attns[-1] = self_attn
            if i not in self_attn_layers:
                del self_attns[i]
        if -1 in cross_attn_layers:
            cross_attns[-1] = cross_attn
            if i not in cross_attn_layers:
                del cross_attns[i]
        xs_h.x = self.norm.forward(xs_h.x)
        if i + 2 in repr_layers:
            hidden_representations[i + 2] = xs_h
        if -1 in repr_layers:
            hidden_representations[-1] = xs_h

        if return_logits:
            head = self.head if self.tied_heads else self.mlm_head
            logits, _ = pad_input(
                hidden_states=head.forward(xs_h.x),
                indices=xs_indices,
                batch=xs.size(0),
                seqlen=xs.size(1),
            )
        else:
            logits = None
        return PoET2Output(
            self_attns=self_attns,
            cross_attns=cross_attns,
            reprs={
                k: self.to_seq_of_seqs(v, xs_seqs_cu_seqlens)
                for k, v in hidden_representations.items()
            },
            logits=logits,
        )

    def get_decoder_memory(
        self, *, xs_h: PackedTensorSequences, decoder: TransformerDecoder | None = None
    ) -> list[PackedTensorSequences]:
        decoder = decoder or self.clm_decoder
        memory = []
        layer: RotaryTransformerDecoderLayer
        for layer in decoder.layers:
            attn = layer.multihead_attn
            km = attn.k_proj(xs_h.x).view(-1, attn.num_heads, attn.head_dim)
            vm = attn.v_proj(xs_h.x).view(-1, attn.num_heads, attn.head_dim)
            # we don't transform value, b/c we never transform value for rotary
            _, km, _ = attn._transform_qkv(
                None,
                km,
                None,
                query_positions=None,
                key_positions=xs_h.positions,
                transform_query=False,
                transform_key=True,
                transform_value=False,
            )
            key = copy.copy(xs_h)
            key.x = km
            value = copy.copy(xs_h)
            value.x = vm
            memory.append(key)
            memory.append(value)
            if self.tied_cross_attn_kv:
                break
        if self.tied_cross_attn_kv:
            memory = memory * len(decoder.layers)
        return memory

    def outputs_from_memory(
        self,
        *,
        decoder: TransformerDecoder,
        memory: list[PackedTensorSequences],
        ys: torch.Tensor,
        ys_segment_sizes: torch.Tensor | None = None,
        ys_plddts: torch.Tensor | None = None,
        ys_s3dis: torch.Tensor | None = None,
        ys_atomxs: torch.Tensor | None = None,
        ys_atombs: torch.Tensor | None = None,
        ys_seqids: torch.Tensor | None = None,
        ys_refs: torch.Tensor | None = None,
        ys_ref_values: torch.Tensor | None = None,
        self_attn_layers: set[int] | tuple[int, ...] = (),
        cross_attn_layers: set[int] | tuple[int, ...] = (),
        repr_layers: set[int] | tuple[int, ...] = (),
        return_logits: bool = True,
        inference_params: Optional[InferenceParams] = None,
        decoder_norm: nn.Module | None = None,
        decoder_head: nn.Module | None = None,
    ) -> PoET2Output:
        decoder_norm = decoder_norm or self.norm
        if decoder_head is None:
            if self.tied_heads:
                decoder_head = self.head
            else:
                if decoder is self.mlm_decoder:
                    decoder_head = self.mlm_head
                else:
                    assert decoder is self.clm_decoder
                    decoder_head = self.clm_head
        if ys_refs is not None:
            assert ys_ref_values is not None
        if inference_params is not None:
            assert ys_plddts is None and ys_atomxs is None and ys_atombs is None
        B, L_y = ys.size()
        if ys_segment_sizes is None:
            ys_segment_sizes = torch.full(
                (B, 1), L_y, dtype=torch.long, device=ys.device
            )

        """
        - ys batch size < memory batch size -> not allowed
        - ys batch size == memory batch size -> convert ys to seq of seqs
        - ys batch size > memory batch size
            - memory batch size == 1 -> flatten ys and convert to seq of seqs
            - memory batch size > 1 -> not allowed
        """
        memory_B = memory[0].cu_seqlens.numel() - 1
        assert B >= memory_B, f"{B=} {memory_B=}"
        assert B <= memory_B or memory_B == 1, f"{B=} {memory_B=}"

        self_attns: dict[int, torch.Tensor] = {}
        cross_attns: dict[int, torch.Tensor] = {}
        hidden_representations: dict[int, PackedTensorSequences] = {}
        ys_h, ys_seqs_cu_seqlens, ys_indices, attn_mask = self._embed_and_pack(
            xs=ys,
            xs_seqids=ys_seqids,
            xs_plddts=ys_plddts,
            xs_s3dis=ys_s3dis,
            xs_atomxs=ys_atomxs,
            xs_atombs=ys_atombs,
            segment_sizes=ys_segment_sizes,
            bias_need_pad_input=ys.device.type == "cpu"
            or len(self_attn_layers) > 0
            or len(cross_attn_layers) > 0,
        )
        if ys_ref_values is not None:
            if ys_refs is not None:
                mask = ys_refs != -100
                ys_h.x[mask] /= 2
                ys_h.x[mask] += ys_ref_values / 2
            else:
                ys_h.x /= 2
                ys_h.x += ys_ref_values / 2
        if ys.device.type == "cpu":
            ys_h.make_to_paddedable()
        if B > 1 and memory_B == 1:
            ys_seqs_cu_seqlens = ys_h.cu_seqlens[[0, -1]]
        if len(self_attn_layers) > 0:
            ys_h.make_to_paddedable()
        if len(cross_attn_layers) > 0:
            ys_h.make_to_paddedable()
            for this_memory in memory:
                this_memory.make_to_paddedable()
            del this_memory
        if 0 in repr_layers:
            hidden_representations[0] = ys_h
        layer: RotaryTransformerDecoderLayer
        for i, layer in enumerate(decoder.layers):
            key, value = memory[i * 2], memory[i * 2 + 1]
            last_layer = i == len(decoder.layers) - 1
            return_self_attention = i in self_attn_layers or (
                -1 in self_attn_layers and last_layer
            )
            return_cross_attention = i in cross_attn_layers or (
                -1 in cross_attn_layers and last_layer
            )
            if self.atomx_layer_idxs is not None and i not in self.atomx_layer_idxs:
                layer_attn_mask = None
            elif inference_params is not None or POET_2_ATOM3_BIAS_FORCE_FLASH_ATTN:
                # inference_params implies no plddt or atomx, as asserted above
                # when no plddt or atomx, the bias weight is always the mask bias
                layer_attn_mask = (
                    layer.bias_weights[:, -2 if self.use_confidence_bucket else -1]
                    .float()
                    .contiguous()
                    .view(1, -1, 1, 1)
                    .detach()
                )
            else:
                layer_attn_mask = replace(attn_mask, weights=layer.bias_weights)
            result = layer.forward_packed_from_key_value(
                ys_h,
                key,
                value,
                seqs_cu_seqlens=ys_seqs_cu_seqlens,
                tgt_mask=layer_attn_mask,
                return_self_attention=return_self_attention,
                return_cross_attention=return_cross_attention,
                inference_params=(
                    replace(
                        inference_params,
                        key_value_memory_dict={
                            i: inference_params.key_value_memory_dict[i]
                        },
                    )
                    if inference_params is not None
                    else None
                ),
            )
            if return_self_attention or return_cross_attention:
                ys_h, (self_attn, cross_attn) = result
            else:
                ys_h = result
            if return_self_attention:
                self_attns[i] = self_attn
            if return_cross_attention:
                if B > 1 and memory_B == 1:
                    # NOTE: output is not contiguous
                    cross_attn = cross_attn.view(
                        cross_attn.size(1), B, -1, cross_attn.size(3)
                    ).transpose(0, 1)
                cross_attns[i] = cross_attn
            if i + 1 in repr_layers:
                hidden_representations[i + 1] = ys_h
        if -1 in self_attn_layers:
            self_attns[-1] = self_attn
            if i not in self_attn_layers:
                del self_attns[i]
        if -1 in cross_attn_layers:
            cross_attns[-1] = cross_attn
            if i not in cross_attn_layers:
                del cross_attns[i]
        ys_h.x = decoder_norm.forward(ys_h.x)
        if i + 2 in repr_layers:
            hidden_representations[i + 2] = ys_h
        if -1 in repr_layers:
            hidden_representations[-1] = ys_h

        if return_logits:
            logits = decoder_head.forward(ys_h.x)
            logits, _ = pad_input(logits, ys_indices, B, L_y)  # (B,L,num_tokens)
        else:
            logits = None
        return PoET2Output(
            self_attns,
            cross_attns,
            hidden_representations,
            logits,
        )

    def _logits_from_xs_h(
        self,
        decoder: TransformerDecoder,
        decoder_norm: nn.Module,
        decoder_head: Optional[nn.Module],
        ys: torch.Tensor,
        ys_seqids: torch.Tensor | None,
        ys_plddts: torch.Tensor | None,
        ys_s3dis: torch.Tensor | None,
        ys_atomxs: torch.Tensor | None,
        ys_atombs: torch.Tensor | None,
        ys_segment_sizes: Optional[torch.Tensor],
        xs_h: Union[PackedTensorSequences, list[PackedTensorSequences]],
        ys_refs: torch.Tensor,
        ys_ref_values: torch.Tensor,
    ) -> torch.Tensor:
        B, L_y = ys.size()
        if ys_segment_sizes is None:  # assume we're getting logits for one sequence
            ys_segment_sizes = torch.full(
                (B, 1), L_y, dtype=torch.long, device=ys.device
            )
        if isinstance(xs_h, PackedTensorSequences):
            xs_h = self.get_decoder_memory(decoder=decoder, xs_h=xs_h)
        return self.outputs_from_memory(  # type: ignore
            decoder=decoder,
            decoder_norm=decoder_norm,
            decoder_head=decoder_head,
            ys=ys,
            ys_seqids=ys_seqids,
            ys_plddts=ys_plddts,
            ys_s3dis=ys_s3dis,
            ys_atomxs=ys_atomxs,
            ys_atombs=ys_atombs,
            memory=xs_h,
            ys_segment_sizes=ys_segment_sizes,
            ys_refs=ys_refs,
            ys_ref_values=ys_ref_values,
        ).logits

    def forward(
        self,
        xs: torch.Tensor,
        xs_plddts: torch.Tensor | None,
        xs_s3dis: torch.Tensor | None,
        xs_atomxs: torch.Tensor | None,
        xs_atombs: torch.Tensor | None,
        xs_segment_sizes: torch.Tensor,
        mlm_ys: torch.Tensor,
        mlm_ys_seqids: torch.Tensor | None,
        mlm_ys_plddts: torch.Tensor | None,
        mlm_ys_s3dis: torch.Tensor | None,
        mlm_ys_atomxs: torch.Tensor | None,
        mlm_ys_atombs: torch.Tensor | None,
        mlm_ys_refs: torch.Tensor,
        mlm_ys_segment_sizes: torch.Tensor,
        clm_ys: torch.Tensor,
        clm_ys_seqids: torch.Tensor | None,
        clm_ys_plddts: torch.Tensor | None,
        clm_ys_s3dis: torch.Tensor | None,
        clm_ys_atomxs: torch.Tensor | None,
        clm_ys_atombs: torch.Tensor | None,
        clm_ys_refs: torch.Tensor,
        clm_ys_segment_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mlm_xs_segment_sizes, clm_xs_segment_sizes = xs_segment_sizes.chunk(2)
        mlm_ys_refs = unpad_ys_refs(
            ys_refs=mlm_ys_refs,
            segment_sizes=mlm_ys_segment_sizes,
            src_segment_sizes=mlm_xs_segment_sizes,
        )
        clm_ys_refs = unpad_ys_refs(
            ys_refs=clm_ys_refs,
            segment_sizes=clm_ys_segment_sizes,
            src_segment_sizes=clm_xs_segment_sizes,
        )
        xs_h = self.encoder_outputs(
            xs=xs,
            xs_plddts=xs_plddts,
            xs_s3dis=xs_s3dis,
            xs_atomxs=xs_atomxs,
            xs_atombs=xs_atombs,
            segment_sizes=xs_segment_sizes,
        ).reprs[-1]

        # mlm and clm decoder weights are tied, so we can use either decoder
        mlm_y_ref_values = xs_h.x[: mlm_xs_segment_sizes.sum()][
            mlm_ys_refs[mlm_ys_refs != -100]
        ]
        clm_y_ref_values = xs_h.x[mlm_xs_segment_sizes.sum() :][
            clm_ys_refs[clm_ys_refs != -100]
        ]
        mlm_xs_h, clm_xs_h = self._chunk_xs_h(
            self.get_decoder_memory(decoder=self.mlm_decoder, xs_h=xs_h)
        )
        mlm_ys_logits = self._logits_from_xs_h(
            decoder=self.mlm_decoder,
            decoder_norm=self.norm,
            decoder_head=self.head if self.tied_heads else self.mlm_head,
            ys=mlm_ys,
            ys_seqids=mlm_ys_seqids,
            ys_plddts=mlm_ys_plddts,
            ys_s3dis=mlm_ys_s3dis,
            ys_atomxs=mlm_ys_atomxs,
            ys_atombs=mlm_ys_atombs,
            ys_segment_sizes=mlm_ys_segment_sizes,
            xs_h=mlm_xs_h,
            ys_refs=mlm_ys_refs,
            ys_ref_values=mlm_y_ref_values,
        )
        clm_ys_logits = self._logits_from_xs_h(
            decoder=self.clm_decoder,
            decoder_norm=self.norm,
            decoder_head=self.head if self.tied_heads else self.clm_head,
            ys=clm_ys,
            ys_seqids=clm_ys_seqids,
            ys_plddts=clm_ys_plddts,
            ys_s3dis=clm_ys_s3dis,
            ys_atomxs=clm_ys_atomxs,
            ys_atombs=clm_ys_atombs,
            ys_segment_sizes=clm_ys_segment_sizes,
            xs_h=clm_xs_h,
            ys_refs=clm_ys_refs,
            ys_ref_values=clm_y_ref_values,
        )

        # do this at the end, since the latter bit may require host-device sync
        head = self.head if self.tied_heads else self.mlm_head
        xs_logits, _ = pad_input(
            # no norm here because encoder output is already normed
            hidden_states=head.forward(xs_h.x),
            indices=xs_h.compute_indices(xs_h.cu_seqlens.diff()),
            batch=xs_h.cu_seqlens.numel() - 1,
            seqlen=xs_h.max_s,
        )
        return xs_logits, mlm_ys_logits, clm_ys_logits

    def _chunk_xs_h(
        self, xs_h: Union[PackedTensorSequences, list[PackedTensorSequences]]
    ) -> Union[
        tuple[PackedTensorSequences, PackedTensorSequences],
        tuple[list[PackedTensorSequences], list[PackedTensorSequences]],
    ]:
        # xs_hs is list of memories, for which we can apply chunking to in a loop
        # xs_h is first memory, from which we can extract metadata
        if isinstance(xs_h, PackedTensorSequences):
            xs_hs, return_list = [xs_h], False
        else:
            xs_hs, return_list = xs_h, True
            xs_h = xs_hs[0]
        mlm_xs_seqlens, xs_seqlens = xs_h.cu_seqlens.diff().chunk(2)
        mlm_xs_n = mlm_xs_seqlens.sum().item()
        xs_n = xs_seqlens.sum().item()
        assert (
            len(set(id(xs_h.positions) for xs_h in xs_hs)) == 1
        ), "positions should be shared"
        mlm_positions = xs_h.positions[:mlm_xs_n]
        clm_positions = xs_h.positions[-xs_n:]
        mlm_cu_seqlens = F.pad(mlm_xs_seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0))
        clm_cu_seqlens = F.pad(xs_seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0))
        mlm_max_s = mlm_xs_seqlens.max().item()
        clm_max_s = xs_seqlens.max().item()
        del xs_h
        mlm_xs_hs, clm_xs_hs = [], []
        _mlm_indices, _mlm_batch_size = None, None
        _clm_indices, _clm_batch_size = None, None
        for xs_h in xs_hs:
            mlm_xs_h = PackedTensorSequences(
                packed_tensor=xs_h.x[:mlm_xs_n],
                positions=mlm_positions,
                indices=None,
                cu_seqlens=mlm_cu_seqlens,
                max_s=mlm_max_s,
                batch_size=None,
                to_paddedable=False,
            )
            if xs_h.to_paddedable:
                if _mlm_batch_size is None:
                    mlm_xs_h.make_to_paddedable()
                    _mlm_indices = mlm_xs_h.indices
                    _mlm_batch_size = mlm_xs_h.batch_size
                else:
                    mlm_xs_h.to_paddedable = True
                    mlm_xs_h.indices = _mlm_indices
                    mlm_xs_h.batch_size = _mlm_batch_size
            mlm_xs_hs.append(mlm_xs_h)
            clm_xs_h = PackedTensorSequences(
                packed_tensor=xs_h.x[-xs_n:],
                positions=clm_positions,
                indices=None,
                cu_seqlens=clm_cu_seqlens,
                max_s=clm_max_s,
                batch_size=None,
                to_paddedable=False,
            )
            if xs_h.to_paddedable:
                if _clm_batch_size is None:
                    clm_xs_h.make_to_paddedable()
                    _clm_indices = clm_xs_h.indices
                    _clm_batch_size = clm_xs_h.batch_size
                else:
                    clm_xs_h.to_paddedable = True
                    clm_xs_h.indices = _clm_indices
                    clm_xs_h.batch_size = _clm_batch_size
            clm_xs_hs.append(clm_xs_h)
        if not return_list:
            return mlm_xs_hs[0], clm_xs_hs[0]
        else:
            return mlm_xs_hs, clm_xs_hs

    def outputs_with_ref_values(
        self,
        xs: torch.Tensor,
        segment_sizes: torch.Tensor,
        xs_plddts: torch.Tensor | None = None,
        xs_s3dis: torch.Tensor | None = None,
        xs_atomxs: torch.Tensor | None = None,
        xs_atombs: torch.Tensor | None = None,
    ) -> tuple[list[PackedTensorSequences], torch.Tensor]:
        xs_h = self.encoder_outputs(
            xs=xs,
            xs_plddts=xs_plddts,
            xs_s3dis=xs_s3dis,
            xs_atomxs=xs_atomxs,
            xs_atombs=xs_atombs,
            segment_sizes=segment_sizes,
        ).reprs[-1]
        B = xs_h.cu_seqlens.numel() - 1
        idxs = torch.arange(0, segment_sizes[:, 0].min().item(), device=xs_h.x.device)
        idxs = idxs.unsqueeze(0).expand(B, -1).clone()
        idxs += F.pad(segment_sizes[:-1].sum(dim=1).cumsum(dim=0), (1, 0)).unsqueeze(1)
        # clone to allow xs_h.x to be garbage collected
        ys_ref_values = xs_h.x[idxs.flatten()].clone()
        memory = self.get_decoder_memory(decoder=self.clm_decoder, xs_h=xs_h)
        return memory, ys_ref_values

    def get_inference_cache(
        self, batch_size: int, max_length: int
    ) -> dict[int, torch.Tensor]:
        # TODO: could cache inference cache... but user needs to be aware of memory
        #       usage implications, so maybe it's not worth it
        param = next(self.parameters())
        dtype, device = param.dtype, param.device
        layer: RotaryTransformerDecoderLayer = self.clm_decoder.layers[0]
        hidden_dim, num_heads = layer.dim, layer.num_heads
        head_dim = hidden_dim // num_heads
        return {
            layer_idx: torch.empty(
                (batch_size, max_length, 2, num_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            for layer_idx in range(len(self.clm_decoder.layers))
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        seqid: torch.Tensor | None,
        memory: list[PackedTensorSequences],
        self_prompt: torch.Tensor | None,
        plddt: torch.Tensor | None,
        s3di: torch.Tensor | None,
        atomx: torch.Tensor | None,
        atomb: torch.Tensor | None,
        ys_ref_values: Optional[torch.Tensor],
        max_length: int,  # TODO: account for start/stop?
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
        force_consistent: bool = False,
        fast: Optional[bool] = None,
        enable_timing: bool = False,
        show_pbar: bool = False,
        ensemble_weights: torch.Tensor | None = None,
        ensemble_method: Literal["arithmetic", "geometric"] = "arithmetic",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (
            can_fast_mode := input_ids.device.type == "cuda"
            and plddt is None
            and atomx is None
        ):
            assert not fast
        return decode(
            input_ids=input_ids,
            model=PoET2LMHeadModel(
                model=self,
                plddt=plddt,
                s3di=s3di,
                atomx=atomx,
                atomb=atomb,
                seqid=seqid,
            ),
            memory=memory,
            self_prompt=self_prompt,
            ys_ref_values=ys_ref_values,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            force_consistent=force_consistent,
            vocab_size=None,
            tensor_parallel=1,
            cg=False,
            enable_timing=enable_timing,
            fast=fast if fast is not None else can_fast_mode,
            show_pbar=show_pbar,
            ensemble_weights=ensemble_weights,
            ensemble_method=ensemble_method,
        )


class PoET2LMHeadModel:
    def __init__(
        self,
        model: PoET2,
        plddt: torch.Tensor | None,
        s3di: torch.Tensor | None,
        atomx: torch.Tensor | None,
        atomb: torch.Tensor | None,
        seqid: torch.Tensor | None,
    ) -> None:
        """seqid: (2,)"""
        if plddt is not None:
            assert plddt.ndim == 1
        if s3di is not None:
            assert s3di.ndim == 1
        if atomx is not None:
            assert atomx.ndim == 3
        if atomb is not None:
            assert atomb.ndim == 2
        self.model = model
        self.plddt, self.s3di, self.atomx, self.atomb = plddt, s3di, atomx, atomb
        self.seqid = seqid

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: list[PackedTensorSequences],
        ys_refs: torch.Tensor | None = None,
        ys_ref_values: torch.Tensor | None = None,
        inference_params: Optional[InferenceParams] = None,
    ) -> torch.Tensor:
        """
        input_ids: (memory_B, B, L)
        ys_refs: (B, L)
        ys_ref_values: (memory_B * L,)
        """
        memory_B, B, L = input_ids.size()
        return self.model.outputs_from_memory(
            decoder=self.model.clm_decoder,
            decoder_norm=self.model.norm,
            decoder_head=self.model.head,
            ys=input_ids.flatten(start_dim=1, end_dim=2),
            memory=memory,
            ys_seqids=(
                self.seqid
                if self.seqid is None
                else self.seqid.expand(memory_B, B * L, 2)
            ),
            ys_plddts=(
                self.plddt[ys_refs.view(-1)].expand(memory_B, B * L)
                if ys_refs is not None and self.plddt is not None
                else None
            ),
            ys_s3dis=(
                self.s3di[ys_refs.view(-1)].expand(memory_B, B * L)
                if ys_refs is not None
                and self.model.s3di_embed is not None
                and self.s3di is not None
                else None
            ),
            ys_atomxs=(
                self.atomx[ys_refs.view(-1)].expand(memory_B, B * L, 3, 3)
                if ys_refs is not None and self.atomx is not None
                else None
            ),
            ys_atombs=(
                self.atomb[ys_refs.view(-1)].expand(memory_B, B * L, N_ATOMB)
                if ys_refs is not None
                and self.model.atomb_embed is not None
                and self.atomb is not None
                else None
            ),
            ys_segment_sizes=torch.full(
                (memory_B, B), L, dtype=torch.long, device=input_ids.device
            ),
            ys_refs=None,
            ys_ref_values=ys_ref_values,
            inference_params=inference_params,
        ).logits.chunk(2, dim=2)[0]


@torch.inference_mode()
def decode(
    input_ids: torch.Tensor,
    model: PoET2LMHeadModel,
    memory: list[PackedTensorSequences],
    self_prompt: torch.Tensor | None,
    ys_ref_values: torch.Tensor | None,
    max_length: int,
    top_k: int = 0,
    top_p: float = 0.0,
    temperature: float = 1.0,
    force_consistent: bool = False,
    vocab_size: int | None = None,
    tensor_parallel: int = 1,
    cg: bool = False,
    fast: bool = True,
    enable_timing: bool = False,
    show_pbar: bool = False,
    alphabet: Alphabet = Alphabet(),
    ensemble_weights: torch.Tensor | None = None,
    ensemble_method: Literal["arithmetic", "geometric"] = "arithmetic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is
    applied first, then top-p.

    This function is adapted from flash_attn.utils.generation
    https://github.com/Dao-AILab/flash-attention/blob/23e8fa5a263d1c7122bc46a86ef32030ee7130f9/flash_attn/utils/generation.py#L94

    Args:
        input_ids: (batch, seq_len)
        max_length: max length of seqs to generate, including tokens in input_ids
    Returns:
        logits: (batch, memory batch, length, vocab_size)
        sequences: (batch, length)
        logp: (batch, memory batch)
    """
    assert input_ids.size(1) == 1, "ys_refs code path does not support larger input"
    assert not cg
    if enable_timing:
        assert input_ids.device.type == "cuda"
    B, memory_B = input_ids.size(0), memory[0].cu_seqlens.numel() - 1
    if self_prompt is not None:
        self_prompt = self_prompt.long()
        assert self_prompt.ndim == 1
        assert ys_ref_values is not None
        assert ys_ref_values.size(0) == memory_B * len(self_prompt)
    if ys_ref_values is not None:
        assert ys_ref_values.size(0) % memory_B == 0
    L = max_length - input_ids.size(1)  # max number of tokens to sample
    assert L > 0
    if model.seqid is not None:
        assert self_prompt is None
    if ensemble_weights is not None:
        assert (ensemble_weights.ndim == 1) and (ensemble_weights.numel() == memory_B)
        ensemble_weights = ensemble_weights.float()
        if ensemble_method == "arithmetic":
            ensemble_weights = ensemble_weights / ensemble_weights.sum()
        log_ensemble_weights = ensemble_weights.float().log()
    else:
        ensemble_weights = torch.full(
            (memory_B,), 1 / memory_B, dtype=torch.float32, device=input_ids.device
        )
        log_ensemble_weights = ensemble_weights.log()

    base_invalid_token_ids = list(range(20, 25)) + [
        alphabet.start_token,
        alphabet.cls_token,
    ]
    no_gap_token_ids = base_invalid_token_ids + [alphabet.gap_token]
    no_stop_token_ids = base_invalid_token_ids + [alphabet.stop_token]
    no_gap_or_stop_token_ids = base_invalid_token_ids + [
        alphabet.gap_token,
        alphabet.stop_token,
    ]
    no_aa_token_ids = base_invalid_token_ids + list(range(20))
    no_aa_or_gap_token_ids = no_aa_token_ids + [alphabet.gap_token]
    no_aa_or_stop_token_ids = no_aa_token_ids + [alphabet.stop_token]

    inference_params = InferenceParams(
        max_seqlen=max_length,
        max_batch_size=B,
        key_value_memory_dict=model.model.get_inference_cache(
            batch_size=B * memory_B, max_length=max_length
        ),
    )

    if ys_ref_values is not None:
        ys_refs_offsets = torch.arange(
            0, memory_B, dtype=torch.long, device=input_ids.device
        ) * (ys_ref_values.size(0) // memory_B)
    else:
        ys_refs_offsets = None

    def get_logits(
        input_ids: torch.Tensor,
        ys_refs: torch.Tensor | None,
        ys_ref_values: torch.Tensor | None,
        inference_params: InferenceParams | None,
    ) -> torch.Tensor:
        """logits: (B, memory_B, L, V)"""
        ys_ref_values = (
            ys_ref_values[
                (
                    ys_refs.unsqueeze(0).expand(memory_B, -1, -1)
                    + ys_refs_offsets.unsqueeze(1).unsqueeze(2)
                ).flatten()
            ]
            if ys_refs is not None
            else None
        )
        logits = (
            model.forward(
                input_ids.unsqueeze(0).expand(memory_B, -1, -1),
                memory=memory,
                ys_refs=ys_refs,
                ys_ref_values=ys_ref_values,
                inference_params=inference_params,
            )
            .float()
            .log_softmax(dim=-1)
        )
        logits = logits.view(memory_B, B, input_ids.size(1), -1).transpose(0, 1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def score_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -F.cross_entropy(
            logits.permute(0, 3, 1, 2),  # (B, V, memory_B, L)
            target.unsqueeze(1).expand(-1, memory_B, -1),  # (B, memory_B, L)
            ignore_index=alphabet.mask_token,
            reduction="none",
        ).sum(dim=2)

    def _sample_tokens_no_prompt(
        logits: torch.Tensor, inference_params: InferenceParams, ys_refs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = inference_params.seqlen_offset
        if force_consistent and idx >= max_length - 1:
            token = torch.full(
                (logits.size(0),),
                alphabet.stop_token,
                dtype=torch.long,
                device=logits.device,
            )
        else:
            # TODO: should force consistent affect whether we sample gaps?
            logits[:, no_gap_token_ids] += -torch.inf
            token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        # if model.seqid is not None, always reference first token in reference seq
        # otherwise, this can be anything
        ys_refs = torch.zeros((B,), dtype=torch.long, device=input_ids.device)
        return token, token, ys_refs

    def _sample_tokens_prompt(
        logits: torch.Tensor, inference_params: InferenceParams, ys_refs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ys_refs = ys_refs[:, -1].clone()
        current_input_is_last = ys_refs == len(self_prompt) - 1
        current_input_id = self_prompt[ys_refs.clip(max=len(self_prompt) - 1)]
        current_input_id[ys_refs > len(self_prompt) - 1] = -100
        next_input_is_last = ys_refs == len(self_prompt) - 2
        next_input_id = self_prompt[(ys_refs + 1).clip(max=len(self_prompt) - 1)]
        next_input_id[ys_refs + 1 > len(self_prompt) - 1] = -100
        if not force_consistent:
            logits[:, base_invalid_token_ids] += -torch.inf
            token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
            is_skip_ins = (token == alphabet.gap_token) & (
                next_input_id == alphabet.gap_token
            )
            ys_refs[is_skip_ins] += 2
            is_extend_ins = (token < 20) & (current_input_id == alphabet.gap_token)
            ys_refs[~is_skip_ins & ~is_extend_ins] += 1
            next_input_token = token.where(
                (token != alphabet.gap_token) | (ys_refs > len(self_prompt) - 1),
                self_prompt[ys_refs.clip_(max=len(self_prompt) - 1)],
            )
            # ys_refs.clip_(max=len(self_prompt) - 1)
            # next_input_token = token
            return token, next_input_token, ys_refs
        idx = inference_params.seqlen_offset
        if idx >= max_length - 1:
            token = torch.full(
                (logits.size(0),),
                (
                    alphabet.gap_token
                    if self_prompt[-1] == alphabet.stop_token
                    else alphabet.stop_token
                ),
                dtype=torch.long,
                device=logits.device,
            )
            ys_refs = ys_refs + 1
            next_input_token = token.where(
                (token != alphabet.gap_token) | (ys_refs > len(self_prompt) - 1),
                self_prompt[ys_refs.clip_(max=len(self_prompt) - 1)],
            )
            # ys_refs.clip_(max=len(self_prompt) - 1)
            # next_input_token = token
            return (
                token,
                next_input_token,
                ys_refs,
            )

        logits[
            torch.where(next_input_is_last & (next_input_id == alphabet.mask_token))[
                0
            ].unsqueeze(1),
            no_aa_or_gap_token_ids,
        ] += -torch.inf
        logits[
            torch.where(next_input_is_last & (next_input_id == alphabet.stop_token))[
                0
            ].unsqueeze(1),
            no_aa_or_stop_token_ids,
        ] += -torch.inf

        n_remaining_nongap = torch.hstack(
            [(self_prompt[i:] != alphabet.gap_token).sum() for i in ys_refs]
        )
        if self_prompt[-1] == alphabet.gap_token:
            n_remaining_nongap += 1
        can_indel = max_length - idx - n_remaining_nongap > 0

        logits[
            torch.where(
                (~can_indel)
                & next_input_is_last
                & (next_input_id == alphabet.gap_token)
            )[0].unsqueeze(1),
            no_aa_or_gap_token_ids,
        ] += -torch.inf
        logits[
            torch.where(
                (~can_indel)
                & (~next_input_is_last)
                & (next_input_id == alphabet.gap_token)
            )[0].unsqueeze(1),
            no_aa_or_stop_token_ids,
        ] += -torch.inf
        logits[
            torch.where(
                (~can_indel)
                & (~next_input_is_last)
                & (next_input_id < alphabet.mask_token)
            )[0].unsqueeze(1),
            no_aa_or_stop_token_ids,
        ] += -torch.inf
        logits[
            torch.where(
                (~can_indel)
                & (~next_input_is_last)
                & (next_input_id == alphabet.mask_token)
            )[0].unsqueeze(1),
            no_gap_or_stop_token_ids,
        ] += -torch.inf

        logits[
            torch.where(
                current_input_is_last
                & (current_input_id == alphabet.gap_token)
                & (~can_indel)
            )[0].unsqueeze(1),
            no_aa_or_gap_token_ids,
        ] += -torch.inf
        logits[
            torch.where(
                current_input_is_last
                & (current_input_id == alphabet.gap_token)
                & can_indel
            )[0].unsqueeze(1),
            no_gap_token_ids,
        ] += -torch.inf

        c1 = next_input_id == alphabet.mask_token
        c2 = current_input_id == alphabet.gap_token
        c4 = next_input_id == alphabet.gap_token
        c3 = (~c1) & (~c2) & (~c4)
        logits[
            torch.where((~current_input_is_last) & (~next_input_is_last) & c1)[
                0
            ].unsqueeze(1),
            no_gap_token_ids,
        ] += -torch.inf
        logits[
            torch.where(
                can_indel & (~current_input_is_last) & (~next_input_is_last) & c2
            )[0].unsqueeze(1),
            no_stop_token_ids,
        ] += -torch.inf
        logits[
            torch.where(
                (~can_indel) & (~current_input_is_last) & (~next_input_is_last) & c2
            )[0].unsqueeze(1),
            no_aa_or_stop_token_ids,
        ] += -torch.inf
        logits[
            torch.where((~current_input_is_last) & (~next_input_is_last) & c3)[
                0
            ].unsqueeze(1),
            no_aa_or_stop_token_ids,
        ] += -torch.inf
        logits[
            torch.where(
                (~can_indel) & (~current_input_is_last) & (~next_input_is_last) & c4
            )[0].unsqueeze(1),
            no_aa_or_stop_token_ids,
        ] += -torch.inf
        logits[
            torch.where(
                can_indel & (~current_input_is_last) & (~next_input_is_last) & c4
            )[0].unsqueeze(1),
            no_stop_token_ids,
        ] += -torch.inf

        assert not torch.isinf(logits).all(dim=1).any()
        token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        is_skip_ins = (token == alphabet.gap_token) & (
            next_input_id == alphabet.gap_token
        )
        ys_refs[is_skip_ins] += 2
        is_extend_ins = (token < 20) & (current_input_id == alphabet.gap_token)
        ys_refs[~is_skip_ins & ~is_extend_ins] += 1
        next_input_token = token.where(
            (token != alphabet.gap_token) | (ys_refs > len(self_prompt) - 1),
            self_prompt[ys_refs.clip_(max=len(self_prompt) - 1)],
        )
        # ys_refs.clip_(max=len(self_prompt) - 1)
        # next_input_token = token
        return token, next_input_token, ys_refs

    def sample_tokens(
        logits: torch.Tensor, inference_params: InferenceParams, ys_refs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: input logits can be mutated
        # return rearrange(token, "b -> b 1")
        if self_prompt is None:
            fn = _sample_tokens_no_prompt
        else:
            fn = _sample_tokens_prompt
        token, next_input_token, ys_refs = fn(
            logits=logits, inference_params=inference_params, ys_refs=ys_refs
        )
        return token.unsqueeze(1), next_input_token.unsqueeze(1), ys_refs.unsqueeze(1)

    def should_stop(
        current_token: torch.Tensor,
        inference_params: InferenceParams,
        done_idxs: set[int] = set(),
    ) -> bool:
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        this_done_idxs = torch.where((current_token == alphabet.stop_token).any(dim=1))[
            0
        ].tolist()
        done_idxs |= set(this_done_idxs)
        if len(done_idxs) == B:
            return True
        return False

    if enable_timing:
        start = torch.cuda.Event(enable_timing=enable_timing)
        end = torch.cuda.Event(enable_timing=enable_timing)
        if tensor_parallel > 1:
            torch.distributed.barrier()
        start.record()
    prompt_logps = torch.zeros((B, memory_B), device=input_ids.device)
    logits: list[torch.Tensor] = []
    sequences: list[torch.Tensor] = [input_ids]
    sequences_input: list[torch.Tensor] = [input_ids]
    ys_refs: list[torch.Tensor] = [
        torch.zeros((B, 1), dtype=torch.long, device=input_ids.device)
    ]
    use_ys_refs = self_prompt is not None or model.seqid is not None
    if show_pbar:
        pbar = tqdm(total=L)
    while not should_stop(sequences_input[-1], inference_params):
        if not fast:
            # do at start of loop to avoid clearing value when exiting loop
            inference_params.seqlen_offset = 0
            prompt_logps.zero_()

        logits.append(
            get_logits(
                sequences_input[-1],
                ys_refs=ys_refs[-1] if use_ys_refs else None,
                ys_ref_values=ys_ref_values if use_ys_refs else None,
                inference_params=inference_params if fast else None,
            )
        )
        if sequences[-1].size(1) > 1:
            prompt_logps += score_from_logits(
                logits[-1][:, :, :-1], sequences[-1][:, 1:]
            )
        if ensemble_method == "arithmetic":
            ensemble_logits_numerator = (
                prompt_logps.unsqueeze(2)
                + logits[-1][:, :, -1]
                + log_ensemble_weights.view(-1, 1)
            ).logsumexp(dim=1)
            ensemble_logits_denominator = (
                (prompt_logps + log_ensemble_weights).logsumexp(dim=1).unsqueeze(1)
            )
        elif ensemble_method == "geometric":
            ensemble_logits_numerator = (
                (prompt_logps.unsqueeze(2) + logits[-1][:, :, -1])
                * ensemble_weights.view(-1, 1)
            ).sum(dim=1)
            ensemble_logits_denominator = (
                (prompt_logps * ensemble_weights).sum(dim=1).unsqueeze(1)
            )
        else:
            raise ValueError(ensemble_method)
        ensemble_logits = ensemble_logits_numerator - ensemble_logits_denominator
        inference_params.seqlen_offset += sequences[-1].size(1)
        next_tokens, next_input_tokens, next_ys_refs = sample_tokens(
            ensemble_logits, inference_params, ys_refs[-1]
        )
        sequences.append(next_tokens)
        sequences_input.append(next_input_tokens)
        ys_refs.append(next_ys_refs)
        prompt_logps += score_from_logits(logits[-1][:, :, [-1]], sequences[-1])

        if not fast:
            logits = [logits[-1]]
            sequences = [torch.cat(sequences, dim=1)]
            sequences_input = [torch.cat(sequences_input, dim=1)]
            ys_refs = [torch.cat(ys_refs, dim=1)]
        if show_pbar:
            pbar.update(1)
    if show_pbar:
        pbar.close()
    if enable_timing:
        end.record()
        if tensor_parallel > 1:
            torch.distributed.barrier()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    return (
        torch.cat(logits, dim=2),
        torch.cat(sequences, dim=1),
        torch.cat(ys_refs, dim=1),
        prompt_logps,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.half
    model = PoET2(
        n_vocab=21,
        n_out=42,
        hidden_dim=64,
        n_layers=2,
        nhead=2,
        in_seqid=False,
        in_plddt=False,
        in_s3di=None,
        atomx_layer_idxs=None,
        n_distance_buckets=126,
        n_angle_buckets=None,
        use_confidence_bucket=True,
        in_atomb=False,
        version=1,
    )
    model = model.to(device=device, dtype=dtype)
    # fmt: off
    xs = torch.tensor([
        [0, 0,1, 0,1,2, 21,21],
        [0,1, 0,1,2,3,4,5],
    ], device=device)
    xs_segment_sizes = torch.tensor([[1, 2, 3], [2, 6, 0]], device=device)
    mlm_ys = torch.tensor([
        [0, 0,1],
    ], device=device)
    mlm_ys_segment_sizes = torch.tensor([[1, 2]], device=device)
    mlm_ys_refs = torch.tensor([[-100, 3,5]], device=device)
    clm_ys = torch.tensor([
        [0,1, 0,1,2,3],
    ], device=device)
    clm_ys_segment_sizes = torch.tensor([[2, 4]], device=device)
    clm_ys_refs = torch.tensor([[-100,-100, 2,3,5,7]], device=device)
    # if device.type == "cuda":
    #     torch.cuda.set_sync_debug_mode(1)
    # fmt: on
    xs_logits, mlm_ys_logits, clm_ys_logits = model.forward(
        xs=xs,
        xs_plddts=None,
        xs_s3dis=None,
        xs_atomxs=None,
        xs_atombs=None,
        xs_segment_sizes=xs_segment_sizes,
        mlm_ys=mlm_ys,
        mlm_ys_seqids=None,
        mlm_ys_plddts=None,
        mlm_ys_s3dis=None,
        mlm_ys_atomxs=None,
        mlm_ys_atombs=None,
        mlm_ys_refs=mlm_ys_refs,
        mlm_ys_segment_sizes=mlm_ys_segment_sizes,
        clm_ys=clm_ys,
        clm_ys_seqids=None,
        clm_ys_plddts=None,
        clm_ys_s3dis=None,
        clm_ys_atomxs=None,
        clm_ys_atombs=None,
        clm_ys_refs=clm_ys_refs,
        clm_ys_segment_sizes=clm_ys_segment_sizes,
    )
    import pdb; pdb.set_trace()  # fmt: skip
