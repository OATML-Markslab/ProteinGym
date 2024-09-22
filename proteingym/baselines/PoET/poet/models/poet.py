import copy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from poet.alphabets import Uniprot21
from poet.models.modules.activation import gelu
from poet.models.modules.attention import MultiheadAttention
from poet.models.modules.embedding import RotaryEmbedding
from poet.models.modules.packed_sequence import (
    PackedTensorSequences,
    get_mask,
    pad_input,
    unpad_input,
)
from poet.models.modules.transformer import TransformerEncoder
from poet.models.modules.transformer_rotary import TieredRotaryTransformerEncoderLayer


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317

    Adapted from: https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
    """
    if top_k is not None:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


class LogitsAllocateMemoryMixin(object):
    """
    Stateless mixin providing methods for preallocating memory for logits calculations.
    """

    @classmethod
    def logits_allocate_memory(
        cls,
        memory: Optional[list[PackedTensorSequences]],
        batch_size: int,
        length: int,
    ) -> Optional[list[PackedTensorSequences]]:
        """
        Modifies the tensors in `memory` to preallocate memory needed for self.logits
        Can raise a CUDA OOM error, in which case `memory` may be in an inconsistent
        state.

        Args:
          memory:
            output of self.embed or self.logits_allocate_memory
            all sequences in each individual memory in the list must be identical
          batch_size:
            batch size that self.logits will be used with
          length:
            additional padding to add to memory
            can be negative
            the total padding should be equal to the length of the sequences
            that self.logits will be used with

        Returns:
          reference to modified input memory

        Raises:
          ValueError: for invalid combinations of current_batch_size, batch_size, and
          length
        """
        if memory is None or len(memory) == 0:
            return memory

        current_batch_size = memory[0].cu_seqlens.numel() - 1
        if length == 0 and batch_size == current_batch_size:
            return memory
        elif length == 0 and batch_size < current_batch_size:
            memory = cls._logits_allocate_memory_reduce_batch_size(memory, batch_size)
        elif length <= 0 and batch_size == 1:
            memory = cls._logits_allocate_memory_reduce_length(memory, length)
        else:
            memory = cls._logits_allocate_memory(memory, batch_size, length)
        return memory

    @staticmethod
    def can_logits_allocate_memory_heuristic(
        memory: Optional[list[PackedTensorSequences]],
        batch_size: int,
    ) -> bool:
        """
        Determine whether or not there is likely to be sufficient CPU or GPU memory
        to successfully self.logits_allocate_memory(memory, batch_size, length=0)

        Args:
          memory:
            memory to allocate RAM for; all of memory should be on the same device
          batch_size:
            batch size that memory will be preallocated for

        Returns:
          whether or not there is likely to be sufficient memory, based on a heuristic
        """
        if memory is None or len(memory) == 0:
            return True

        if memory[0].x.device == torch.device("cpu"):
            return True  # just assuming here, this may be false
        else:
            memory_usage = (
                len(memory) * memory[0].x.element_size() * memory[0].x.nelement()
            )
            new_memory_usage = batch_size * memory_usage
            # overestimate by 1.5x just in case
            additional_memory_usage = new_memory_usage * 1.5
            torch.cuda.empty_cache()
            available_memory = torch.cuda.get_device_properties(
                memory[0].x.device
            ).total_memory
            # try to keep at least 5GB vram free regardless
            sufficient_memory = (available_memory - additional_memory_usage) / (
                1024**3
            ) > 5
            return sufficient_memory

    @staticmethod
    def _logits_allocate_memory_reduce_batch_size(
        memory: list[PackedTensorSequences],
        batch_size: int,
    ) -> list[PackedTensorSequences]:
        """
        Reduces the batch size of each sequence in memory to batch_size.
        Assumes batch_size <= batch size of each sequence.
        """
        B = batch_size
        for mem in memory:
            mem.x = mem.x[: mem.max_s * B]
            mem.positions = mem.positions[: mem.max_s * B]
            mem.cu_seqlens = mem.cu_seqlens[: B + 1]
            mem.cu_seqlens_cpu = mem.cu_seqlens_cpu[: B + 1]
            mem.to_paddedable = False
        return memory

    @staticmethod
    def _logits_allocate_memory_reduce_length(
        memory: list[PackedTensorSequences],
        length: int,
    ) -> list[PackedTensorSequences]:
        """
        Reduces the length of each sequence in memory by |length|.
        Assumes length <= 0 and the batch sizes are 1.
        """
        L_x = length
        for mem in memory:
            mem.x = mem.x[: mem.max_s + L_x]
            mem.positions = mem.positions[: mem.max_s + L_x]
            mem.cu_seqlens = torch.tensor(
                [0, mem.max_s + L_x], device=mem.cu_seqlens.device
            )
            mem.cu_seqlens_cpu = torch.tensor(
                [0, mem.max_s + L_x], device=mem.cu_seqlens_cpu.device
            )
            mem.max_s = mem.max_s + L_x
            mem.to_paddedable = False
        return memory

    @staticmethod
    def _logits_allocate_memory(
        memory: list[PackedTensorSequences],
        batch_size: int,
        length: int,
    ) -> list[PackedTensorSequences]:
        B, L_x = batch_size, length
        for mem in memory:
            if L_x >= 0:
                mem.x = (
                    torch.cat(
                        [
                            mem.x[: mem.max_s],
                            torch.empty(
                                (L_x, mem.x.size(1), mem.x.size(2)),
                                dtype=mem.x.dtype,
                                device=mem.x.device,
                            ),
                        ],
                        dim=0,
                    )
                    .expand(B, -1, -1, -1)
                    .flatten(start_dim=0, end_dim=1)
                )
                mem.positions = torch.cat(
                    (
                        mem.positions[: mem.max_s].unsqueeze(0).expand(B, mem.max_s),
                        torch.arange(L_x, device=mem.positions.device)
                        .unsqueeze(0)
                        .expand(B, L_x),
                    ),
                    dim=1,
                ).flatten()
            else:
                mem.x = (
                    mem.x[: mem.max_s + L_x]
                    .expand(B, -1, -1, -1)
                    .flatten(start_dim=0, end_dim=1)
                )
                mem.positions = (
                    mem.positions[: mem.max_s + L_x]
                    .expand(B, mem.max_s + L_x)
                    .flatten()
                )
            mem.cu_seqlens = F.pad(
                (
                    torch.full(
                        (B,), mem.max_s + L_x, device=mem.cu_seqlens.device
                    ).cumsum(dim=0, dtype=torch.int32)
                ),
                (1, 0),
            )
            mem.cu_seqlens_cpu = F.pad(
                (
                    torch.full(
                        (B,), mem.max_s + L_x, device=mem.cu_seqlens_cpu.device
                    ).cumsum(dim=0, dtype=torch.int32)
                ),
                (1, 0),
            )
            mem.max_s = mem.max_s + L_x
            mem.to_paddedable = False
        return memory


def _packed_sequence_expand_and_append(
    packed_sequence: PackedTensorSequences,
    x: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> None:
    B, L = x.size(0), x.size(1)
    if positions is None:
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
    assert positions.size(0) == B
    assert positions.size(1) == L

    packed_sequence.x = torch.cat(
        [
            packed_sequence.x.unsqueeze(0).expand(B, *packed_sequence.x.size()),
            x,
        ],
        dim=1,
    ).flatten(start_dim=0, end_dim=1)
    packed_sequence.positions = torch.cat(
        [
            packed_sequence.positions.unsqueeze(0).expand(B, -1),
            positions,
        ],
        dim=1,
    ).flatten()
    packed_sequence.cu_seqlens = F.pad(
        (packed_sequence.cu_seqlens.diff() + L)
        .expand(B)
        .cumsum(dim=0, dtype=packed_sequence.cu_seqlens.dtype),
        (1, 0),
    )
    packed_sequence.cu_seqlens_cpu = F.pad(
        (packed_sequence.cu_seqlens_cpu.diff() + L)
        .expand(B)
        .cumsum(dim=0, dtype=packed_sequence.cu_seqlens_cpu.dtype),
        (1, 0),
    )
    packed_sequence.max_s = packed_sequence.max_s + L
    packed_sequence.to_paddedable = False


def _packed_sequence_append(
    packed_sequence: PackedTensorSequences,
    x: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> None:
    B, L = x.size(0), x.size(1)

    current_batch_size = packed_sequence.cu_seqlens.numel() - 1
    if current_batch_size == 1:
        return _packed_sequence_expand_and_append(packed_sequence, x, positions)
    if current_batch_size != B:
        raise ValueError(current_batch_size, B)

    if positions is None:
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
    assert positions.size(0) == B
    assert positions.size(1) == L

    new_x = torch.empty(
        (packed_sequence.x.size(0) + B * L, *packed_sequence.x.size()[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    new_cu_seqlens = F.pad(
        (packed_sequence.cu_seqlens.diff() + L).cumsum(
            dim=0, dtype=packed_sequence.cu_seqlens.dtype
        ),
        (1, 0),
    )
    new_cu_seqlens_cpu = F.pad(
        (packed_sequence.cu_seqlens_cpu.diff() + L).cumsum(
            dim=0, dtype=packed_sequence.cu_seqlens_cpu.dtype
        ),
        (1, 0),
    )
    original_idxs, new_idxs = [], []
    old_lengths = packed_sequence.cu_seqlens_cpu.diff()
    for idx in range(new_cu_seqlens_cpu.numel() - 1):
        new_start = new_cu_seqlens_cpu[idx]
        old_length = old_lengths[idx]
        new_range = torch.arange(new_start, new_start + old_length + L, device=x.device)
        original_idxs.append(new_range[:old_length])
        new_idxs.append(new_range[old_length:])
    original_idxs = torch.hstack(original_idxs)
    new_idxs = torch.hstack(new_idxs)
    new_x[original_idxs] = packed_sequence.x
    new_x[new_idxs] = x.flatten(start_dim=0, end_dim=1)
    packed_sequence.x = new_x

    new_positions = torch.empty(
        (packed_sequence.positions.size(0) + B * L,),
        device=x.device,
        dtype=packed_sequence.positions.dtype,
    )
    new_positions[original_idxs] = packed_sequence.positions
    new_positions[new_idxs] = positions.flatten()
    packed_sequence.positions = new_positions

    packed_sequence.cu_seqlens = new_cu_seqlens
    packed_sequence.cu_seqlens_cpu = new_cu_seqlens_cpu
    packed_sequence.max_s = packed_sequence.max_s + L
    packed_sequence.to_paddedable = False


def _compute_attn_memory(
    x_norm: PackedTensorSequences, attn: MultiheadAttention
) -> tuple[PackedTensorSequences, PackedTensorSequences]:
    """Compute the keys and values of x_norm for the the attention module attn."""
    x_norm_km = attn.k_proj.forward(x_norm.x)
    x_norm_vm = attn.v_proj.forward(x_norm.x)
    x_norm_km = x_norm_km.view(-1, attn.num_heads, attn.head_dim)
    x_norm_vm = x_norm_vm.view(-1, attn.num_heads, attn.head_dim)
    _, x_norm_km, _ = attn._transform_qkv(
        None,
        x_norm_km,
        None,
        query_positions=x_norm.positions,
        key_positions=x_norm.positions,
        transform_query=False,
        transform_key=True,
        transform_value=False,
    )
    x_norm_key, x_norm_value = copy.copy(x_norm), copy.copy(x_norm)
    x_norm_key.x, x_norm_value.x = x_norm_km, x_norm_vm
    return x_norm_key, x_norm_value


def _update_causal_prefix_memory(
    x_norm: PackedTensorSequences,
    x_norm_km: torch.Tensor,
    x_norm_vm: torch.Tensor,
    key_memory: PackedTensorSequences,
    value_memory: PackedTensorSequences,
    batch_size: int,
    length: int,
    preallocated_memory: bool,
) -> tuple[PackedTensorSequences, PackedTensorSequences]:
    B, L_x = batch_size, length
    if preallocated_memory:
        this_memory_batch_size = key_memory.cu_seqlens.shape[0] - 1
        if this_memory_batch_size != B:
            for _memory in [key_memory, value_memory]:
                _memory.x = _memory.x.view(
                    this_memory_batch_size,
                    -1,
                    _memory.x.size(1),
                    _memory.x.size(2),
                )[:B].view(-1, _memory.x.size(1), _memory.x.size(2))
                _memory.positions = _memory.positions.view(this_memory_batch_size, -1)[
                    :B
                ].flatten()
                _memory.cu_seqlens = _memory.cu_seqlens[: B + 1]
                _memory.cu_seqlens_cpu = _memory.cu_seqlens_cpu[: B + 1]
        key_memory.x.view(B, -1, key_memory.x.size(1), key_memory.x.size(2))[
            :, -L_x:
        ] = x_norm_km.view(B, L_x, key_memory.x.size(1), key_memory.x.size(2))
        value_memory.x.view(B, -1, value_memory.x.size(1), value_memory.x.size(2))[
            :, -L_x:
        ] = x_norm_vm.view(B, L_x, value_memory.x.size(1), value_memory.x.size(2))
    elif (
        key_memory.cu_seqlens.numel() == 2
        and key_memory.cu_seqlens.numel() - 1 < batch_size
    ):
        # batch size of memory and data to append are different
        # assume memory needs to be duplicated
        for _memory, _m in zip([key_memory, value_memory], [x_norm_km, x_norm_vm]):
            _memory.x = torch.cat(
                (
                    _memory.x.unsqueeze(0).expand(
                        B,
                        _memory.max_s,
                        _memory.x.size(1),
                        _memory.x.size(2),
                    ),
                    _m.view(B, L_x, _memory.x.size(1), _memory.x.size(2)),
                ),
                dim=1,
            ).view(-1, _memory.x.size(1), _memory.x.size(2))
            _memory.positions = torch.cat(
                (
                    _memory.positions.unsqueeze(0).expand(B, _memory.max_s),
                    x_norm.positions.view(B, L_x),
                ),
                dim=1,
            ).flatten()
            _memory.cu_seqlens = F.pad(
                (
                    torch.ones((B,), device=x_norm.x.device)
                    .fill_(L_x + _memory.cu_seqlens[1])
                    .cumsum(dim=0, dtype=torch.int32)
                ),
                (1, 0),
            )
            _memory.cu_seqlens_cpu = F.pad(
                (
                    torch.ones((B,))
                    .fill_(L_x + _memory.cu_seqlens_cpu[1])
                    .cumsum(dim=0, dtype=torch.int32)
                ),
                (1, 0),
            )
            _memory.max_s = _memory.max_s + L_x
    elif key_memory.cu_seqlens.numel() - 1 == batch_size:
        for _memory, _m in zip([key_memory, value_memory], [x_norm_km, x_norm_vm]):
            _packed_sequence_append(
                _memory,
                _m.unflatten(0, (batch_size, length)),
                x_norm.positions.unflatten(0, (batch_size, length)),
            )
    else:
        raise ValueError
    return key_memory, value_memory


def _apply_causal_prefix_attention(
    decoder: TransformerEncoder,
    x: PackedTensorSequences,
    batch_size: int,
    length: int,
    self_memory: Optional[list[PackedTensorSequences]],
    memory: Optional[list[PackedTensorSequences]],
    preallocated_memory: bool,
) -> tuple[
    PackedTensorSequences,
    Optional[list[PackedTensorSequences]],
    Optional[list[PackedTensorSequences]],
]:
    B, L_x = batch_size, length

    for layer_idx, layer in enumerate(decoder.layers):
        layer: TieredRotaryTransformerEncoderLayer

        # apply the self attention layer on the sequences independently
        x_norm = copy.copy(x)
        x_norm.x = layer.norm1.forward(x.x)
        x_norm_key, x_norm_value = _compute_attn_memory(x_norm, layer.self_attn)
        if self_memory is not None:
            key_memory, value_memory = (
                copy.copy(self_memory[2 * layer_idx]),
                copy.copy(self_memory[2 * layer_idx + 1]),
            )
            key_memory.x, value_memory.x = (
                key_memory.x.to(x.x.device),
                value_memory.x.to(x.x.device),
            )
            key_memory, value_memory = _update_causal_prefix_memory(
                x_norm=x_norm,
                x_norm_km=x_norm_key.x,
                x_norm_vm=x_norm_value.x,
                key_memory=key_memory,
                value_memory=value_memory,
                batch_size=B,
                length=L_x,
                preallocated_memory=preallocated_memory,
            )
        else:
            key_memory, value_memory = x_norm_key, x_norm_value
        try:
            layer.self_attn.self_attention = False
            x2: torch.Tensor
            x2, _ = layer.self_attn.forward_packed(
                x_norm,
                key_memory,
                value_memory,
                attn_mask=None,
                key_padding_mask=None,
                return_weights=False,
                transform_query=True,
                transform_key=False,
                transform_value=False,
            )
        finally:
            layer.self_attn.self_attention = True
        x = copy.copy(x)
        x.x = x.x + layer.dropout1.forward(x2.x)

        # apply the sequence-of-sequence attention layer on the reshaped sequences
        x_norm = copy.copy(x)
        x_norm.x = layer.norm2.forward(x.x)
        x_norm_key, x_norm_value = _compute_attn_memory(x_norm, layer.multihead_attn)
        if memory is not None:
            key_memory, value_memory = (
                copy.copy(memory[2 * layer_idx]),
                copy.copy(memory[2 * layer_idx + 1]),
            )
            key_memory.x, value_memory.x = (
                key_memory.x.to(x.x.device),
                value_memory.x.to(x.x.device),
            )
            key_memory, value_memory = _update_causal_prefix_memory(
                x_norm=x_norm,
                x_norm_km=x_norm_key.x,
                x_norm_vm=x_norm_value.x,
                key_memory=key_memory,
                value_memory=value_memory,
                batch_size=B,
                length=L_x,
                preallocated_memory=preallocated_memory,
            )
        else:
            key_memory, value_memory = x_norm_key, x_norm_value
        try:
            layer.multihead_attn.self_attention = False
            x2: torch.Tensor
            x2, _ = layer.multihead_attn.forward_packed(
                x_norm,
                key_memory,
                value_memory,
                attn_mask=None,
                key_padding_mask=None,
                return_weights=False,
                transform_query=True,
                transform_key=False,
                transform_value=False,
            )
        finally:
            layer.multihead_attn.self_attention = True
        x = copy.copy(x)
        x.x = x.x + layer.dropout2.forward(x2.x)

        x2 = layer.linear2(layer.dropout(gelu(layer.linear1(layer.norm3(x.x)))))
        x.x = x.x + layer.dropout3(x2)
    return x


def _apply_causal_prefix_attention_buffered(
    decoder: TransformerEncoder,
    x: PackedTensorSequences,
    memory: Optional[list[PackedTensorSequences]],
    self_buffer: list[torch.Tensor],
    buffer: list[torch.Tensor],
) -> PackedTensorSequences:
    """
    does not implement self_memory b/c we won't be testing that code path atm
    also, it technically requires more calculations relating to position to make the
    code "look right", even though it is not necessary to do for RoPE
    """
    for layer_idx, layer in enumerate(decoder.layers):
        layer: TieredRotaryTransformerEncoderLayer

        # apply the self attention layer on the sequences independently
        x_norm = copy.copy(x)
        x_norm.x = layer.norm1.forward(x.x)
        x_norm_key, x_norm_value = _compute_attn_memory(x_norm, layer.self_attn)
        key_buffer, value_buffer = (
            self_buffer[2 * layer_idx],
            self_buffer[2 * layer_idx + 1],
        )
        key_buffer[:, -1], value_buffer[:, -1] = x_norm_key.x, x_norm_value.x
        key_memory = PackedTensorSequences.pack_input(key_buffer)
        value_memory = PackedTensorSequences.pack_input(value_buffer)
        key_memory.x = key_memory.x.unflatten(1, (x_norm_key.x.size(1), -1))
        value_memory.x = value_memory.x.unflatten(1, (x_norm_value.x.size(1), -1))
        try:
            layer.self_attn.self_attention = False
            x2: torch.Tensor
            x2, _ = layer.self_attn.forward_packed(
                x_norm,
                key_memory,
                value_memory,
                attn_mask=None,
                key_padding_mask=None,
                return_weights=False,
                transform_query=True,
                transform_key=False,
                transform_value=False,
            )
        finally:
            layer.self_attn.self_attention = True
        x = copy.copy(x)
        x.x = x.x + layer.dropout1.forward(x2.x)

        # apply the sequence-of-sequence attention layer on the reshaped sequences
        x_norm = copy.copy(x)
        x_norm.x = layer.norm2.forward(x.x)
        x_norm_key, x_norm_value = _compute_attn_memory(x_norm, layer.multihead_attn)
        key_buffer, value_buffer = (
            buffer[2 * layer_idx],
            buffer[2 * layer_idx + 1],
        )
        key_buffer[:, -1], value_buffer[:, -1] = x_norm_key.x, x_norm_value.x
        if memory is not None:
            key_memory, value_memory = (
                copy.copy(memory[2 * layer_idx]),
                copy.copy(memory[2 * layer_idx + 1]),
            )
            key_memory.x, value_memory.x = (
                key_memory.x.to(x.x.device),
                value_memory.x.to(x.x.device),
            )
            _packed_sequence_append(key_memory, x=key_buffer)
            _packed_sequence_append(value_memory, x=value_buffer)
        else:
            # TODO: this code path may be untested
            key_memory = PackedTensorSequences.pack_input(key_buffer)
            value_memory = PackedTensorSequences.pack_input(value_buffer)
            key_memory.x = key_memory.x.unflatten(1, (x_norm_key.x.size(1), -1))
            value_memory.x = value_memory.x.unflatten(1, (x_norm_value.x.size(1), -1))
        try:
            layer.multihead_attn.self_attention = False
            x2: torch.Tensor
            x2, _ = layer.multihead_attn.forward_packed(
                x_norm,
                key_memory,
                value_memory,
                attn_mask=None,
                key_padding_mask=None,
                return_weights=False,
                transform_query=True,
                transform_key=False,
                transform_value=False,
            )
        finally:
            layer.multihead_attn.self_attention = True
        x = copy.copy(x)
        x.x = x.x + layer.dropout2.forward(x2.x)

        x2 = layer.linear2(layer.dropout(gelu(layer.linear1(layer.norm3(x.x)))))
        x.x = x.x + layer.dropout3(x2)
    return x


class PoET(nn.Module, LogitsAllocateMemoryMixin):
    def __init__(
        self,
        n_vocab: int,
        hidden_dim: int = 768,
        ff_dim: Optional[int] = None,
        num_layers: int = 6,
        nhead: int = 12,
        dropout: float = 0,
        use_multi_rotary: bool = True,
        norm: bool = False,
        mask_token: int = 21,  # kept just to maintain compatability with old models
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.token_embed = nn.Embedding(n_vocab, hidden_dim)
        # kept just to maintain compatability with old models
        self.rotary_emb = RotaryEmbedding(hidden_dim // nhead)

        ff_dim = ff_dim or 4 * hidden_dim

        self.decoder = TransformerEncoder(
            encoder_layer=TieredRotaryTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                use_multi_rotary=use_multi_rotary,
                batch_first=True,
                causal=True,
            ),
            num_layers=num_layers,
        )

        if norm:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.norm = nn.Identity()

        self.linear = nn.Linear(hidden_dim, n_vocab)

    def embed(
        self,
        xs: torch.Tensor,
        segment_sizes: torch.Tensor,
        allow_cpu_offload: bool = False,
        pbar_position: Optional[int] = None,
    ) -> list[PackedTensorSequences]:
        """
        Returns the memory of each layer in a list. The memory is the input to the
        multi-sequence attention.

        Args:
          xs:
            (B, L) sequence of sequences
          segment_sizes:
            (B, N) the lengths of each sequence in the sequence of sequences
          allow_cpu_offload:
            whether or not memory should be offloaded to cpu if CUDA OOMs
          pbar_position:
            position of a tqdm progress bar if not None

        Returns:
          The memory. If allow_cpu_offload and there is insufficient GPU memory to
          store the tensors, the tensors will be stored in CPU memory instead.
        """
        seqs_seqlens = segment_sizes.sum(dim=1).type(torch.int32)
        xs, _, _, _ = unpad_input(xs.unsqueeze(2), ~get_mask(seqs_seqlens))
        xs = xs.squeeze(1)
        h = self.token_embed.forward(xs)

        segment_sizes_cpu = segment_sizes.cpu()
        seqs_seqlens_cpu = segment_sizes_cpu.sum(dim=1).type(torch.int32)
        nonzero_segment_sizes_cpu = (
            segment_sizes_cpu[segment_sizes_cpu > 0].flatten().type(torch.int32)
        )
        cu_seqlens_cpu = F.pad(
            nonzero_segment_sizes_cpu.cumsum(
                dim=0, dtype=nonzero_segment_sizes_cpu.dtype
            ),
            (1, 0),
        )
        cu_seqlens = cu_seqlens_cpu.to(xs.device)
        h = PackedTensorSequences(
            packed_tensor=h,
            positions=torch.cat(
                [
                    torch.arange(segment_size, dtype=xs.dtype, device=xs.device)
                    for segment_size in nonzero_segment_sizes_cpu
                ]
            ),
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_s=nonzero_segment_sizes_cpu.max(),
            # only needed for unpadding (used in standard attn)
            to_paddedable=False,
            indices=None,
            batch_size=None,
        )

        memory = []
        output_device: Optional[torch.device] = None
        if pbar_position is None:
            layers = self.decoder.layers
        else:
            layers = tqdm(
                self.decoder.layers,
                desc=f"[{pbar_position}] encoding",
                leave=False,
                position=pbar_position,
            )
        for layer in layers:
            layer: TieredRotaryTransformerEncoderLayer
            try:
                h, (_, _), (key, value) = layer.forward(
                    h,
                    seqs_cu_seqlens=F.pad(
                        seqs_seqlens.cumsum(dim=0, dtype=seqs_seqlens.dtype), (1, 0)
                    ),
                    seqs_cu_seqlens_cpu=F.pad(
                        seqs_seqlens_cpu.cumsum(dim=0, dtype=seqs_seqlens.dtype),
                        (1, 0),
                    ),
                    return_memory=True,
                )
                if output_device is not None:
                    key.x = key.x.to(output_device)
                    value.x = value.x.to(output_device)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and allow_cpu_offload:
                    if pbar_position is not None:
                        tqdm.write(
                            "OOMed during encoding, retrying by offloading to cpu"
                        )
                    torch.cuda.empty_cache()
                    output_device = torch.device("cpu")
                    for this_memory in memory:
                        this_memory.x = this_memory.x.to(output_device)
                    torch.cuda.empty_cache()
                    h, (_, _), (key, value) = layer.forward(
                        h,
                        seqs_cu_seqlens=F.pad(
                            seqs_seqlens.cumsum(dim=0, dtype=seqs_seqlens.dtype), (1, 0)
                        ),
                        seqs_cu_seqlens_cpu=F.pad(
                            seqs_seqlens_cpu.cumsum(dim=0, dtype=seqs_seqlens.dtype),
                            (1, 0),
                        ),
                        return_memory=True,
                    )
                    key.x = key.x.to(output_device)
                    value.x = value.x.to(output_device)
                else:
                    raise e
            memory.append(key)
            memory.append(value)
        return memory

    def logits(
        self,
        x: torch.Tensor,
        memory: Optional[list[PackedTensorSequences]],
        preallocated_memory: bool = False,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the next token probability distributions given a precomputed memory
        (see self.embed and/or self.logits_allocate_memory).

        Args
          x:
            (B, L) sequence of sequences of tokens
          memory:
            output of self.embed
            if not preallocated_memory, has batch size 1 (it will be expanded if necessary)
            if memory is not on the same device as x, a copy of memory will be made to the
            device of x as necessary
          preallocated_memory:
            whether or not additional memory needed for this method was preallocated
            using self.logits_allocate_memory

        Returns:
          logits:
            (B, L, V) logits of the next token probability distributions. Here, V is
            the vocabulary size
        """
        B, L_x = x.size()

        x: PackedTensorSequences = PackedTensorSequences.pack_input(x.unsqueeze(2))
        x.x = self.token_embed.forward(x.x.squeeze(1))

        x = _apply_causal_prefix_attention(
            decoder=self.decoder,
            x=x,
            batch_size=B,
            length=L_x,
            self_memory=None,
            memory=memory,
            preallocated_memory=preallocated_memory,
        )

        embeddings = self.norm(x.x)
        logits = self.linear.forward(embeddings).view(B, L_x, -1)
        if not return_embeddings:
            return logits
        else:
            return logits, embeddings.view(B, L_x, -1)

    def sample(
        self,
        xs: torch.Tensor,
        segment_sizes: torch.Tensor,
        temperature: float = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        maxlen: int = 1000,
        alphabet: Uniprot21 = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        ),
        remove_invalid: bool = True,
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, float]:
        """Sample batch_size sequences.

        Note: this implementation is out of date
        """
        return self.sample_given_memory(
            memory=self.embed(xs, segment_sizes),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            maxlen=maxlen,
            alphabet=alphabet,
            remove_invalid=remove_invalid,
            batch_size=batch_size,
        )

    @torch.inference_mode()
    def sample_given_memory(
        self,
        memory: Optional[list[PackedTensorSequences]],
        temperature: float = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        maxlen: int = 1000,
        alphabet: Uniprot21 = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        ),
        remove_invalid: bool = True,
        batch_size: int = 1,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Sample batch_size sequences from memory.

        Assumes memory represents one prompt, and samples each sequence from that one
        prompt.

        Note: this implementation is out of date

        Args:
          memory:
            Output of self.embed
            Must only describe one sequence of sequences i.e. have a batch size of 1
          temperature:
            Controls the randomness of the sampling by dividing the logits
          top_k:
            Controls the number of most probable tokens to consider at each step of
            sampling
            Default is None, which means all tokens are considered
          top_p:
            Controls the cumulative probability of the most probable tokens to consider
            at each step of sampling as in nucleus sampling
            Default is None, which is equivalent to the behavior with top_p=1
          maxlen:
            Maximum sequence length to sample, not including start and stop tokens
            Thus, returned sequences with have length up to maxlen+2, where the first
            token is the start token, and the last token is the stop token if the
            sequence terminates within maxlen tokens.
          alphabet:
            The alphabet encoding the sequence.
          remove_invalid:
            Whether or not to avoid sampling non-amino acids within a sequence.
          batch_size:
            Number of sequences to sample in parallel

        Returns:
          A tuple (sample_xs, sample_scores), where sample_xs is a list containing the
          sampled sequences as tensors encoded by alphabet, and sample_scores is a
          tensor containing the negative log likelihood of each sampled sequence.
        """
        criteria = nn.CrossEntropyLoss(
            ignore_index=alphabet.mask_token, reduction="none"
        )
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        invalid_tokens = torch.tensor(
            [alphabet.mask_token, alphabet.start_token, alphabet.gap_token],
            device=device,
        )
        nhead = self.decoder.layers[0].num_heads
        head_dim = self.decoder.layers[0].dim // nhead

        # initialize memory buffer
        buffer_size = (batch_size, maxlen + 2, nhead, head_dim)
        self_buffer = [
            torch.empty(buffer_size, device=device, dtype=dtype)
            for _ in range(2 * len(self.decoder.layers))
        ]
        buffer = [
            torch.empty(buffer_size, device=device, dtype=dtype)
            for _ in range(2 * len(self.decoder.layers))
        ]

        # initialize x
        current_token = (
            torch.ones((batch_size, 1), dtype=torch.long, device=device)
            * alphabet.start_token
        )
        current_x = current_token
        current_position = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        current_position_int = 0
        current_logits: Optional[torch.Tensor] = None

        # sample rest of x
        sampled_xs, sampled_scores = [], []
        while True:
            # get logits for current x
            x: PackedTensorSequences = PackedTensorSequences.pack_input(
                current_token.unsqueeze(2),
                positions=current_position,
            )
            x.x = self.token_embed.forward(x.x.squeeze(1))
            x = _apply_causal_prefix_attention_buffered(
                decoder=self.decoder,
                x=x,
                memory=memory,
                self_buffer=[buf[:, : current_position_int + 1] for buf in self_buffer],
                buffer=[buf[:, : current_position_int + 1] for buf in buffer],
            )
            embeddings = self.norm(x.x)
            logits = self.linear.forward(embeddings).unsqueeze(1)

            # sample the next token
            next_token_logits = logits[:, -1].log_softmax(dim=1)
            if remove_invalid:
                next_token_logits[:, invalid_tokens] += -torch.inf
            next_token_logits /= temperature
            next_token_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            next_token = torch.multinomial(
                next_token_logits.float().softmax(dim=-1), 1
            ).flatten()

            # update state
            current_token = next_token.unsqueeze(1)
            current_x = torch.cat([current_x, current_token], dim=1)
            current_position = current_position + 1
            current_position_int += 1
            if current_logits is None:
                current_logits = logits
            else:
                current_logits = torch.cat([current_logits, logits], dim=1)

            # apply sampling termination conditions
            is_stop_batch_filter = (
                (next_token == alphabet.stop_token)
                if current_x.size(1) < maxlen + 2
                else torch.ones((current_x.size(0),), dtype=torch.bool, device=device)
            )
            if is_stop_batch_filter.sum() > 0:
                is_stop_batch_idxs = torch.where(is_stop_batch_filter)[0]
                not_is_stop_batch_idxs = torch.where(~is_stop_batch_filter)[0]

                sampled_xs.extend(current_x[is_stop_batch_idxs].unbind())
                sampled_scores.append(
                    -criteria.forward(
                        current_logits[is_stop_batch_idxs].transpose(1, 2),
                        current_x[is_stop_batch_idxs, 1:].cuda(),
                    )
                    .float()
                    .sum(dim=1)
                )
                if is_stop_batch_idxs.numel() == current_x.size(0):
                    break
                else:
                    # remove terminated sequences from state
                    _filter = not_is_stop_batch_idxs
                    current_token = current_token[_filter]
                    current_x = current_x[_filter]
                    current_position = current_position[_filter]
                    current_logits = current_logits[_filter]
                    for idx in range(len(self_buffer)):
                        self_buffer[idx] = self_buffer[idx][_filter]
                    for idx in range(len(buffer)):
                        buffer[idx] = buffer[idx][_filter]
        return sampled_xs, torch.hstack(sampled_scores)

    @torch.inference_mode()
    def sample_given_memories(
        self,
        memory: list[PackedTensorSequences],
        temperature: float = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        maxlen: int = 1000,
        alphabet: Uniprot21 = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        ),
        remove_invalid: bool = True,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Sample one sequence for each prompt described by memory.

        Unlike self.sample_given_memory, memory can represent multiple prompts.

        This method may have higher memory requirements than self.sample_given_memory
        and self.sample_given_memories_ensemble. Roughly speaking, it may allocate
        additional memory equal to the total memory used by `memory`, whereas the other
        methods may allocate additional memory equal to the memory used by only two
        items in `memory` e.g. memory[0] and memory[1].

        Note: this implementation is out of date

        Args:
          memory:
            Output of self.embed
          temperature:
            Controls the randomness of the sampling by dividing the logits
          top_k:
            Controls the number of most probable tokens to consider at each step of
            sampling
            Default is None, which means all tokens are considered
          top_p:
            Controls the cumulative probability of the most probable tokens to consider
            at each step of sampling as in nucleus sampling
            Default is None, which is equivalent to the behavior with top_p=1
          maxlen:
            Maximum sequence length to sample, not including start and stop tokens
            Thus, returned sequences with have length up to maxlen+2, where the first
            token is the start token, and the last token is the stop token if the
            sequence terminates within maxlen tokens.
          alphabet:
            The alphabet encoding the sequence.
          remove_invalid:
            Whether or not to avoid sampling non-amino acids within a sequence.

        Returns:
          A tuple (sample_xs, sample_scores), where sample_xs is a list containing the
          sampled sequences as tensors encoded by alphabet, and sample_scores is a
          tensor containing the negative log likelihood of each sampled sequence.

          The order of the samples corresponds to the order of the prompts i.e. the nth
          sample in sample_xs/sample_scores is sampled from the nth prompt in memory.
        """
        criteria = nn.CrossEntropyLoss(
            ignore_index=alphabet.mask_token, reduction="none"
        )
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        invalid_tokens = torch.tensor(
            [alphabet.mask_token, alphabet.start_token, alphabet.gap_token],
            device=device,
        )
        batch_size = memory[0].cu_seqlens.numel() - 1
        nhead = self.decoder.layers[0].num_heads
        head_dim = self.decoder.layers[0].dim // nhead

        # initialize memory buffer
        buffer_size = (batch_size, maxlen + 2, nhead, head_dim)
        self_buffer = [
            torch.empty(buffer_size, device=device, dtype=dtype)
            for _ in range(2 * len(self.decoder.layers))
        ]
        buffer = [
            torch.empty(buffer_size, device=device, dtype=dtype)
            for _ in range(2 * len(self.decoder.layers))
        ]

        # initialize x
        current_token = (
            torch.ones((batch_size, 1), dtype=torch.long, device=device)
            * alphabet.start_token
        )
        current_x = current_token
        current_position = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        current_position_int = 0
        current_logits: Optional[torch.Tensor] = None

        # sample rest of x
        sampled_xs, sampled_scores = [], []
        sampled_order, remaining_order = [], torch.arange(batch_size, device=device)
        while True:
            # get logits for current x
            x: PackedTensorSequences = PackedTensorSequences.pack_input(
                current_token.unsqueeze(2),
                positions=current_position,
            )
            x.x = self.token_embed.forward(x.x.squeeze(1))
            B = x.cu_seqlens.numel() - 1
            x = _apply_causal_prefix_attention_buffered(
                decoder=self.decoder,
                x=x,
                memory=memory,
                self_buffer=[buf[:, : current_position_int + 1] for buf in self_buffer],
                buffer=[buf[:, : current_position_int + 1] for buf in buffer],
            )
            embeddings = self.norm(x.x)
            logits = self.linear.forward(embeddings).view(B, 1, -1)

            # sample the next token
            next_token_logits = logits[:, -1].log_softmax(dim=1)
            if remove_invalid:
                next_token_logits[:, invalid_tokens] += -torch.inf
            next_token_logits /= temperature
            next_token_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            next_token = torch.multinomial(
                next_token_logits.float().softmax(dim=-1), 1
            ).flatten()

            # update state
            current_token = next_token.unsqueeze(1)
            current_x = torch.cat([current_x, current_token], dim=1)
            current_position = current_position + 1
            current_position_int += 1
            if current_logits is None:
                current_logits = logits
            else:
                current_logits = torch.cat([current_logits, logits], dim=1)

            # apply sampling termination conditions
            is_stop_batch_filter = (
                (next_token == alphabet.stop_token)
                if current_x.size(1) < maxlen + 2
                else torch.ones((current_x.size(0),), dtype=torch.bool, device=device)
            )
            if is_stop_batch_filter.sum() > 0:
                is_stop_batch_idxs = torch.where(is_stop_batch_filter)[0]
                not_is_stop_batch_idxs = torch.where(~is_stop_batch_filter)[0]
                not_is_stop_batch_idxs_cpu = not_is_stop_batch_idxs.cpu()

                sampled_order.append(remaining_order[is_stop_batch_idxs])
                remaining_order = remaining_order[not_is_stop_batch_idxs]
                sampled_xs.extend(current_x[is_stop_batch_idxs].unbind())
                sampled_scores.append(
                    -criteria.forward(
                        current_logits[is_stop_batch_idxs].transpose(1, 2),
                        current_x[is_stop_batch_idxs, 1:].cuda(),
                    )
                    .float()
                    .sum(dim=1)
                )
                if is_stop_batch_idxs.numel() == current_x.size(0):
                    break
                else:
                    # remove terminated sequences from state
                    _filter = not_is_stop_batch_idxs
                    _filter_cpu = not_is_stop_batch_idxs_cpu
                    current_token = current_token[_filter]
                    current_x = current_x[_filter]
                    current_position = current_position[_filter]
                    current_logits = current_logits[_filter]
                    for idx in range(len(self_buffer)):
                        self_buffer[idx] = self_buffer[idx][_filter]
                    for idx in range(len(buffer)):
                        buffer[idx] = buffer[idx][_filter]

                    new_start_idxs = memory[0].cu_seqlens_cpu[:-1][_filter_cpu]
                    new_end_idxs = memory[0].cu_seqlens_cpu[1:][_filter_cpu]
                    filtered_idxs = torch.hstack(
                        [
                            torch.arange(
                                new_start_idxs[idx], new_end_idxs[idx], device=device
                            )
                            for idx in range(_filter.numel())
                        ]
                    )
                    memory = [copy.copy(mem) for mem in memory]
                    for mem in memory:
                        mem.x = mem.x[filtered_idxs]
                        mem.positions = mem.positions[filtered_idxs]
                        mem.cu_seqlens = F.pad(
                            mem.cu_seqlens.diff()[_filter].cumsum(
                                dim=0, dtype=torch.int32
                            ),
                            (1, 0),
                        )
                        mem.cu_seqlens_cpu = F.pad(
                            mem.cu_seqlens_cpu.diff()[_filter_cpu].cumsum(
                                dim=0, dtype=torch.int32
                            ),
                            (1, 0),
                        )
                        mem.max_s = mem.cu_seqlens_cpu.diff().max()
                        mem.to_paddedable = False

        # order sampled sequences by the order of the input memories
        sampled_order = torch.hstack(sampled_order).argsort()
        sampled_xs = [sampled_xs[i] for i in sampled_order]
        sampled_scores = torch.hstack(sampled_scores)[sampled_order]
        return sampled_xs, sampled_scores

    @torch.inference_mode()
    def sample_given_memories_ensemble(
        self,
        memory: list[PackedTensorSequences],
        temperature: float = 1,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        maxlen: int = 1000,
        alphabet: Uniprot21 = Uniprot21(
            include_gap=True, include_startstop=True, distinct_startstop=True
        ),
        remove_invalid: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one sequence by ensembling the prompts described by memory.

        Note: this implementation is out of date

        Args:
          memory:
            Output of self.embed
          temperature:
            Controls the randomness of the sampling by dividing the logits
          top_k:
            Controls the number of most probable tokens to consider at each step of
            sampling
            Default is None, which means all tokens are considered
          top_p:
            Controls the cumulative probability of the most probable tokens to consider
            at each step of sampling as in nucleus sampling
            Default is None, which is equivalent to the behavior with top_p=1
          maxlen:
            Maximum sequence length to sample, not including start and stop tokens
            Thus, returned sequences with have length up to maxlen+2, where the first
            token is the start token, and the last token is the stop token if the
            sequence terminates within maxlen tokens.
          alphabet:
            The alphabet encoding the sequence.
          remove_invalid:
            Whether or not to avoid sampling non-amino acids within a sequence.

        Returns:
          A tuple (sample_x, sample_scores), where sample_x is the sampled sequence
          encoded by alphabet, and sample_scores is a tensor containing the negative
          log likelihood of sample_x conditioned on each prompt in memory.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        invalid_tokens = torch.tensor(
            [alphabet.mask_token, alphabet.start_token, alphabet.gap_token],
            device=device,
        )
        batch_size = memory[0].cu_seqlens.numel() - 1
        nhead = self.decoder.layers[0].num_heads
        head_dim = self.decoder.layers[0].dim // nhead

        # initialize memory buffer
        buffer_size = (batch_size, maxlen + 2, nhead, head_dim)
        self_buffer = [
            torch.empty(buffer_size, device=device, dtype=dtype)
            for _ in range(2 * len(self.decoder.layers))
        ]
        buffer = [
            torch.empty(buffer_size, device=device, dtype=dtype)
            for _ in range(2 * len(self.decoder.layers))
        ]

        # initialize x
        current_token = (
            torch.ones((batch_size, 1), dtype=torch.long, device=device)
            * alphabet.start_token
        )
        current_x = current_token
        current_position = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        current_position_int = 0
        current_logits_sum = torch.zeros(
            (batch_size,), dtype=torch.float32, device=device
        )

        # sample rest of x
        while True:
            # get logits for current x
            x: PackedTensorSequences = PackedTensorSequences.pack_input(
                current_token.unsqueeze(2),
                positions=current_position,
            )
            x.x = self.token_embed.forward(x.x.squeeze(1))
            B = x.cu_seqlens.numel() - 1
            x = _apply_causal_prefix_attention_buffered(
                decoder=self.decoder,
                x=x,
                memory=memory,
                self_buffer=[buf[:, : current_position_int + 1] for buf in self_buffer],
                buffer=[buf[:, : current_position_int + 1] for buf in buffer],
            )
            embeddings = self.norm(x.x)
            logits = self.linear.forward(embeddings).view(B, 1, -1)

            # sample the next token
            next_token_logits = logits[:, -1].log_softmax(dim=1)
            weights = current_logits_sum.softmax(dim=0)
            per_memory_next_token_logits = next_token_logits
            next_token_logits = (next_token_logits * weights.unsqueeze(1)).sum(dim=0)
            if remove_invalid:
                next_token_logits[invalid_tokens] += -torch.inf
            next_token_logits /= temperature
            next_token_logits = top_k_top_p_filtering(
                next_token_logits.unsqueeze(0), top_k=top_k, top_p=top_p
            ).squeeze(0)
            next_token = torch.multinomial(next_token_logits.float().softmax(dim=-1), 1)

            # update state
            current_token = next_token.unsqueeze(0).expand(batch_size, -1)
            current_x = torch.cat([current_x, current_token], dim=1)
            current_position = current_position + 1
            current_position_int += 1
            current_logits_sum += per_memory_next_token_logits[:, next_token].flatten()

            # apply sampling termination conditions
            if next_token == alphabet.stop_token or current_x.size(1) == maxlen + 2:
                return current_x[0], current_logits_sum

    def forward(self, xs: torch.Tensor, segment_sizes: torch.Tensor) -> torch.Tensor:
        """
        Compute the next token probability distributions.

        Examples:
          Example input with batch size 1

          xs: [$ A B * $ A B C * $ E F]
          segment_sizes: [[4, 5, 3]]

          Note that the last sequence in a sequence of sequences does not need to have a
          stop token.

        Args:
          xs:
            (B, L) sequence of sequences of tokens
          segment_sizes:
            (B, N) the lengths of each sequence in the sequence of sequences

        Returns:
          (B, L, V) logits of the next token probability distributions. Here, V is
          the vocabulary size

        """
        B, L = xs.size()

        seqs_seqlens = segment_sizes.sum(dim=1).type(torch.int32)
        xs, indices, _, _ = unpad_input(xs.unsqueeze(2), ~get_mask(seqs_seqlens))
        xs = xs.squeeze(1)
        h = self.token_embed.forward(xs)

        segment_sizes_cpu = segment_sizes.cpu()
        seqs_seqlens_cpu = segment_sizes_cpu.sum(dim=1).type(torch.int32)
        nonzero_segment_sizes_cpu = (
            segment_sizes_cpu[segment_sizes_cpu > 0].flatten().type(torch.int32)
        )
        cu_seqlens_cpu = F.pad(
            nonzero_segment_sizes_cpu.cumsum(
                dim=0, dtype=nonzero_segment_sizes_cpu.dtype
            ),
            (1, 0),
        )
        cu_seqlens = cu_seqlens_cpu.to(xs.device)
        h = PackedTensorSequences(
            packed_tensor=h,
            positions=torch.cat(
                [
                    torch.arange(segment_size, dtype=xs.dtype, device=xs.device)
                    for segment_size in nonzero_segment_sizes_cpu
                ]
            ),
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            max_s=nonzero_segment_sizes_cpu.max(),
            # only needed for unpadding (used in standard attn)
            to_paddedable=False,
            indices=None,
            batch_size=None,
        )
        h = self.decoder.forward(
            h,
            seqs_cu_seqlens=F.pad(
                seqs_seqlens.cumsum(dim=0, dtype=seqs_seqlens.dtype), (1, 0)
            ),
            seqs_cu_seqlens_cpu=F.pad(
                seqs_seqlens_cpu.cumsum(dim=0, dtype=seqs_seqlens_cpu.dtype),
                (1, 0),
            ),
        )

        logits = self.linear.forward(self.norm(h.x))
        logits, _ = pad_input(logits, indices, B, L)  # (B,L,num_tokens)
        return logits
