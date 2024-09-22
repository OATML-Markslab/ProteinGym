from typing import Optional, Union

import torch
import torch.nn.functional as F


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, dim)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, dim), where total_nnz = number of tokens in selected in attention_mask.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    assert hidden_states.size(0) == attention_mask.size(0)
    # padding/unpadding is not invertible when sequence length is less than the mask size
    # because the final position(s) is masked in all sequences...
    # this causes indices to not match with the tensor given by max_seqlen_in_batch
    # there are two possible solutions:
    #   1) first remove these positions from hidden_states
    #   2) set max_seqlen_in_batch to be the number of columns even if fully masked
    # let's opt for (2), because we assume those columns are wanted for some reason

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # max_seqlen_in_batch = seqlens_in_batch.max().item()
    max_seqlen_in_batch = attention_mask.size(-1)
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )

    b, s, d = hidden_states.size()
    hidden_states = hidden_states.reshape(b * s, d)

    selected_hidden_states = torch.gather(
        hidden_states, 0, indices.unsqueeze(1).expand(indices.size(0), d)
    )
    return selected_hidden_states, indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states, indices, batch, seqlen, return_mask=False):
    """
    Arguments:
        hidden_states: (total_nnz, dim), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, dim)
    """
    dim = hidden_states.shape[-1]
    output = torch.zeros(
        (batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype
    )
    output[indices] = hidden_states
    output = output.view(batch, seqlen, dim)
    if return_mask:
        mask = torch.ones(
            (batch * seqlen), device=hidden_states.device, dtype=torch.bool
        )
        mask[indices] = False
        mask = mask.view(batch, seqlen)
        return output, mask
    return output, None


def get_mask(batch_sizes: torch.Tensor) -> torch.Tensor:
    """
    batch_sizes: (B,)

    Returns a bool tensor of shape n_samples x max_batch_size.
    0s are non-masked and 1s and masked elements
    """
    max_len = batch_sizes.max()
    # taken from https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/3
    mask = (
        torch.arange(max_len, device=batch_sizes.device)[None, :]
        >= batch_sizes[:, None]
    )
    return mask


class PackedTensorSequences:
    def __init__(
        self,
        packed_tensor: torch.Tensor,
        positions: torch.Tensor,
        indices: Optional[torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_cpu: torch.Tensor,
        max_s: Union[torch.Tensor, int],
        batch_size: Optional[int],
        to_paddedable: bool = True,
    ):
        """
        If to_paddedable, indicies and batch_size must be set to values that allow this
        object to be correctly padded.
        """
        if to_paddedable:
            assert batch_size is not None

        self.x = packed_tensor
        self.positions = positions
        self.indices = indices
        self.cu_seqlens = cu_seqlens
        self.cu_seqlens_cpu = cu_seqlens_cpu
        self.max_s = max_s
        self.batch_size = batch_size
        self.to_paddedable = to_paddedable

    @property
    def dtype(self):
        return self.x.dtype

    @property
    def is_cuda(self):
        return self.x.is_cuda

    @property
    def device(self):
        return self.x.device

    @staticmethod
    def pack_input(x: torch.Tensor, positions=None, key_padding_mask=None):
        b = x.size(0)
        s = x.size(1)
        if positions is None:
            positions = (
                torch.arange(s, dtype=torch.long, device=x.device)
                .unsqueeze(0)
                .expand(b, s)
            )
        if key_padding_mask is None:
            x_packed = x.reshape(b * s, -1)
            positions = positions.reshape(b * s)
            indices = None
            cu_seqlens = torch.arange(
                0, (b + 1) * s, step=s, dtype=torch.int32, device=x.device
            )
            cu_seqlens_cpu = torch.arange(
                0,
                (b + 1) * s,
                step=s,
                dtype=torch.int32,
            )
            max_s = s
        else:
            # flash attention padding function expects 1 for valid and 0 for invalid positions...
            key_padding_mask_bool = ~(key_padding_mask.bool())
            x_packed, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask_bool)
            cu_seqlens_cpu = cu_seqlens.cpu()
            positions, _, _, _ = unpad_input(
                positions.unsqueeze(2), key_padding_mask_bool
            )
            positions = positions.squeeze(1)
        return PackedTensorSequences(
            x_packed, positions, indices, cu_seqlens, cu_seqlens_cpu, max_s, b
        )

    def to_padded(self, return_mask=False, return_positions=False):
        if not self.to_paddedable:
            raise ValueError("Cannot be to_padded")

        s = self.max_s
        b = self.batch_size
        mask = None
        x = self.x
        pos = self.positions
        if self.indices is None:
            # we are just a flattened matrix...
            x = x.view(b, s, *x.shape[1:])
            pos = pos.view(b, s)
        else:
            dims = None
            if x.ndim > 2:
                dims = x.shape[1:]
                x = x.view(x.size(0), -1)
            x, mask = pad_input(x, self.indices, b, s, return_mask=return_mask)
            pos, _ = pad_input(pos.unsqueeze(1), self.indices, b, s)
            pos = pos.squeeze(2)
            if dims is not None:
                x = x.view(x.size(0), x.size(1), *dims)

        if return_mask and return_positions:
            return x, mask, pos
        elif return_mask:
            return x, mask
        elif return_positions:
            return x, pos
        else:
            return x

    @staticmethod
    def compute_indices(seqlens: torch.Tensor):
        indices_mask = get_mask(seqlens)
        indices = torch.nonzero(~indices_mask.flatten(), as_tuple=False).flatten()
        return indices
