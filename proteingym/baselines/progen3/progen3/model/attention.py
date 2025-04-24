import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from transformers.cache_utils import Cache
from transformers.utils import logging

from ..config import ProGen3Config

logger = logging.get_logger(__name__)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch,
    num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Adapted from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding and
# transformers.models.llama.modeling_llama.apply_rotary_pos_emb
class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000, device: torch.device | None = None
    ):
        super().__init__()

        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = base ** -(torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_sin_cos_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
        )

    def _set_sin_cos_cache(self, seq_len: int, device: torch.device) -> None:
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq.to(device))
        angles = torch.cat((angles, angles), dim=1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [bsz, n, num_attention_heads, head_size]
        # position_ids: [bsz, n]
        device, dtype = q.device, q.dtype

        # max position id can be different from number of tokens in the sequence
        # For example, for GLM/Infilling
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_sin_cos_cache(seq_len=seq_len, device=device)

        # angles_cached[position_ids] gets us something of shape (batch_size, seq_len, head_dim),
        # so unsqueeze dimension -2 to broadcast to (batch_size, seq_len, n_heads, head_dim).
        idxs = position_ids.to(device)
        cos = self.cos_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]
        sin = self.sin_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention
class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: ProGen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.max_num_seqs = config.max_num_sequences
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout
        self.clip_qkv = config.clip_qkv

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Cache | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        val_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        if self.clip_qkv is not None:
            query_states = query_states.clamp(-self.clip_qkv, self.clip_qkv)
            key_states = key_states.clamp(-self.clip_qkv, self.clip_qkv)
            val_states = val_states.clamp(-self.clip_qkv, self.clip_qkv)

        query_states, key_states = self.rotary_emb(
            query_states,
            key_states,
            position_ids,
        )

        if use_cache and past_key_value is not None:
            key_states, val_states = key_states.transpose(1, 2), val_states.transpose(1, 2)
            key_states, val_states = past_key_value.update(key_states, val_states, self.layer_idx)
            key_states, val_states = key_states.transpose(1, 2), val_states.transpose(1, 2)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        if input_dtype != target_dtype:
            logger.warning_once(
                f"The input hidden states seems to be silently casted in {input_dtype}. "
                f"This might be because you have upcasted embedding or layer norm layers "
                f"in {input_dtype}. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            val_states = val_states.to(target_dtype)

        return query_states, key_states, val_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        query_states, key_states, val_states = self.prepare_qkv(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        attn_output, attn_weights = self._attn(
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
            output_attentions=output_attentions,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

    def _attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        output_attentions: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert not output_attentions, "output_attentions not supported"
        return self._sdpa_attn(
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
        )

    def _sdpa_attn(
        self, query_states: torch.Tensor, key_states: torch.Tensor, val_states: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        val_states = val_states.transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        # enable_gqa in F.sdpa is not supported for all backends yet
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        val_states = repeat_kv(val_states, self.num_key_value_groups)

        bsz, q_len = query_states.shape[0], query_states.shape[2]
        k_len = key_states.shape[2]

        causal_mask = None
        if k_len > q_len:
            causal_mask = causal_lower_right(q_len, k_len)
        elif k_len < q_len:
            raise ValueError("k_len must be greater than or equal to q_len")

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                val_states,
                is_causal=causal_mask is None,
                attn_mask=causal_mask,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return attn_output, None
