# mypy: ignore-errors
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import functools
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import megablocks.layers.moe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from megablocks.layers.dmoe import dMoE
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import GenerationMixin, PreTrainedModel
from transformers.utils import logging

from progen3.common.model_loading import init_empty_weights

try:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn
except ImportError:
    raise ImportError(
        "triton_rms_norm requires Flash Attention to be installed. " + "Please pip install flash-attn.",
    )

from .config import ProGen3Config
from .model.attention import Attention
from .model.mb_wrapper import mb_setup_args
from .model.moe import MOE_CLASSES

logger = logging.get_logger(__name__)


def _update_state_dict(
    state_dict: Mapping[str, Any],
    config: ProGen3Config,
):
    # Make state dict interoperable between megablocks implementations
    key_sub = {}
    if config.moe_implementation == "eager":
        # TODO: add megablocks to eager substitutions here
        key_sub = {}

    def update_key(key):
        for k, v in key_sub.items():
            key = re.sub(k, v, key)
        return key

    return {update_key(k): v for k, v in state_dict.items()}


@dataclass
class MoeModelOutputWithPast(ModelOutput):
    """Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

        router_weights (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_weights=True` and `config.add_router_weights=True` is passed or when `config.output_router_weights=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router weights (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_weights: tuple[torch.FloatTensor] | None = None


@dataclass
class MoeCausalOutputWithPast(ModelOutput):
    """Base class for joint causal/masked language model with mixture of experts outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total loss.

        ar_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Autoregressive language modeling loss.

        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.

        router_weights (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_weights=True` or `config.output_router_weights=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Raw router logits (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
            loss for Mixture of Experts models.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: torch.FloatTensor | None = None
    ar_loss: torch.FloatTensor | None = None
    aux_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_weights: tuple[torch.FloatTensor] | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

        if not isinstance(hidden_size, int):
            raise ValueError("TritonRMSNorm only supports 1D tensors")

        self.rms_norm_fn = rms_norm_fn

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        return self.rms_norm_fn(
            hidden_states,
            self.weight,
            None,  # no bias
            residual=None,
            eps=self.variance_epsilon,
            dropout_p=0.0,  # no dropout by default
            prenorm=False,
            residual_in_fp32=False,
        ).to(input_dtype)


class NormAttentionNorm(nn.Module):
    def __init__(self, config: ProGen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, Cache | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, residual, self_attn_weights, present_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: ProGen3Config, layer_idx: int, **moe_kwargs):
        super().__init__()
        self.initializer_range = config.initializer_range
        self.hidden_size = config.hidden_size
        self.fused_attention_norm = config.fused_attention_norm
        if self.fused_attention_norm:
            self.norm_attn_norm = NormAttentionNorm(config, layer_idx)
        else:
            self.self_attn = Attention(config, layer_idx)
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_sparse_moe = MOE_CLASSES[config.moe_implementation](config, **moe_kwargs)
        self.moe_implementation = config.moe_implementation

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = None,
        output_router_weights: bool | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.fused_attention_norm:
            hidden_states, residual, self_attn_weights, present_key_value = self.norm_attn_norm(
                hidden_states=hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        # Fully Connected
        if self.moe_implementation == "megablocks":
            hidden_states = self.block_sparse_moe(hidden_states)
        else:
            hidden_states, router_weights = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_weights:
            outputs += (router_weights,)
        return outputs


class ProGen3PreTrainedModel(PreTrainedModel):
    config_class = ProGen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _transformer_layer_cls = [DecoderLayer]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = True
    _vocab_keys = []

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def post_init(self):
        super().post_init()
        self._set_update_state_dict()

    def _set_update_state_dict(self, update_fn: Callable | None = None):
        if update_fn is None:
            update_fn = functools.partial(
                _update_state_dict,
                config=self.config,
            )
        self._update_state_dict = update_fn
        for child in self._modules.values():
            child._update_state_dict = update_fn
            if isinstance(child, ProGen3PreTrainedModel):
                child._set_update_state_dict(update_fn)

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        state_dict = self._update_state_dict(state_dict)
        return super()._load_from_state_dict(state_dict, *args, **kwargs)

    def param_init_fn(self, module):
        std = self.config.initializer_range
        if isinstance(module, dMoE):
            module.experts.mlp.w1.data.normal_(mean=0.0, std=std)
            module.experts.mlp.w2.data.normal_(mean=0.0, std=std)
            if hasattr(module.experts.mlp, "v1"):
                module.experts.mlp.v1.data.normal_(mean=0.0, std=std)
        else:
            self._init_weights(module)

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable(dict(use_reentrant=False))

    def fsdp_wrap_fn(self, module):
        if hasattr(module, "_fsdp_kwargs_dict"):
            return module._fsdp_kwargs_dict
        return isinstance(module, tuple(self._transformer_layer_cls))

    def activation_checkpointing_fn(self, module):
        attn_cls = NormAttentionNorm if self.config.fused_attention_norm else (Attention, RMSNorm)
        ckpt_cls = attn_cls if self.config.no_ffn_gradient_checkpointing else tuple(self._transformer_layer_cls)
        return isinstance(module, ckpt_cls)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, *args, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)


class ProGen3Model(ProGen3PreTrainedModel):
    _vocab_keys = ["embed_tokens.weight"]

    def __init__(self, config: ProGen3Config, meta_init: bool = False):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_seq_id = nn.Embedding(config.max_num_sequences, config.hidden_size)
        p = next(self.embed_tokens.parameters())
        if config.moe_implementation == "megablocks":
            mb_args, device_mesh = mb_setup_args(config, dtype=p.dtype, device=p.device)
            kwargs = dict(args=mb_args, device_mesh=device_mesh)
            self.mb_args = mb_args
            self.expert_parallel_device_mesh = device_mesh
        else:
            kwargs = dict()
        ctx = init_empty_weights if meta_init else contextlib.nullcontext
        with ctx():
            self.layers = nn.ModuleList([DecoderLayer(config, i, **kwargs) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_weights: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...] | MoeModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_weights = (
            output_router_weights if output_router_weights is not None else self.config.output_router_weights
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if output_router_weights and self.config.moe_implementation == "megablocks":
            raise ValueError(f"{output_router_weights=} not compatible with megablocks MoE implementation")
        if self.config.moe_implementation == "megablocks":
            megablocks.layers.moe.clear_load_balancing_loss()

        # retrieve input_ids and inputs_embeds
        batch_size, seq_length = input_ids.shape

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        elif not use_cache:
            # To avoid weirdness with gradient checkpointing: https://github.com/huggingface/transformers/issues/28499
            past_key_values = None

        position_ids = position_ids.view(-1, seq_length).long()
        sequence_ids = sequence_ids.view(-1, seq_length).long()
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds + self.embed_seq_id(sequence_ids)

        # In case we need to do any manual typecasting
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        elif self.config.fused_attention_norm:
            target_dtype = self.layers[0].norm_attn_norm.self_attn.q_proj.weight.dtype
        else:
            target_dtype = self.layers[0].self_attn.q_proj.weight.dtype
        hidden_states = inputs_embeds.to(target_dtype)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_weights = () if output_router_weights else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_weights,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_weights=output_router_weights,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_weights:
                all_router_weights += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_weights,
                ]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_weights=all_router_weights,
        )


class ProGen3ForCausalLM(ProGen3PreTrainedModel, GenerationMixin):
    _vocab_keys = ["model.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: ProGen3Config, meta_init: bool = False):
        super().__init__(config)
        self.model = ProGen3Model(config, meta_init=meta_init)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gradient_checkpointing = config.gradient_checkpointing
        self.post_init()

    @property
    def device_mesh(self):
        return self.model.device_mesh

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_weights: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, ...] | MoeCausalOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_weights = (
            output_router_weights if output_router_weights is not None else self.config.output_router_weights
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_ids=sequence_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_weights=output_router_weights,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        loss = None

        # Compute autoregressive languag modeling loss
        logits = self.lm_head(hidden_states).float()
        if labels is not None:
            # Shift inputs & labels so that tokens < n predict n, and flatten them
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            ar_loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
            mask = shift_labels != self.model.padding_idx
            n_ar = mask.sum()
            ar_loss = (ar_loss * mask.to(ar_loss)).sum() / (1 if n_ar == 0 else n_ar)
            loss = ar_loss
        else:
            n_ar, ar_loss = 0, 0

        aux_loss = None
        if self.config.moe_implementation == "megablocks" and self.training:
            aux_loss = megablocks.layers.moe.batched_load_balancing_loss(self.model.mb_args)
            if loss is not None:
                loss += aux_loss
            aux_loss /= self.router_aux_loss_coef

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_weights:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalOutputWithPast(
            loss=loss,
            ar_loss=None if labels is None else ar_loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_weights=outputs.router_weights,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        position_ids,
        sequence_ids,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case
        if past_key_values is not None:
            if cache_position[-1] >= input_ids.shape[1]:  # Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
                position_ids = position_ids[:, cache_position]
                sequence_ids = sequence_ids[:, cache_position]
            elif input_ids.shape[1] != len(cache_position):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
                position_ids = position_ids[:, cache_position]
                sequence_ids = sequence_ids[:, cache_position]

        model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            position_ids=position_ids,
            sequence_ids=sequence_ids,
            past_key_values=past_key_values,
            use_cache=kwargs.get("use_cache", None),
            output_router_weights=kwargs.get("output_router_weights", None),
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if isinstance(past_key_values, Cache):
            return past_key_values.reorder_cache(beam_idx)

        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return DynamicCache.from_legacy_cache(reordered_past)

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        num_new_tokens: int = 1,
        **kwargs,
    ) -> dict[str, Any]:
        # Change made in transformers>4.42.0 to return two values,
        # cache_name and past_key_values, instead of a single past_key_values
        cache_name, cache = self._extract_past_from_model_output(outputs)
        assert cache_name == "past_key_values", "Only past_key_values is supported"
        model_kwargs["past_key_values"] = cache

        # update position_ids with one plus last value
        pos_ids = model_kwargs["position_ids"]
        new_delta = torch.arange(num_new_tokens, device=pos_ids.device, dtype=pos_ids.dtype).unsqueeze(0) + 1
        model_kwargs["position_ids"] = torch.cat([pos_ids, pos_ids[:, -1:] + new_delta], dim=-1)

        # update sequence_ids with last value
        seq_ids = model_kwargs["sequence_ids"]
        model_kwargs["sequence_ids"] = torch.cat([seq_ids, seq_ids[:, -1:].repeat(1, num_new_tokens)], dim=-1)

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1,
                past_positions[-1] + num_new_tokens + 1,
                dtype=past_positions.dtype,
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

        return model_kwargs
