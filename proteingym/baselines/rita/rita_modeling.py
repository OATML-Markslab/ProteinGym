import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput
)

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .rita_configuration import RITAConfig
import torch.nn.functional as F
logger = logging.get_logger(__name__)

@torch.jit.script
def RITA_gelu(hidden_states):
    return hidden_states * 0.5 * (1.0 + torch.tanh(0.79788456 * hidden_states * (1 + 0.044715 * hidden_states * hidden_states)))

class RITAGELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states):
        return RITA_gelu(hidden_states)

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.max_seq_len = config.max_seq_len
        
        head_dim = self.d_model // self.num_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    
    def forward(self, x: torch.FloatTensor, seq_dim=1) -> torch.FloatTensor:
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached
    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    
class SelfAttention(nn.Module):
    """Implementation of MultiHeadAttention following `Karpathy's MinGPT <https://github.com/karpathy/minGPT>`_.
    modified to use rotary embeddings.
    
    Parameters
    ----------
    d_model: int,
         total dimension of the model.
    num_heads: int,
        number of parallel attention heads.
    num_layers: int,
        number of layers in the model, used for the Megatron-like init.
    rotaty_embedding: Optional[Block], default None,
        a RotaryEmbedding Block to add positionnal information in Queries and Keys
    dropout: float, default 0.1,
        amount of dropout on the attention weights.
    sigma: float, default 0.02,
        standard deviation used for the init.
    trainable: bool, default True,
        if False, the Module parameters will be hidden from the optimizer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        rotary_embedding= None,
        dropout: float = 0.1,
        sigma=0.02,
        use_cache: bool = False,
        bias=True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.sigma = sigma
        self.bias = bias

        # key, query, value projections for all heads
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        self.rotary_embedding = rotary_embedding
        self.layer_id = None  # will be set by the Transformer itself
        self.use_cache = use_cache
        self.qkv = None
        self.bias = bias

    def forward(
        self,
        x,
        causal_mask: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        N, L, D = x.size()  # Batch_size, Context_size, d_model

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(N, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        )  # (N, nh, L, hs)
        q = (
            self.query(x).view(N, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        )  # (N, nh, L, hs)
        v = (
            self.value(x).view(N, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        )  # (N, nh, L, hs)
        
        if self.rotary_embedding is not None:
            cos, sin = self.rotary_embedding(x)
            q, k = self.rotary_embedding.apply_rotary_pos_emb(q, k, cos, sin)

        # causal self-attention; Self-attend: (N, nh, L, hs) x (N, nh, hs, L) -> (N, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if causal_mask is not None:
            att[:,:,-L:, -L: ].masked_fill_(causal_mask.view(1, 1, L, L), float("-inf"))
            
        att = (
            att.transpose(0, 2)
            .masked_fill(attention_mask.view(1, 1, N, L)==0, float("-inf"))
            .transpose(0, 2)
            if attention_mask is not None
            else att
        )
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (N, nh, L, L) x (N, nh, L, hs) -> (N, nh, L, hs)
        y = (
            y.transpose(1, 2).contiguous().view(N, L, D)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class DecoderLayer(nn.Module):
    """Transformer block containing the self-attention module and the feedfoward module."""

    def __init__(
        self, config
    ):
        super().__init__()
        self.self_attention = SelfAttention(config.d_model, config.num_heads, config.dropout, rotary_embedding=RotaryEmbedding(config))
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_feedforward, bias=True),
            RITAGELU(),
            nn.Linear(config.d_feedforward, config.d_model, bias=True),
        )
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp_dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.FloatTensor,
        causal_mask: torch.BoolTensor,
        attention_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        y = self.attn_norm(x)
        y = self.self_attention(y, causal_mask=causal_mask, attention_mask=attention_mask)
        x = x + self.attn_dropout(y)

        y = self.mlp_norm(x)
        y = self.mlp(y)
        x = x + self.mlp_dropout(y)
        return x

class RITAModel(PreTrainedModel):
    config_class = RITAConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    
    def __init__(
        self,
        config
    ):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,  # NOT USED
        attention_mask=None,
        causal_mask=None,
        token_type_ids=None, # NOT USED
        position_ids=None, # NOT USED
        head_mask=None, # NOT USED
        inputs_embeds=None,
        encoder_hidden_states=None,  # NOT USED
        encoder_causal_mask=None, # NOT USED
        labels=None,
        use_cache=None, # NOT USED
        output_attentions=None, # NOT USED
        output_hidden_states=None, # NOT USED
        return_dict=None # NOT USED
        ) -> torch.FloatTensor:
        if inputs_embeds == None:
            x = self.embedding(input_ids)  # N x L x D
        else:
            x = inputs_embeds
        if causal_mask == None:
            causal_mask = (torch.triu(torch.ones(input_ids.size(1), input_ids.size(1))) == 0).transpose(0, 1).contiguous().to(input_ids.device)
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask, attention_mask=attention_mask)
        x = self.final_norm(x)  # N x L x D

        return BaseModelOutput(
            hidden_states=x,
        )

    #Some common HF functions.
    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RITAModelForCausalLM(PreTrainedModel):
    config_class = RITAConfig
    base_model_prefix = "transformer"
    is_parallelizable = False

    def __init__(
        self,
        config
    ):
        super().__init__(config)
        self.transformer = RITAModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,  # NOT USED
        attention_mask=None,
        causal_mask=None,
        token_type_ids=None, # NOT USED
        position_ids=None, # NOT USED
        head_mask=None, # NOT USED
        inputs_embeds=None,
        encoder_hidden_states=None,  # NOT USED
        encoder_causal_mask=None, # NOT USED
        labels=None,
        use_cache=None, # NOT USED
        output_attentions=None, # NOT USED
        output_hidden_states=None, # NOT USED
        return_dict=None # NOT USED
        ) -> torch.FloatTensor:

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            causal_mask=causal_mask,
            attention_mask = attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        logits = self.lm_head(transformer_outputs.hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
        )

    #Some common HF functions.
    def get_input_embeddings(self):
        return self.transformer.embedding

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embedding = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, lm_head):
        self.lm_head = lm_head

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class RITAModelForSequenceClassification(PreTrainedModel):
    config_class = RITAConfig
    base_model_prefix = "transformer"
    is_parallelizable = False

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = RITAModel(config)
        self.score = nn.Linear(config.d_model, self.num_labels, bias=False)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        causal_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=self.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
        )
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
