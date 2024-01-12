from dataclasses import dataclass
from typing import Optional, Tuple
import math
import os
import pandas as pd
import uuid
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss
import torch.nn.functional as F
from transformers import GPT2PreTrainedModel

from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from tranception.activations import tranception_ACT2FN
from tranception.config import TranceptionConfig
from tranception.outputs import (
    TranceptionCausalLMOutputWithCrossAttentions,
)
from tranception.utils import msa_utils
from tranception.utils import scoring_utils

def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def get_slopes(n, mode="standard_alibi", verbose=False):
    """
    Function to compute the m constant for each attention head. Code has been adapted from the official ALiBi codebase at:
    https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py
    """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]
    if mode=="grouped_alibi":
        n = n // 4
    if math.log2(n).is_integer():
        result = get_slopes_power_of_2(n)                   
    else:
        #Workaround when the number of heads is not a power of 2
        closest_power_of_2 = 2**math.floor(math.log2(n))  
        result = get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    if mode=="grouped_alibi":
        result = result * 4
        if verbose:
            print("ALiBi slopes: {}".format(result))
    return result

class SpatialDepthWiseConvolution(nn.Module):
    def __init__(self, head_dim: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=head_dim, out_channels=head_dim, kernel_size=(kernel_size,), padding=(kernel_size - 1,), groups=head_dim)
    
    def forward(self, x: torch.Tensor):
        batch_size, heads, seq_len, head_dim = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * heads, head_dim, seq_len)
        x = self.conv(x)
        if self.kernel_size>1:
            x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, head_dim, seq_len)
        x = x.permute(0, 1, 3, 2)
        return x

class TranceptionBlockAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, SDWC_kernel_size=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

        self.attention_mode=config.attention_mode
        
        if self.attention_mode=="tranception":
            assert self.num_heads%4==0, "Invalid number of heads. Tranception requires the number of heads to be a multiple of 4."
            self.num_heads_per_kernel_size = self.num_heads // 4
            self.query_depthwiseconv = nn.ModuleDict()
            self.key_depthwiseconv = nn.ModuleDict()
            self.value_depthwiseconv = nn.ModuleDict()
            for kernel_idx, kernel in enumerate([3,5,7]):
                self.query_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(self.head_dim,kernel)
                self.key_depthwiseconv[str(kernel_idx)]   = SpatialDepthWiseConvolution(self.head_dim,kernel)
                self.value_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(self.head_dim,kernel)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None, alibi_bias=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if alibi_bias is not None:
            attn_weights = attn_weights + alibi_bias[:,:,:attn_weights.size(-1)]

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi_bias=None,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        if self.attention_mode=="tranception":
            # We do not do anything on the first self.num_heads_per_kernel_size heads (kernel =1)
            query_list=[query[:,:self.num_heads_per_kernel_size,:,:]]
            key_list=[key[:,:self.num_heads_per_kernel_size,:,:]]
            value_list=[value[:,:self.num_heads_per_kernel_size,:,:]]
            for kernel_idx in range(3):
                query_list.append(self.query_depthwiseconv[str(kernel_idx)](query[:,(kernel_idx+1)*self.num_heads_per_kernel_size:(kernel_idx+2)*self.num_heads_per_kernel_size,:,:]))
                key_list.append(self.key_depthwiseconv[str(kernel_idx)](key[:,(kernel_idx+1)*self.num_heads_per_kernel_size:(kernel_idx+2)*self.num_heads_per_kernel_size,:,:]))
                value_list.append(self.value_depthwiseconv[str(kernel_idx)](value[:,(kernel_idx+1)*self.num_heads_per_kernel_size:(kernel_idx+2)*self.num_heads_per_kernel_size,:,:]))
            query=torch.cat(query_list, dim=1)
            key=torch.cat(key_list, dim=1)
            value=torch.cat(value_list, dim=1)
        
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask, alibi_bias=alibi_bias)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class TranceptionBlockMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = tranception_ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TranceptionBlock(nn.Module):
    def __init__(self, config, SDWC_kernel_size=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TranceptionBlockAttention(config, SDWC_kernel_size=SDWC_kernel_size)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = TranceptionBlockAttention(config, is_cross_attention=True, SDWC_kernel_size=SDWC_kernel_size)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = TranceptionBlockMLP(inner_dim, config)
    
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi_bias=None,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            alibi_bias=alibi_bias,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        feed_forward_hidden_states = self.mlp(hidden_states)
        
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

class TranceptionModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.position_embedding = config.position_embedding if hasattr(config, "position_embedding") else "learned"
        if self.position_embedding=="learned":
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
            self.alibi = None
        elif self.position_embedding=="grouped_alibi":
            maxpos = config.n_positions
            attn_heads = config.n_head
            self.slopes = torch.Tensor(get_slopes(attn_heads, mode=self.position_embedding))
            #The softmax operation is invariant to translation, and bias functions used are always linear. 
            alibi = self.slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(0).unsqueeze(0).expand(attn_heads, -1, -1)
            alibi = alibi.view(attn_heads, 1, maxpos)
            self.register_buffer('alibi',alibi)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([TranceptionBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
    
    def parallelize(self, device_map=None, num_cores=None):
        self.device_map = (
                get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
            )
        device_prefix="cuda:"
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else device_prefix + str(min(self.device_map.keys()))
        self.last_device = device_prefix + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        if self.position_embedding=="learned":
            self.wpe = self.wpe.to(self.first_device)
        for k, v in self.device_map.items():
            print("k,v :"+str(k)+","+str(v))
            for block in v:
                cuda_device = device_prefix + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        self.ln_f = self.ln_f.to(self.last_device)
    
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        if self.position_embedding=="learned":
            self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        if self.position_embedding=="learned":
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi_bias=self.alibi if hasattr(self, "alibi") else None
                )

            hidden_states = outputs[0]
            
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            if self.model_parallel:
                device_prefix="cuda:"
                for k, v in self.device_map.items():
                    if i == v[-1] and device_prefix + str(k) != self.last_device:
                        hidden_states = hidden_states.to(device_prefix + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        
        return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )

class TranceptionLMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.transformer = TranceptionModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.config = config

        self.init_weights()
        
        self.default_model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.clustal_hash = str(uuid.uuid4()) 
        self.retrieval_aggregation_mode = config.retrieval_aggregation_mode if hasattr(config, "retrieval_aggregation_mode") else None
        if self.retrieval_aggregation_mode is not None:
            print("Model leverages both autoregressive and retrieval inference")
            self.MSA_filename = config.MSA_filename if hasattr(config, "MSA_filename") else False
            self.MSA_folder = '/'.join(self.MSA_filename.split(os.sep)[:-1])
            self.MSA_name = self.MSA_filename.split(os.sep)[-1]
            self.retrieval_inference_weight_LR = config.retrieval_inference_weight if hasattr(config, "retrieval_inference_weight") else 0.6
            self.retrieval_inference_weight_RL = config.retrieval_inference_weight if hasattr(config, "retrieval_inference_weight") else 0.6
            self.MSA_start=config.MSA_start
            self.MSA_end=config.MSA_end
            self.full_protein_length = config.full_protein_length if hasattr(config, "full_protein_length") else -1
            
            self.MSA_log_prior = torch.log(torch.tensor(
                                                        msa_utils.get_msa_prior(
                                                            MSA_data_file=self.MSA_filename, 
                                                            MSA_weight_file_name=config.MSA_weight_file_name, 
                                                            retrieval_aggregation_mode=self.retrieval_aggregation_mode,
                                                            MSA_start=self.MSA_start,
                                                            MSA_end=self.MSA_end,
                                                            len_target_seq=self.full_protein_length, 
                                                            vocab=config.tokenizer.get_vocab(), 
                                                            verbose=False
                                                        )
                                            ).float().to(self.default_model_device))
        else:
            print("Model only uses autoregressive inference")

    def parallelize(self, device_map=None, num_cores=None, num_pipelines=1):
        self.num_pipelines=num_pipelines
        self.device_map = (
                get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
                if device_map is None
                else device_map
            )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map, num_cores=num_cores)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        
        return {
                "input_ids": input_ids,
                "past_key_values": past,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "flip": kwargs.get("flip", None),
            }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        flip=None,
        start_slice=None,
        end_slice=None,
        mutated_sequence=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = transformer_outputs[0]
            
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
            self.MSA_log_prior = self.MSA_log_prior.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        fused_shift_log_probas = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            if self.retrieval_aggregation_mode is not None:
                batch_size = input_ids.size(0)
                if self.retrieval_aggregation_mode=="aggregate_indel":
                    assert batch_size==1, "Aggregate indel is only supported for batch size of 1"
                    truncated_sequence_text = mutated_sequence[0][start_slice[0]:end_slice[0]]
                    if len(truncated_sequence_text)!=shift_logits.shape[1]-1: # shift_logits only has one extra token compared to truncated_sequence_text (the BOS token)
                        print("Tokenization error -- seq length: {} and shift_logits length - 1 : {}".format(len(mutated_sequence),shift_logits.shape[1]-1))
                    MSA_log_prior, MSA_start, MSA_end = msa_utils.update_retrieved_MSA_log_prior_indel(self, self.MSA_log_prior, self.MSA_start, self.MSA_end, mutated_sequence[0], self.clustal_hash) 
                
                elif self.retrieval_aggregation_mode=="aggregate_substitution":
                    MSA_log_prior=self.MSA_log_prior
                    MSA_start=self.MSA_start
                    MSA_end=self.MSA_end
                
                shift_log_probas = torch.log_softmax(shift_logits,dim=-1)
                fused_shift_log_probas = shift_log_probas.clone()
                if flip is None:
                    flip = torch.zeros(batch_size).to(fused_shift_log_probas.device)
                flip = flip > 0
                
                for seq_index in range(batch_size):
                    min_prior_slice = max(start_slice[seq_index], MSA_start) 
                    max_prior_slice = min(end_slice[seq_index], MSA_end)
                    
                    # This will cascade into an error
                    if max_prior_slice <= min_prior_slice:
                        print("Non overlapping region detected: min_prior_slice {} and max_prior_slice {}".format(min_prior_slice,max_prior_slice))
                        continue
                    
                    slice_prior = MSA_log_prior[min_prior_slice:max_prior_slice,:].to(fused_shift_log_probas.device) 
                    if flip[seq_index]:
                        slice_prior = torch.flip(slice_prior,dims=(0,))
                        min_logits_slice = max(0,end_slice[seq_index]-MSA_end) 
                        max_logits_slice = min_logits_slice + (max_prior_slice-min_prior_slice)
                        fused_shift_log_probas[seq_index,min_logits_slice:max_logits_slice,:] = (1-self.retrieval_inference_weight_RL)*shift_log_probas[seq_index,min_logits_slice:max_logits_slice,:] + self.retrieval_inference_weight_RL*slice_prior
                    else:
                        min_logits_slice = max(0, MSA_start-start_slice[seq_index]) 
                        max_logits_slice = min_logits_slice + (max_prior_slice-min_prior_slice)
                        fused_shift_log_probas[seq_index,min_logits_slice:max_logits_slice,:] = (1-self.retrieval_inference_weight_LR)*shift_log_probas[seq_index,min_logits_slice:max_logits_slice,:] + self.retrieval_inference_weight_LR*slice_prior
                
                if self.retrieval_aggregation_mode=="aggregate_indel":
                    try:
                        # If a given residue colume is an added zero-column, then we overwrite prior fusion and only predict based on the autoregressive transformer inference mode.
                        inserted_retrieval_positions = [True if slice_prior[i].sum()==0 else False for i in range(len(slice_prior))]+[True] #Last True is for the end of sentence token
                        fused_shift_log_probas[:,inserted_retrieval_positions,:]=shift_log_probas[:,inserted_retrieval_positions,:]
                    except Exception as e:
                        print("Error when adding zero column(s) to account for insertion mutations.")
                        raise e
                
                loss_fct = NLLLoss(reduction='none')
                loss = loss_fct(input=fused_shift_log_probas.view(-1, fused_shift_log_probas.size(-1)), target=shift_labels.view(-1)).view(fused_shift_log_probas.shape[0],fused_shift_log_probas.shape[1])
                mask = attention_mask[..., 1:].float()
                mask[mask==0]=float('nan')
                loss *= mask
                loss = nanmean(loss, dim=1).mean()
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:] 
            return ((loss,) + output) if loss is not None else output
        
        return TranceptionCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            fused_shift_log_probas=fused_shift_log_probas
        )


    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
    
    def score_mutants(self, DMS_data, target_seq=None, scoring_mirror=True, batch_size_inference=10, num_workers=10, indel_mode=False):
        """
        Method to score mutants in an input DMS file.
        DMS_data: (dataframe) Dataframe containing the list of mutated sequences for scoring.
        target_seq: (string) Full reference sequence (wild type) that is mutated in the DMS assay. If not None, returned scores are delta log likelihood wrt that sequence.
        scoring_mirror: (bool) Whether to score mutated sequences from both directions (Left->Right and Right->Left).
        batch_size_inference: (int) Batch size for scoring.
        num_workers: (int) Number of workers to be used in the data loader.
        indel_mode: (bool) Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions.
        """
        df = DMS_data.copy()
        if ('mutated_sequence' not in df) and (not indel_mode): df['mutated_sequence'] = df['mutant'].apply(lambda x: scoring_utils.get_mutated_sequence(target_seq, x))
        
        # Daniel R: Temporary since indel files all have a 'mutant' column containing the mutated sequence. Should probably
        # make all of those mutated sequence columns and take this out entirely 
        # if indel_mode:
        #     df["mutated_sequence"] = df["mutant"]
        
        assert ('mutated_sequence' in df), "DMS file to score does not have mutated_sequence column"
        if 'mutant' not in df: df['mutant'] = df['mutated_sequence'] #if mutant not in DMS file we default to mutated_sequence
        #if 'DMS_score' in df: del df['DMS_score'] 
        #if 'DMS_score_bin' in df: del df['DMS_score_bin'] 
        df = df[['mutated_sequence','mutant']]
        if target_seq is not None:
            df_left_to_right_slices = scoring_utils.get_sequence_slices(df, target_seq=target_seq, model_context_len = self.config.n_ctx - 2, indel_mode=indel_mode, scoring_window=self.config.scoring_window)
        else:
            df_left_to_right_slices = scoring_utils.get_sequence_slices(df, target_seq=list(df['mutated_sequence'])[0], model_context_len = self.config.n_ctx - 2, indel_mode=indel_mode, scoring_window='sliding')
        print("Scoring sequences from left to right")
        scores_L_to_R = scoring_utils.get_tranception_scores_mutated_sequences(model=self, mutated_sequence_df=df_left_to_right_slices, batch_size_inference=batch_size_inference, score_var_name='avg_score_L_to_R', target_seq=target_seq, num_workers=num_workers, indel_mode=indel_mode)
        if scoring_mirror: 
            print("Scoring sequences from right to left")
            df_right_to_left_slices = df_left_to_right_slices.copy()
            df_right_to_left_slices['sliced_mutated_sequence'] = df_right_to_left_slices['sliced_mutated_sequence'].apply(lambda x: x[::-1])
            scores_R_to_L = scoring_utils.get_tranception_scores_mutated_sequences(model=self, mutated_sequence_df=df_right_to_left_slices, batch_size_inference=batch_size_inference, score_var_name='avg_score_R_to_L', target_seq=target_seq, num_workers=num_workers, reverse=True, indel_mode=indel_mode)
            all_scores = pd.merge(scores_L_to_R, scores_R_to_L, on='mutated_sequence', how='left', suffixes=('','_R_to_L'))
            all_scores['avg_score'] = (all_scores['avg_score_L_to_R'] + all_scores['avg_score_R_to_L']) / 2.0
        else:
            all_scores = scores_L_to_R
            all_scores['avg_score'] = all_scores['avg_score_L_to_R']
        #By design "get_tranception_scores_mutated_sequences" drops the WT from the output. We add it back if that was one of the sequences to score in the DMS (score=0 by definition)
        if indel_mode:
            mutant_column = "mutant"
        else:
            mutant_column = "mutated_sequence"
        if target_seq in DMS_data[mutant_column].values:
            if scoring_mirror:
                wt_row = pd.DataFrame([[target_seq,0,0,0]], columns=[mutant_column,'avg_score_L_to_R','avg_score_R_to_L','avg_score'])
            else:
                wt_row = pd.DataFrame([[target_seq,0,0]], columns=[mutant_column,'avg_score_L_to_R','avg_score'])
            all_scores = pd.concat([all_scores,wt_row], ignore_index=True)
        return all_scores

    def encode_batch(self, protein_sequence, sequence_name="sliced_mutated_sequence"):
        """
        Method to process an input AA sequence batch (protein_sequence) and return a tokenized sequence (via the tokenizer associated to the model).
        """
        protein_sequence[sequence_name] = scoring_utils.sequence_replace(sequences=protein_sequence[sequence_name], char_to_replace='X', char_replacements='ACDEFGHIKLMNPQRSTVWY')
        protein_sequence[sequence_name] = scoring_utils.sequence_replace(sequences=protein_sequence[sequence_name], char_to_replace='B', char_replacements='DN')
        protein_sequence[sequence_name] = scoring_utils.sequence_replace(sequences=protein_sequence[sequence_name], char_to_replace='J', char_replacements='IL')
        protein_sequence[sequence_name] = scoring_utils.sequence_replace(sequences=protein_sequence[sequence_name], char_to_replace='Z', char_replacements='EQ')
        return self.config.tokenizer(list(protein_sequence[sequence_name]), add_special_tokens=True, truncation=True, padding=True, max_length=self.config.n_ctx)

