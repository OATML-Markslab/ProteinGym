from abc import ABC, abstractmethod
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from ..config import ProGen3Config
from .mb_wrapper import mb_build_dmoe


def promote_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.view(1) if len(x.size()) == 0 else x


class LogitConverter(ABC):
    @classmethod
    @abstractmethod
    def logits_to_probs(cls, logits: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        raise NotImplementedError


class SoftmaxMixIn(LogitConverter):
    """Converts logits to probabilities using a softmax."""

    @classmethod
    def logits_to_probs(cls, logits: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = logits.dtype if dtype is None else dtype
        return F.softmax(logits, dim=-1, dtype=dtype)


class MLP(nn.Module):
    def __init__(self, config: ProGen3Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act_fn(self.w1(hidden_states)))


class GLUMLP(nn.Module):
    def __init__(self, config: ProGen3Config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class SparseMoeBlock(nn.Module):
    """Strictly equivalent to standard MoE with full capacity (no dropped tokens)."""

    def __init__(self, config: ProGen3Config, **kwargs: dict):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.n_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.expert_selector: Type[LogitConverter] = MOE_EXPERT_SELECTION[config.moe_expert_selection]
        mlp_cls = GLUMLP if config.gated_mlp else MLP
        self.experts = nn.ModuleList([mlp_cls(config) for _ in range(self.n_experts)])
        if self.n_experts > 1:
            self.gate = nn.Linear(self.hidden_dim, self.n_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # router_logits: (batch_size * sequence_length, n_experts)
        # router_weights: (batch_size * sequence_length, n_experts)
        bsz, seqlen, dim = hidden_states.shape
        if self.n_experts == 1:
            final_hidden_states = self.experts[0](hidden_states)
            router_logits = torch.zeros(
                (bsz * seqlen, self.n_experts),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            return final_hidden_states, F.softmax(router_logits, dim=-1)

        router_logits = self.gate(hidden_states)
        routing_weights = self.expert_selector.logits_to_probs(router_logits, dtype=torch.float32)
        router_logits = router_logits.view(-1, router_logits.shape[-1])
        routing_weights = routing_weights.view(-1, routing_weights.shape[-1])
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Dense mixture of experts
        if self.n_experts == self.top_k:
            routing_weights = routing_weights.unsqueeze(1)
            final_hidden_states = torch.stack([e(hidden_states) for e in self.experts], dim=-1)
            final_hidden_states = torch.einsum("lde,lde->ld", final_hidden_states, routing_weights)
            final_hidden_states = final_hidden_states.reshape(bsz, seqlen, dim)
            return final_hidden_states, F.softmax(router_logits, dim=-1)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        hidden_states = hidden_states.view(-1, dim)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.n_experts).permute(2, 1, 0)
        final_hidden_states = torch.zeros_like(hidden_states)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.n_experts):
            expert = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = expert(hidden_states[top_x_list])
            current_state = current_state * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_state.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(bsz, seqlen, dim)
        return final_hidden_states, F.softmax(router_logits, dim=-1)


MOE_CLASSES = {
    "eager": SparseMoeBlock,
    "megablocks": mb_build_dmoe,
}
MOE_EXPERT_SELECTION = {"switch": SoftmaxMixIn}
