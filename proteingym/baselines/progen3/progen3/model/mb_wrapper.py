# mypy: ignore-errors
# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Adapted from LLM foundry."""
import functools
import logging

import megablocks
import megablocks.layers.arguments
import megablocks.layers.common
import megablocks.layers.dmoe
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Shard
from torch.distributed.tensor.device_mesh import init_device_mesh
from transformers.activations import ACT2FN

from ..config import ProGen3Config

log = logging.getLogger(__name__)

__all__ = [
    "mb_build_dmoe",
    "mb_setup_args",
]

functional_ACT2FN = {**ACT2FN}
functional_ACT2FN["gelu"] = torch.nn.functional.gelu
functional_ACT2FN["silu"] = torch.nn.functional.silu


def dtensorify_param(
    param: nn.Parameter,
    mesh: DeviceMesh,
    placements: list[Placement],
):
    """Construct a DTensor from an already sharded local parameter."""
    param_dtensor = DTensor.from_local(
        param.data,
        device_mesh=mesh,
        placements=placements,
        run_check=False,
    )
    return nn.Parameter(param_dtensor)


def get_mb_device_mesh(config: ProGen3Config) -> DeviceMesh:
    """Helper function to get the device mesh for MegaBlocks MoE.

    Args:
        moe_world_size (int): The MoE world size.
        world_size (int): The world size.

    Raises:
        ValueError: If the device mesh configuration is not valid.

    Returns:
        The device mesh for MegaBlocks MoE.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert world_size >= config.moe_world_size and world_size % config.moe_world_size == 0
    return init_device_mesh(
        "cuda",
        (world_size // config.moe_world_size, config.moe_world_size),
        mesh_dim_names=("weight_parallel", "expert_parallel"),
    )


def mb_setup_args(
    config: ProGen3Config,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> tuple[megablocks.layers.arguments.Arguments, DeviceMesh]:
    """Setup the MegaBlocks args.

    Args:
        config (MixtralConfig): The model config object.
        device (Optional[str]): The device to run the FFN on.

    Returns:
        tuple[megablocks.layers.arguments.Arguments, DeviceMesh]:
            The MegaBlocks args and the device mesh for FSDP/expert parallelism.
    """
    # Configure device mesh for expert parallelism if desired
    device_mesh = None
    if config.moe_world_size > 1:
        device_mesh = get_mb_device_mesh(config)
        kwargs.update(
            moe_expert_model_parallelism=True,
            expert_parallel_group=device_mesh["expert_parallel"].get_group(0),
        )

    args = megablocks.layers.arguments.Arguments(
        hidden_size=config.hidden_size,
        ffn_hidden_size=config.intermediate_size,
        num_layers=config.num_hidden_layers,
        bias=False,
        return_bias=False,
        activation_fn=functional_ACT2FN[config.hidden_act],
        moe_num_experts=config.num_experts,
        moe_top_k=config.num_experts_per_tok,
        moe_loss_weight=config.router_aux_loss_coef,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        device=device,
        mlp_type="glu" if config.gated_mlp else "mlp",
        mlp_impl="grouped" if config.moe_grouped_gemm else "sparse",
        memory_optimized_mlp=config.moe_memory_optimized,
        moe_normalize_expert_weights=1,
        init_method=functools.partial(torch.nn.init.normal_, mean=0.0, std=config.initializer_range),
        **kwargs,
    )

    return args, device_mesh


def attach_ffn_mb_args(
    ffn: megablocks.layers.dmoe.dMoE,
    args: megablocks.layers.arguments.Arguments,
):
    """Attach arguments used in parameter initialization to the FFN.

    Args:
        ffn (nn.Module): The FFN module.
        args (megablocks.layers.arguments.Arguments): The arguments for MegaBlocks.
    """
    ffn.experts.mlp.hidden_size = args.ffn_hidden_size
    ffn.experts.mlp.expert_parallel_group = args.expert_parallel_group


def set_ffn_device_mesh(
    ffn: nn.Module,
    moe_world_size: int,
    device_mesh: DeviceMesh,
):
    """Sets the device mesh in FSDP kwargs.

    Args:
        ffn (nn.Module): The FFN module.
        moe_world_size (int): The MoE world size.
        device_mesh (DeviceMesh): The full device mesh.

    Raises:
        RuntimeError: If the device mesh is 3D.
        ValueError: If the device mesh is not 2D or 3D.
    """
    if moe_world_size > 1 and device_mesh is not None:
        expert_mesh = device_mesh["expert_parallel"]
        expert_placements: list[Placement] = [Shard(dim=0)]
        # Register in two loops as you cannot overwrite parameters while iterating over named_parameters()
        dtensorified_params = [
            (
                name,
                dtensorify_param(
                    param=parameter,
                    mesh=expert_mesh,
                    placements=expert_placements,
                ),
            )
            for name, parameter in ffn.experts.mlp.named_parameters()
        ]
        for name, dtensorified_param in dtensorified_params:
            ffn.experts.mlp.register_parameter(name, dtensorified_param)

        ffn.experts._fsdp_kwargs_dict = {"device_mesh": device_mesh["weight_parallel"]}


def mb_build_dmoe(
    config: ProGen3Config,
    args: megablocks.layers.arguments.Arguments,
    device_mesh: DeviceMesh | None = None,
    **kwargs,
) -> megablocks.layers.dmoe.dMoE:
    ffn = megablocks.layers.dmoe.dMoE(args)
    attach_ffn_mb_args(
        ffn=ffn,
        args=args,
    )
    set_ffn_device_mesh(
        ffn=ffn,
        moe_world_size=config.moe_world_size,
        device_mesh=device_mesh,
    )
    return ffn
