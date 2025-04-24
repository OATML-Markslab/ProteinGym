import os
from typing import Any, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_utils import PreTrainedModel


def get_world_size(group: Any = None) -> int:
    if os.environ.get("RANK", -1) == -1 or not dist.is_initialized():
        return 1
    return dist.get_world_size(group=group)


def get_rank(group: Any = None) -> int:
    if os.environ.get("RANK", -1) == -1 or not dist.is_initialized():
        return 0
    return dist.get_rank(group=group)


def get_device() -> int:
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return torch.device("cpu")


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0)) if dist.is_initialized() else 0


def setup_dist() -> None:
    rank = int(os.environ.get("RANK", -1))
    if dist.is_available() and torch.cuda.is_available() and rank != -1:
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def is_initialized() -> bool:
    return dist.is_initialized()


@torch.no_grad()
def generate(
    model: Union[FullyShardedDataParallel, PreTrainedModel], *args: Any, **kwargs: Any
) -> Union[GenerateOutput, torch.LongTensor]:
    if any(isinstance(m, FullyShardedDataParallel) for m in [model, *model.named_children()]):
        kwargs["synced_gpus"] = True
        with FullyShardedDataParallel.summon_full_params(model, writeback=False, recurse=False):
            return model.generate(*args, **kwargs)
    return model.generate(*args, **kwargs)
