import contextlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import CustomPolicy
from transformers.modeling_utils import PreTrainedModel

from progen3.common.dist import get_rank


def get_model(
    model_name_or_path: Union[str, Path],
    model_class: PreTrainedModel,
    dtype: Optional[torch.dtype] = None,
    fsdp: bool = False,
) -> Union[PreTrainedModel, FullyShardedDataParallel]:
    init_ctx = contextlib.nullcontext if get_rank() == 0 or not fsdp else init_empty_weights
    device = torch.cuda.current_device() if not fsdp else torch.device("cpu")
    with init_ctx():
        model = model_class.from_pretrained(model_name_or_path, device_map=device, torch_dtype=dtype)
    if fsdp:
        model = _fsdp_wrap(model, mixed_precision=dtype)
    return model


def _fsdp_wrap(model: PreTrainedModel, mixed_precision: Optional[torch.dtype] = None) -> FullyShardedDataParallel:
    if mixed_precision is not None:
        mp = MixedPrecision(
            param_dtype=mixed_precision,
            reduce_dtype=mixed_precision,
            buffer_dtype=mixed_precision,
        )
    else:
        mp = None
    return FullyShardedDataParallel(
        model,
        auto_wrap_policy=_auto_wrap_policy(model),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=True,
    )


def _auto_wrap_policy(obj: PreTrainedModel) -> CustomPolicy:
    def lambda_fn(module: torch.nn.Module) -> Union[bool, dict]:
        ret = False
        if hasattr(module, "_fsdp_wrap"):
            ret = bool(module._fsdp_wrap)
        elif hasattr(obj, "fsdp_wrap_fn") and callable(obj.fsdp_wrap_fn):
            ret = obj.fsdp_wrap_fn(module)
            # TODO: may need to modify a dict ret in case some values are strings when they shouldn't be
        return ret

    return CustomPolicy(lambda_fn)


# Modified from https://github.com/huggingface/accelerate/blob/main/src/accelerate/big_modeling.py
@contextmanager
def init_empty_weights(include_buffers: bool = False) -> Generator[None, None, None]:
    """Meta initialization context manager.

    A context manager under which models are initialized with all parameters
    on the meta device, therefore creating an empty model. Useful when just
    initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].

    </Tip>
    """
    with init_on_device(
        torch.device("meta"),
        include_buffers=include_buffers,
    ) as f:
        yield f


# Modified from https://github.com/huggingface/accelerate/blob/main/src/accelerate/big_modeling.py
@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False) -> Generator[None, None, None]:
    """Device initialization context manager.

    A context manager under which models are initialized with all parameters
    on the specified device.

    Args:
        device (`torch.device`): Device to initialize all parameters on.
        include_buffers (`bool`, *optional*, defaults to `False`): Whether or
            not to also put all buffers on the meta device while initializing.

    Example:
    ```python
    import torch.nn as nn

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(
        self: torch.nn.Module,
        name: str,
        param: Optional[torch.nn.Parameter],
    ) -> None:
        old_register_parameter(self, name, param)
        if param is not None:
            parameter = self._parameters[name]
            assert parameter is not None
            if isinstance(parameter, DTensor):
                self._parameters[name] = parameter.to(device)  # type: ignore
            else:
                param_cls = type(parameter)
                kwargs = parameter.__dict__
                self._parameters[name] = param_cls(
                    parameter.to(device),
                    **kwargs,
                )

    def register_empty_buffer(
        self: torch.nn.Module,
        name: str,
        tensor: Optional[torch.Tensor],
        persistent: bool = True,
    ) -> None:
        old_register_buffer(self, name, tensor, persistent=persistent)
        if tensor is not None:
            named_buffer = self._buffers[name]
            assert named_buffer is not None
            self._buffers[name] = named_buffer.to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter  # type: ignore
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer  # type: ignore
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(
                torch,
                torch_function_name,
                patch_tensor_constructor(getattr(torch, torch_function_name)),
            )
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter  # type: ignore
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer  # type: ignore
        for (
            torch_function_name,
            old_torch_function,
        ) in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)
