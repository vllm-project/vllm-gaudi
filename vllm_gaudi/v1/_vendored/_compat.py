# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Helpers removed from still-existing vLLM modules in aac1b2a4f3, vllm-project/vllm#63373.
# Local changes: copied verbatim.
"""V1 helpers that upstream vLLM removed together with Model Runner V1."""

from typing import TypeVar

import torch

from vllm.utils.torch_utils import TORCH_DTYPE_TO_NUMPY_DTYPE, make_ndarray_with_pad

_K = TypeVar("_K")
_V = TypeVar("_V")
T = TypeVar("T")


# Vendored from https://github.com/vllm-project/vllm/blob/b46907cce0/vllm/utils/collection_utils.py
def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
    """Swap values between two keys."""
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)


# Vendored from https://github.com/vllm-project/vllm/blob/b46907cce0/vllm/v1/utils.py
def copy_slice(
    from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int
) -> torch.Tensor:
    """Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


# Vendored from https://github.com/vllm-project/vllm/blob/b46907cce0/vllm/utils/torch_utils.py
def make_tensor_with_pad(
    x: list[list[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: int | None = None,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor
