# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference implementation for the causal conv1d kernels.

This module mirrors the public APIs in ``causal_conv1d.py`` but executes with
standard PyTorch tensor ops. The implementation favors readability and
correctness which makes it suitable for testing and CPU execution. It does not
implement Triton-specific optimizations such as the advanced block-level
prefix-caching metadata. When those arguments are supplied a
``NotImplementedError`` is raised to surface the limitation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
# import habana_frameworks.torch.hpu as ht

from vllm.v1.attention.backends.utils import PAD_SLOT_ID


@dataclass(frozen=True)
class _ReshapeSpec:
    """Stores how to reshape flattened continuous-batch tensors back."""

    reshape_fn: Callable[[torch.Tensor], torch.Tensor]
    description: str


def _normalize_activation(activation: bool | str | None) -> str | None:
    if isinstance(activation, bool):
        return "silu" if activation else None
    if activation is None:
        return None
    activation = activation.lower()
    if activation not in {"silu", "swish"}:
        raise ValueError(f"Unsupported activation '{activation}'.")
    return activation


def _ensure_query_start_loc(query_start_loc: torch.Tensor) -> torch.Tensor:
    if query_start_loc is None:
        raise ValueError("'query_start_loc' must be provided for the PyTorch reference implementation.")
    if query_start_loc.dim() != 1:
        raise ValueError("'query_start_loc' must be 1-D.")
    return query_start_loc.to(dtype=torch.int64)


def _make_depthwise_weight(weight: torch.Tensor) -> torch.Tensor:
    dim, width = weight.shape
    return weight.contiguous().view(dim, 1, width)


def _apply_activation(output: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation in {"silu", "swish"}:
        return torch.nn.functional.silu(output)
    return output


def _flatten_inputs_for_update(
    x: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor, _ReshapeSpec]:
    if query_start_loc is None:
        if x.dim() == 2:
            x_3d = x.unsqueeze(-1)
            squeeze_last = True
        elif x.dim() == 3:
            x_3d = x
            squeeze_last = False
        else:
            raise ValueError("When 'query_start_loc' is None, 'x' must be 2-D or 3-D.")
        if x_3d.size(1) != dim:
            raise ValueError("Dimension mismatch between 'x' and 'weight'.")
        batch, _, seqlen = x_3d.shape
        flat = x_3d.permute(1, 0, 2).contiguous().view(dim, batch * seqlen)
        # Create qsl on CPU to avoid CUDA graph capture issues
        qsl = torch.arange(
            0,
            (batch + 1) * seqlen,
            seqlen,
            device=torch.device(x.device),
            dtype=torch.int64,
        )

        def reshape_fn(out: torch.Tensor) -> torch.Tensor:
            restored = out.view(dim, batch, seqlen).permute(1, 0, 2)
            return restored.squeeze(-1) if squeeze_last else restored

        return flat, qsl, _ReshapeSpec(reshape_fn, "batched")

    # query_start_loc provided -> assume x already flattened (dim, cu_seqlen) or (cu_seqlen, dim)
    if x.dim() != 2:
        raise ValueError("Expected 2-D 'x' when 'query_start_loc' is provided.")
    if x.size(0) == dim:
        flat = x

        def reshape_fn(out: torch.Tensor) -> torch.Tensor:
            return out

        qsl = _ensure_query_start_loc(query_start_loc)
        assert qsl is not None
        return flat, qsl, _ReshapeSpec(reshape_fn, "channel-first")

    if x.size(1) == dim:
        flat = x.unsqueeze(2)  # transpose(0, 1).contiguous()

        def reshape_fn(out: torch.Tensor) -> torch.Tensor:
            return out.squeeze(2)  # transpose(0, 1).contiguous()

        qsl = _ensure_query_start_loc(query_start_loc)
        assert qsl is not None
        return flat, qsl, _ReshapeSpec(reshape_fn, "token-first")

    raise ValueError("Could not infer how to flatten 'x' for the provided dimensions.")


def hpu_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
    is_prompt: bool = True,
):
    if any(ptr is not None for ptr in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
    )):
        raise NotImplementedError("Prefix caching metadata is not supported in the PyTorch reference implementation.")

    activation = _normalize_activation(activation)
    original_dtype = x.dtype
    work_dtype = conv_states.dtype if conv_states is not None else x.dtype
    x_work = x.to(work_dtype)
    weight_work = weight.to(work_dtype)
    bias_work = bias.to(work_dtype) if bias is not None else None

    assert conv_states is not None
    if conv_states.device != x_work.device:
        raise ValueError("'conv_states' must reside on the same device as 'x'.")

    # GPU-optimized: Keep all tensors on GPU, no CPU transfers
    # Don't use .to('cuda') during graph capture - use the device from x_work
    qsl = _ensure_query_start_loc(query_start_loc)
    assert qsl is not None

    # Keep on GPU - compute sequence info using tensor operations
    padded_batch = qsl.numel() - 1
    if padded_batch != 1:
        raise ValueError(f"'padded_batch' must be 1 but we get {padded_batch}")
    dim, cu_seqlen = x_work.shape
    _, width = weight_work.shape
    state_len = max(width - 1, 0)

    if validate_data:
        if x_work.dim() != 2:
            raise ValueError("'x' must be 2-D (dim, cu_seq_len).")
        if weight_work.shape != (dim, width):
            raise ValueError("'weight' must have shape (dim, width).")
        if bias_work is not None and bias_work.shape != (dim, ):
            raise ValueError("'bias' must match the feature dimension.")
        if not ((x_work.stride(0) == 1) or (x_work.stride(1) == 1)):
            raise ValueError("Input tensor must be in channel-last or channel-first memory layout.")
        if cache_indices is not None and cache_indices.numel() != padded_batch:
            raise ValueError("'cache_indices' must align with the batch dimension implied by 'query_start_loc'.")
        if has_initial_state is not None and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

    weight_dw = _make_depthwise_weight(weight_work)
    out = torch.zeros_like(x_work)

    # Get cache indices
    if cache_indices is None:
        batch_cache_idx = torch.arange(padded_batch, device=x_work.device, dtype=torch.long)
    else:
        # Ensure cache_indices is on the correct device
        batch_cache_idx = cache_indices.to(x_work.device) if cache_indices.device != x_work.device else cache_indices

    # Take all input data for this call
    # Create tensor to get all data from 0 to lest sequence
    # This works bor padded_batch equal 1
    # ss = torch.arange(seq_starts[0], seq_ends[-1])
    seq_x = x_work[:, qsl[0]:qsl[-1]]

    # Get init_state for all batch
    # if  is_prompt:
    #     init_state = torch.zeros(padded_batch, dim, state_len, device=x_work.device, dtype=work_dtype)
    #     init_state = init_state.squeeze()
    # else:
    #     init_state = conv_states[batch_cache_idx, :, -state_len:]
    #     init_state = init_state.squeeze()
    init_state = torch.where(torch.tensor([is_prompt], device=x_work.device),
                             torch.zeros(padded_batch, dim, state_len, device=x_work.device, dtype=work_dtype),
                             conv_states[batch_cache_idx, :, -state_len:])
    init_state = init_state.squeeze()

    # Prepare input for convolution
    seq_input = torch.cat([init_state, seq_x], dim=1)
    new_state = seq_input[:, -state_len:]

    # Apply convolution
    seq_input = seq_input.unsqueeze(0)
    seq_out = F.conv1d(seq_input, weight_dw, bias=bias_work, groups=dim)
    seq_out = _apply_activation(seq_out, activation)
    # out[:, ss] = seq_out.squeeze(0)
    out[:, qsl[0]:qsl[-1]] = seq_out.squeeze(0)

    # Update conv state
    # Update cache with the latest state_len tokens for this sequence
    with torch.no_grad():
        conv_states[batch_cache_idx, :, -state_len:] = conv_states[batch_cache_idx, :, -state_len:].copy_(new_state)

    return out.to(original_dtype)


def hpu_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
):
    if num_accepted_tokens is not None:
        raise NotImplementedError("Speculative decoding updates are not supported in the reference implementation.")
    if block_idx_last_scheduled_token is not None or initial_state_idx is not None:
        raise NotImplementedError("Prefix caching metadata is not supported in the reference implementation.")
    if max_query_len not in (-1, None):  # Provided only for Triton helper parity
        raise NotImplementedError("'max_query_len' is not used in the reference implementation.")

    activation = _normalize_activation(activation)
    dim = weight.size(0)

    flat_x, qsl, reshape_spec = _flatten_inputs_for_update(x, query_start_loc, dim)

    result = hpu_causal_conv1d_fn_update(
        flat_x,
        weight,
        bias,
        conv_state,
        qsl,
        cache_indices=conv_state_indices,
        has_initial_state=None,
        activation=activation,
        metadata=None,
        validate_data=validate_data,
        is_prompt=False,
    )

    return reshape_spec.reshape_fn(result)


def hpu_causal_conv1d_fn_update(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    metadata=None,
    validate_data: bool = False,
    is_prompt: bool = True,
):
    if any(ptr is not None for ptr in (
            block_idx_first_scheduled_token,
            block_idx_last_scheduled_token,
            initial_state_idx,
            num_computed_tokens,
    )):
        raise NotImplementedError("Prefix caching metadata is not supported in the PyTorch reference implementation.")

    activation = _normalize_activation(activation)
    original_dtype = x.dtype
    work_dtype = conv_states.dtype if conv_states is not None else x.dtype
    x_work = x.to(work_dtype)
    weight_work = weight.to(work_dtype)
    bias_work = bias.to(work_dtype) if bias is not None else None

    assert conv_states is not None
    if conv_states.device != x_work.device:
        raise ValueError("'conv_states' must reside on the same device as 'x'.")

    # GPU-optimized: Keep all tensors on GPU, no CPU transfers
    # Don't use .to('cuda') during graph capture - use the device from x_work
    qsl = _ensure_query_start_loc(query_start_loc)
    assert qsl is not None

    # Keep on GPU - compute sequence info using tensor operations
    padded_batch = qsl.numel() - 1
    _, dim, cu_seqlen = x_work.shape
    _, width = weight_work.shape
    state_len = max(width - 1, 0)

    if validate_data:
        if x_work.dim() != 2:
            raise ValueError("'x' must be 2-D (dim, cu_seq_len).")
        if weight_work.shape != (dim, width):
            raise ValueError("'weight' must have shape (dim, width).")
        if bias_work is not None and bias_work.shape != (dim, ):
            raise ValueError("'bias' must match the feature dimension.")
        if not ((x_work.stride(0) == 1) or (x_work.stride(1) == 1)):
            raise ValueError("Input tensor must be in channel-last or channel-first memory layout.")
        if cache_indices is not None and cache_indices.numel() != padded_batch:
            raise ValueError("'cache_indices' must align with the batch dimension implied by 'query_start_loc'.")
        if has_initial_state is not None and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

    weight_dw = _make_depthwise_weight(weight_work)
    out = torch.zeros_like(x_work)

    # Get cache indices
    if cache_indices is None:
        batch_cache_idx = torch.arange(padded_batch, device=x_work.device, dtype=torch.long)
    else:
        # Ensure cache_indices is on the correct device
        batch_cache_idx = cache_indices.to(x_work.device) if cache_indices.device != x_work.device else cache_indices

    if is_prompt:
        init_state = torch.zeros(padded_batch, dim, state_len, device=x_work.device, dtype=work_dtype)
    else:
        init_state = conv_states[batch_cache_idx, :, -state_len:]

    seq_input = torch.cat([init_state, x_work], dim=2)
    new_state = seq_input[:, :, -state_len:]
    seq_out = F.conv1d(seq_input, weight_dw, bias, groups=dim)
    seq_out = _apply_activation(seq_out, activation)
    out = seq_out

    with torch.no_grad():
        conv_states[batch_cache_idx, :, -state_len:] = conv_states[batch_cache_idx, :, -state_len:].copy_(new_state)

    return out.to(original_dtype)
