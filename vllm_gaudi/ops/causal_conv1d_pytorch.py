# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
"""PyTorch reference implementation for the causal conv1d kernels.

This module mirrors the public APIs in:
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
but executes with standard PyTorch tensor ops. The implementation favors
readability and correctness which makes it suitable for testing and CPU
execution.  It does not implement Triton-specific optimizations such as the
advanced block-level prefix-caching metadata. When those arguments are
supplied a ``NotImplementedError`` is raised to surface the limitation
explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
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


def _apply_activation(output: torch.Tensor, activation: str | None) -> torch.Tensor:
    if activation in {"silu", "swish"}:
        return torch.nn.functional.silu(output)
    return output


def _depthwise_conv1d_tpc(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise 1-D convolution using element-wise TPC ops only.

    Equivalent to::

        F.conv1d(x, weight.unsqueeze(1), bias, groups=x.shape[1])

    For the small kernel widths used by Mamba models (typically 4) this
    avoids dispatching an MME ``spatial_convolution`` whose ``input1``
    weight-transpose creates a TPC stall that prevents TPC/MME
    pipelining on Gaudi.
    """
    # x:      (batch, dim, L)
    # weight: (dim, width)
    width = weight.shape[1]
    if x.shape[2] < width:
        raise ValueError(f"Input length ({x.shape[2]}) is smaller than kernel width"
                         f" ({width}). Convolution is not defined for this configuration.")
    out_len = x.shape[2] - width + 1

    # Cast only weight to float32 for reduced-precision dtypes so that
    # per-tap multiplies are promoted to float32 via PyTorch type promotion
    # (bf16 × fp32 → fp32).  Weight is small (dim × width) so the cast is
    # cheap, whereas casting the full x tensor (batch × dim × seq_len)
    # would add a large node to the Synapse graph and hurt performance.
    orig_dtype = x.dtype
    needs_upcast = orig_dtype in (torch.bfloat16, torch.float16)

    # Broadcast weight: (dim, width) -> (1, dim, width)
    w = weight.unsqueeze(0)
    if needs_upcast:
        w = w.float()

    # Each x_slice (bf16) * w_slice (fp32) auto-promotes to fp32,
    # so accumulation and the running sum stay in fp32.
    out = x[:, :, :out_len] * w[:, :, 0:1]
    for k in range(1, width):
        out = out + x[:, :, k:k + out_len] * w[:, :, k:k + 1]

    if bias is not None:
        out = out + (bias.float() if needs_upcast else bias).unsqueeze(0).unsqueeze(-1)

    if needs_upcast:
        out = out.to(orig_dtype)

    return out


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
    enable_prefix_caching: bool = False,
    load_cache_indices: torch.Tensor | None = None,
    store_cache_indices: torch.Tensor | None = None,
    blocks_caching_range: torch.Tensor | None = None,
    seqlens_offsets_for_blocks: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    metadata=None,
    validate_data: bool = False,
    is_prompt: bool = True,
):

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
        if has_initial_state is not None and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

    # Take all input data for this call
    # Create tensor to get all data from 0 to lest sequence
    # This works bor padded_batch equal 1
    # ss = torch.arange(seq_starts[0], seq_ends[-1])
    seq_x = x_work[:, :]

    # Get init_state for all batch
    if has_initial_state is not None:
        init_state = torch.where(has_initial_state, conv_states[load_cache_indices, -state_len:, :],
                                 torch.zeros(padded_batch, state_len, dim, device=x_work.device, dtype=work_dtype))
    else:
        init_state = torch.zeros(padded_batch, state_len, dim, device=x_work.device, dtype=work_dtype)
    init_state = init_state.transpose(-1, -2)
    init_state = init_state.squeeze()

    seq_input = torch.cat([init_state, seq_x], dim=1)
    if enable_prefix_caching:
        assert seqlens_offsets_for_blocks is not None
        assert blocks_caching_range is not None
        offset = torch.arange(state_len, device=x.device)  # [state_len]
        indices = seqlens_offsets_for_blocks.unsqueeze(1) + offset  # [N, state_len]

        # Gather all slices at once: seq_input is [dim, seq_len+state_len],
        # indices is [N, state_len] -> new_states is [N, dim, state_len]
        new_states = seq_input[:, indices].permute(1, 0, 2)

        # Scatter all updates at once
        conv_states[blocks_caching_range, -state_len:, :] = new_states.transpose(-1, -2)
    if not enable_prefix_caching:
        end = qsl[-1]
        idx = torch.arange(state_len, device=x.device) + end
        new_state = seq_input.index_select(dim=1, index=idx)
        conv_states[store_cache_indices, -state_len:, :] = new_state.transpose(-1, -2)

    # Apply depthwise convolution using element-wise TPC ops.
    # This avoids the MME spatial_convolution input1 weight-transpose
    # that would otherwise stall TPC/MME pipelining on Gaudi.
    seq_input = seq_input.unsqueeze(0)
    seq_out = _depthwise_conv1d_tpc(seq_input, weight_work, bias_work)
    seq_out = _apply_activation(seq_out, activation)

    return seq_out.squeeze(0).to(original_dtype)


def hpu_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    load_cache_indices: torch.Tensor | None = None,
    store_cache_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    initial_state_idx: torch.Tensor | None = None,
    validate_data: bool = False,
):
    if num_accepted_tokens is not None:
        raise NotImplementedError("Speculative decoding updates are not supported in the reference implementation.")
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
        load_cache_indices=load_cache_indices,
        store_cache_indices=store_cache_indices,
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
    load_cache_indices: torch.Tensor | None = None,
    store_cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    metadata=None,
    validate_data: bool = False,
    is_prompt: bool = True,
):

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
        if has_initial_state is not None and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

    out = torch.zeros_like(x_work)

    init_state = conv_states[load_cache_indices, -state_len:, :]
    init_state = init_state.transpose(-1, -2)

    seq_input = torch.cat([init_state, x_work], dim=2)
    new_state = seq_input[:, :, -state_len:]
    # Use element-wise TPC depthwise conv to avoid the MME
    # spatial_convolution input1 weight-transpose stall.
    seq_out = _depthwise_conv1d_tpc(seq_input, weight_work, bias_work)
    seq_out = _apply_activation(seq_out, activation)
    out = seq_out

    with torch.no_grad():
        conv_states[store_cache_indices, -state_len:, :] = new_state.transpose(-1, -2)

    return out.to(original_dtype)
