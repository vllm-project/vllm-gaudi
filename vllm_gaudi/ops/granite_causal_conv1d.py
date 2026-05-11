# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
"""Granite 4.0 specific causal conv1d implementation.

This is a simplified conv1d implementation based on the v0.17.1 code,
adapted for the v0.19.0 metadata interface (separate load/store
cache indices).  It processes one sequence at a time
(padded_batch == 1) and supports prefix caching.

Used exclusively by hpu_mamba_mixer2.py (Granite 4.0).  Other models
continue to use causal_conv1d_pytorch.py.
"""

from __future__ import annotations

import torch

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    _apply_activation,
    _depthwise_conv1d_tpc,
    _ensure_query_start_loc,
    _flatten_inputs_for_update,
    _normalize_activation,
)


def granite_causal_conv1d_fn(
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

    qsl = _ensure_query_start_loc(query_start_loc)
    assert qsl is not None

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
            raise ValueError("Input tensor must be in channel-last or "
                             "channel-first memory layout.")
        if has_initial_state is not None \
                and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

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
    else:
        end = qsl[-1]
        idx = torch.arange(state_len, device=x_work.device) + end
        new_state = seq_input.index_select(dim=1, index=idx)
        conv_states[store_cache_indices, -state_len:, :] = new_state.transpose(-1, -2)

    # Apply depthwise convolution using element-wise TPC ops.
    seq_input = seq_input.unsqueeze(0)
    seq_out = _depthwise_conv1d_tpc(seq_input, weight_work, bias_work)
    seq_out = _apply_activation(seq_out, activation)

    return seq_out.squeeze(0).to(original_dtype)


def granite_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    load_cache_indices: torch.Tensor | None = None,
    store_cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    validate_data: bool = False,
):
    activation = _normalize_activation(activation)
    dim = weight.size(0)

    flat_x, qsl, reshape_spec = _flatten_inputs_for_update(x, query_start_loc, dim)

    result = granite_causal_conv1d_fn_update(
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


def granite_causal_conv1d_fn_update(
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

    qsl = _ensure_query_start_loc(query_start_loc)
    assert qsl is not None

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
            raise ValueError("Input tensor must be in channel-last or "
                             "channel-first memory layout.")
        if has_initial_state is not None \
                and has_initial_state.numel() != padded_batch:
            raise ValueError("'has_initial_state' must align with 'query_start_loc'.")

    init_state = conv_states[load_cache_indices, -state_len:, :]
    init_state = init_state.transpose(-1, -2)

    seq_input = torch.cat([init_state, x_work], dim=2)
    new_state = seq_input[:, :, -state_len:]
    # Use element-wise TPC depthwise conv to avoid the MME
    # spatial_convolution input1 weight-transpose stall.
    seq_out = _depthwise_conv1d_tpc(seq_input, weight_work, bias_work)
    seq_out = _apply_activation(seq_out, activation)

    conv_states[store_cache_indices, -state_len:, :] = new_state.transpose(-1, -2)

    return seq_out.to(original_dtype)
