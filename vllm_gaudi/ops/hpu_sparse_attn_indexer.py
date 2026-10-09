# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.forward_context import get_forward_context


def _fill_invalid(buf, n):
    """Fill topk_indices_buffer rows 0..n-1 with the upstream -1 sentinel.

    -1 means "no token" (see vllm.model_executor.layers.sparse_attn_indexer),
    consumed by forward_mqa_sparse to mask out padding instead of gathering it.
    """
    buf[:n, :].fill_(-1)


def forward_hpu(self, hidden_states, q, k, weights):
    """HPU indexer with FP32 weighted ReLU scoring and physical-slot top-k."""
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    kv_cache = self.k_cache.kv_cache
    if isinstance(kv_cache, tuple):
        kv_cache = kv_cache[0]
    block_size = attn_metadata.block_size
    slot_mapping = attn_metadata.slot_mapping.flatten()

    if kv_cache is None or kv_cache.numel() == 0:
        _fill_invalid(self.topk_indices_buffer, q.shape[0])
        return self.topk_indices_buffer

    if not self.skip_k_cache_insert:
        kv_cache.index_copy_(0, slot_mapping[:k.shape[0]], k)

    if attn_metadata.is_prompt:
        _fill_invalid(self.topk_indices_buffer, q.shape[0])
        return self.topk_indices_buffer

    batch_size = q.shape[0]
    block_list = attn_metadata.block_list
    block_groups = attn_metadata.block_groups
    block_usage = attn_metadata.block_usage

    if block_list is None or block_groups is None or block_usage is None:
        raise ValueError("Sparse decode requires block_list, block_groups, and block_usage")

    block_count = block_list.shape[0]
    if batch_size == 0 or block_count == 0:
        _fill_invalid(self.topk_indices_buffer, batch_size)
        return self.topk_indices_buffer

    positions = torch.arange(block_size, device=block_list.device)
    slots = block_list[:, None] * block_size + positions[None, :]
    valid = positions[None, :] < block_usage.round().long()[:, None]
    keys = kv_cache[slots.reshape(-1)].reshape(block_count, block_size, -1)
    owners = block_groups.clamp(0, batch_size - 1)
    # bf16 GEMM; scores only rank slots, and upstream CUDA computes them from FP8 inputs.
    logits = torch.bmm(q[owners], keys.transpose(1, 2)).float()
    scores = (logits.relu() * weights[owners].float()[:, :, None]).sum(1)

    block_index = getattr(attn_metadata, "indexer_block_index", None)
    if block_index is not None:
        # Rank each request over only its own blocks; the appended row pads short requests.
        # Flat int32 index_select: 2-D int64 gathers fail to compile on HPU.
        scores = torch.cat((scores.masked_fill(~valid, -torch.inf),
                            torch.full((1, block_size), -torch.inf, device=scores.device)))
        flat_index = block_index.flatten()
        candidates = scores.index_select(0, flat_index).view(batch_size, -1)
        selected_scores, indices = candidates.topk(self.topk_tokens, dim=-1)
        indices = indices.int()
        rows = torch.arange(batch_size, device=indices.device, dtype=torch.int32)[:, None] * block_index.shape[1]
        positions = flat_index.index_select(0, (rows + indices // block_size).flatten())
        padded_blocks = torch.cat((block_list.int(), block_list.new_zeros(1, dtype=torch.int32)))
        selected_slots = padded_blocks.index_select(0, positions).view(batch_size, -1) * block_size \
            + indices % block_size
        self.topk_indices_buffer[:batch_size] = selected_slots.masked_fill(selected_scores == -torch.inf, -1)
        return self.topk_indices_buffer

    rows = torch.arange(batch_size, device=block_list.device)
    mask = (block_groups[None, :, None] == rows[:, None, None]) & valid[None, :, :]
    scores = torch.where(mask, scores[None, :, :], -torch.inf)
    topk = min(self.topk_tokens, slots.numel())
    selected_scores, indices = scores.flatten(1).topk(topk, dim=-1)
    selected_slots = slots.flatten()[indices].masked_fill(selected_scores == -torch.inf, -1)

    self.topk_indices_buffer[:batch_size, :topk] = selected_slots
    if topk < self.topk_tokens:
        self.topk_indices_buffer[:batch_size, topk:] = -1

    return self.topk_indices_buffer
