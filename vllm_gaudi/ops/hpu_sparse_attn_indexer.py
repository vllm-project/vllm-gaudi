# SPDX-License-Identifier: Apache-2.0
"""HPU sparse-attn indexer (DSA) decode forward with FP32 weighted-ReLU scoring and physical-slot top-k.

Scoring (bf16 BMM, then FP32 ReLU weighted-sum) keeps the previous FP32 score values; the decode path
removes the per-step index work instead. All HPU index arithmetic uses 1-D flat int32 ``index_select``
(2-D int64 gathers fail HPU torch.compile), and the per-shape index constants (block-row offsets, -inf pad
row) are cached across steps.
"""

import torch
from vllm.forward_context import get_forward_context

_BlkTriple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_BLK_CACHE: dict[tuple[int, int, int, str], _BlkTriple] = {}


def _blk_constants(batch_size: int, width: int, block_size: int, device) -> _BlkTriple:
    """Per-shape int32 index constants, reused across decode steps.

    ``rows_int`` = arange(batch_size, int32)[:, None] * width  (per-request flat-index base, shape (B, 1)).
    ``pad_block_row`` = one int32 zero block id appended so ``block_index`` entries equal to block_count stay valid.
    ``pad_row`` = one -inf score row (the padding candidate every short request sees).
    """
    key = (batch_size, width, block_size, str(device))
    hit = _BLK_CACHE.get(key)
    if hit is None:
        rows_int = torch.arange(batch_size, dtype=torch.int32, device=device)[:, None] * width
        pad_block_row = torch.zeros(1, dtype=torch.int32, device=device)
        pad_row = torch.full((1, block_size), -torch.inf, dtype=torch.float32, device=device)
        hit = (rows_int, pad_block_row, pad_row)
        _BLK_CACHE[key] = hit
    return hit


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

    # int32 physical slots per block; flat int32 index_select (1-D) instead of a 2-D int64 gather.
    positions = torch.arange(block_size, device=block_list.device, dtype=torch.int32)
    slots = block_list.reshape(-1, 1).int() * block_size + positions[None, :]
    slots_flat = slots.reshape(-1)
    valid = positions[None, :] < block_usage.round().long()[:, None]
    keys = kv_cache.index_select(0, slots_flat).reshape(block_count, block_size, -1)

    owners = block_groups.clamp(0, batch_size - 1)
    # Scoring matches the previous implementation bit-for-bit: bf16 BMM, then fp32 ReLU weighted-sum.
    logits = torch.bmm(q[owners], keys.transpose(1, 2))
    scores = (logits.float().relu() * weights[owners].float()[:, :, None]).sum(1)

    block_index = getattr(attn_metadata, "indexer_block_index", None)
    if block_index is not None:
        # Rank each request over only its own blocks; the appended row pads short requests.
        # Flat int32 index_select: 2-D int64 gathers fail to compile on HPU.
        rows_int, pad_block_row, pad_row = _blk_constants(batch_size, block_index.shape[1], block_size, scores.device)
        flat_index = block_index.flatten().int()
        padded_scores = torch.cat((scores.masked_fill(~valid, -torch.inf), pad_row))
        candidates = padded_scores.index_select(0, flat_index).view(batch_size, -1)
        selected_scores, indices = candidates.topk(self.topk_tokens, dim=-1)
        indices = indices.int()
        positions = flat_index.index_select(0, (rows_int + indices // block_size).flatten())
        padded_blocks = torch.cat((block_list.int(), pad_block_row))
        selected_slots = padded_blocks.index_select(0, positions).view(batch_size, -1) * block_size \
            + indices % block_size
        self.topk_indices_buffer[:batch_size] = selected_slots.masked_fill(selected_scores == -torch.inf, -1)
        return self.topk_indices_buffer

    rows = torch.arange(batch_size, device=block_list.device)
    mask = (block_groups[None, :, None] == rows[:, None, None]) & valid[None, :, :]
    scores = torch.where(mask, scores[None, :, :], -torch.inf)
    topk = min(self.topk_tokens, slots.numel())
    selected_scores, indices = scores.flatten(1).topk(topk, dim=-1)
    selected_slots = slots_flat[indices.int()].masked_fill(selected_scores == -torch.inf, -1)

    self.topk_indices_buffer[:batch_size, :topk] = selected_slots
    if topk < self.topk_tokens:
        self.topk_indices_buffer[:batch_size, topk:] = -1

    return self.topk_indices_buffer
