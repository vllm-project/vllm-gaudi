# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention KV block layout helpers shared by the HPU model runner and spec decode.

Under hybrid virtual block splitting a scheduler block of spec.block_size tokens holds
blocks_per_kv_block kernel blocks of attn_block_size tokens: scheduler block s owns kernel
blocks s * ratio .. s * ratio + ratio - 1. The attention cache has num_kernel_blocks + 1
rows; the trailing row is the pad row that padding tokens write into.
Pure Python, so it can be tested without HPU.
"""

import itertools
from collections.abc import Iterator
from typing import Optional


def attn_pad_ids(num_blocks: int, num_attn_kernel_blocks: Optional[int], attn_block_size: int) -> tuple[int, int, int]:
    """Return (num_attn_blocks, pad_block_id, pad_slot_id) in kernel-block units.

    num_blocks is the scheduler block count; num_attn_kernel_blocks is the kernel block count
    of the attention caches (None without virtual splitting, where both are equal).
    """
    num_attn_blocks = num_blocks if num_attn_kernel_blocks is None else num_attn_kernel_blocks
    return num_attn_blocks, num_attn_blocks, num_attn_blocks * attn_block_size


def pad_slot_cycle(pad_slot_id: int, pad_row_len: int) -> Iterator[int]:
    """Slots for padding tokens: cycle within the pad row so they never leave the cache."""
    return itertools.cycle(range(pad_slot_id, pad_slot_id + pad_row_len))


def contiguous_dummy_decode_layout(decode_num_blocks: int, num_attn_blocks: int, num_sched_blocks: int,
                                   decode_block_size: int, min_tokens_per_seq: int,
                                   interleaved_sliding_window: bool) -> tuple[int, int]:
    """Place a contiguous-PA dummy decode request so its highest kernel block id is the bucket - 1.

    Returns (scheduler block_id, tokens per sequence). Before initialize_kv_cache
    num_attn_blocks is 0: fall back to the scheduler pool and ratio 1 (no virtual splitting).
    """
    pool = num_attn_blocks or num_sched_blocks
    blocks_per_kv_block = max(1, pool // max(1, num_sched_blocks))
    block_id, last_kernel_block = divmod(min(decode_num_blocks, pool) - 1, blocks_per_kv_block)
    if blocks_per_kv_block > 1 and not interleaved_sliding_window:
        # Positions 0..len must end in kernel block last_kernel_block.
        min_tokens_per_seq = (last_kernel_block + 1) * decode_block_size - 1
    return block_id, min_tokens_per_seq
