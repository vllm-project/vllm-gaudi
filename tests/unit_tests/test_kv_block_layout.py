# SPDX-License-Identifier: Apache-2.0
import itertools

import pytest

from vllm_gaudi.v1.worker.kv_block_layout import attn_pad_ids, contiguous_dummy_decode_layout, pad_slot_cycle


def test_pad_slot_cycle_stays_in_the_pad_row_and_wraps():
    pad_slot_id, row_len = 640, 4

    slots = list(itertools.islice(pad_slot_cycle(pad_slot_id, row_len), 10))

    assert slots == [640, 641, 642, 643, 640, 641, 642, 643, 640, 641]
    assert all(pad_slot_id <= slot < pad_slot_id + row_len for slot in slots)


def test_pad_slot_cycle_matches_the_pad_row_of_attn_pad_ids():
    _, pad_block_id, pad_slot_id = attn_pad_ids(100, 400, 128)

    assert pad_slot_id == pad_block_id * 128
    assert next(pad_slot_cycle(pad_slot_id, 128)) == pad_block_id * 128


def test_pad_slot_cycle_single_slot_row():
    assert list(itertools.islice(pad_slot_cycle(7, 1), 3)) == [7, 7, 7]


def test_dummy_layout_without_splitting_puts_the_last_block_at_bucket_minus_one():
    block_id, tokens = contiguous_dummy_decode_layout(decode_num_blocks=50,
                                                      num_attn_blocks=100,
                                                      num_sched_blocks=100,
                                                      decode_block_size=128,
                                                      min_tokens_per_seq=128,
                                                      interleaved_sliding_window=False)

    assert (block_id, tokens) == (49, 128)


def test_dummy_layout_before_kv_cache_init_uses_the_scheduler_pool():
    # num_attn_blocks is 0 before initialize_kv_cache
    block_id, tokens = contiguous_dummy_decode_layout(10, 0, 100, 128, 128, False)

    assert (block_id, tokens) == (9, 128)


def test_dummy_layout_clamps_the_bucket_to_the_pool():
    block_id, _ = contiguous_dummy_decode_layout(500, 100, 100, 128, 128, False)

    assert block_id == 99


@pytest.mark.parametrize("decode_num_blocks, expected_block, expected_tokens", [
    (1, 0, 127),
    (4, 0, 4 * 128 - 1),
    (5, 1, 128 - 1),
    (11, 2, 3 * 128 - 1),
])
def test_dummy_layout_with_virtual_splitting_ends_in_the_requested_kernel_block(decode_num_blocks, expected_block,
                                                                                expected_tokens):
    # 4 kernel blocks per scheduler block: 400 kernel blocks, 100 scheduler blocks
    block_id, tokens = contiguous_dummy_decode_layout(decode_num_blocks, 400, 100, 128, 128, False)

    assert (block_id, tokens) == (expected_block, expected_tokens)
    # kernel block id the layout reaches: scheduler block * ratio + last kernel block within it
    last_kernel_block = (tokens + 1) // 128 - 1
    assert block_id * 4 + last_kernel_block == decode_num_blocks - 1


def test_dummy_layout_with_virtual_splitting_keeps_tokens_for_interleaved_sliding_window():
    block_id, tokens = contiguous_dummy_decode_layout(11, 400, 100, 128, 640, True)

    assert (block_id, tokens) == (2, 640)
