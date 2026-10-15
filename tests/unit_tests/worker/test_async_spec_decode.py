# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for async scheduling with speculative decoding in HPUModelRunner.

The host prepares a step before the previous one is verified and counts all
its drafts as accepted. A request that ran in that step is delta positions
ahead (delta = drafts + 1 - valid output tokens), and the device moves its
tokens back: positions, slots and block usage. Its first input token is the
last valid output of the previous step, the next ones are that step's drafts.
"""

import types

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm_gaudi.extension.ops import flat_pa
from vllm_gaudi.extension.runtime import clear_config, get_config
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm_gaudi.v1.worker.hpu_model_runner import (HPUAttentionMetadataProcessor, HPUModelRunner, trim_attn_metadata)

DEVICE = current_platform.device_type
BLOCK_SIZE = 4


def _paged_attn_stub():
    runner = types.SimpleNamespace(attn_block_size=BLOCK_SIZE,
                                   use_contiguous_pa=False,
                                   _PAD_BLOCK_ID=99,
                                   model_config=types.SimpleNamespace(dtype=torch.float32))
    # Two padding blocks after the real ones.
    runner.bucketing_manager = types.SimpleNamespace(find_decode_bucket=lambda bs, n: (bs, 1, n + 2))
    runner.get_dp_padding = lambda n: 0
    return runner


def _host_buffers(tables, last_positions, with_block_ends=True):
    slots = [[table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE] for table, p in zip(tables, last_positions)]
    return HPUModelRunner.get_habana_paged_attn_buffers(_paged_attn_stub(),
                                                        tables,
                                                        slots,
                                                        len(tables),
                                                        block_size=BLOCK_SIZE,
                                                        with_block_ends=with_block_ends)


def test_shifted_buffers_count_the_positions_left_in_each_block():
    # Last positions 5, 8 and 3; rows 0 and 1 move back by one, row 2 does not.
    tables = [[10, 11], [20, 21, 22], [30]]
    last_positions = [5, 8, 3]
    shifts = [1, 1, 0]
    _, groups, usage, ends = _host_buffers(tables, last_positions)

    unshifted = HPUModelRunner._shift_paged_attn_buffers(usage, groups, ends, torch.zeros(3, dtype=torch.int32),
                                                         BLOCK_SIZE)
    assert [t.tolist() for t in unshifted] == [usage.tolist(), groups.tolist()]

    shifted_usage, shifted_groups = HPUModelRunner._shift_paged_attn_buffers(usage, groups, ends,
                                                                             torch.tensor(shifts, dtype=torch.int32),
                                                                             BLOCK_SIZE)
    expected_usage, expected_groups = [], []
    seen = [0] * len(tables)
    for group, old_usage in zip(groups.tolist(), usage.tolist()):
        if group < 0:
            expected_usage.append(old_usage)
            expected_groups.append(group)
            continue
        block, seen[group] = seen[group], seen[group] + 1
        left = min(BLOCK_SIZE, last_positions[group] - shifts[group] + 1 - block * BLOCK_SIZE)
        # A block with no positions left is padding.
        expected_usage.append(left if left > 0 else 1)
        expected_groups.append(group if left > 0 else -1)
    assert shifted_usage.tolist() == expected_usage
    assert shifted_groups.tolist() == expected_groups
    # Row 1 moved back across a block boundary: its last block left the row.
    assert expected_groups[:6] == [0, 0, 1, 1, -1, 2]


@pytest.fixture
def runtime_config():
    clear_config()
    get_config()
    yield
    clear_config()


def _decode_attention(query, key_cache, value_cache, block_list, block_groups, block_usage):
    """flat_pa over decode buffers, with the bias and block mapping the metadata processor builds."""
    metadata = HPUAttentionMetadataV1.make_decode_metadata(block_list=block_list.to(DEVICE),
                                                           block_usage=block_usage.to(DEVICE),
                                                           block_groups=block_groups.to(DEVICE),
                                                           input_positions=None,
                                                           slot_mapping=None,
                                                           block_size=BLOCK_SIZE,
                                                           window_block_list=None,
                                                           window_block_usage=None,
                                                           window_block_groups=None,
                                                           chunked_block_list=None,
                                                           chunked_block_usage=None,
                                                           chunked_block_groups=None)
    metadata = HPUAttentionMetadataProcessor._set_block_mapping(types.SimpleNamespace(block_size=BLOCK_SIZE),
                                                                trim_attn_metadata(metadata), query.shape[0], DEVICE,
                                                                query.dtype)

    def fetch(cache, blocks, **kwargs):
        return cache.index_select(0, blocks)

    return flat_pa(query,
                   key_cache,
                   value_cache,
                   metadata.block_list,
                   metadata.block_mapping,
                   metadata.attn_bias,
                   metadata.block_groups,
                   BLOCK_SIZE,
                   1.0,
                   matmul_qk_op=torch.matmul,
                   position_bias=None,
                   matmul_av_op=torch.matmul,
                   batch2block_matmul_op=torch.matmul,
                   block2batch_matmul_op=torch.matmul,
                   keys_fetch_func=fetch,
                   values_fetch_func=fetch,
                   sinks=None,
                   k_scales=None,
                   v_scales=None).cpu()


def test_attention_over_shifted_buffers_matches_the_real_positions(runtime_config):
    # Three requests with two rows each; the host counted a's and b's drafts
    # as accepted, but both were rejected. a's second row and b's first row
    # moved past a block boundary on the host, so moving back empties a block.
    host_positions = [7, 8, 8, 9, 5, 6]
    shifts = [1, 1, 1, 1, 0, 0]
    request_blocks = [[10, 11, 12], [10, 11, 12], [20, 21, 22], [20, 21, 22], [30, 31], [30, 31]]
    real_positions = [p - s for p, s in zip(host_positions, shifts)]

    def tables(positions):
        return [blocks[:p // BLOCK_SIZE + 1] for blocks, p in zip(request_blocks, positions)]

    host_list, host_groups, host_usage, host_ends = _host_buffers(tables(host_positions), host_positions)
    usage, groups = HPUModelRunner._shift_paged_attn_buffers(host_usage.to(DEVICE), host_groups.to(DEVICE),
                                                             host_ends.to(DEVICE),
                                                             torch.tensor(shifts, dtype=torch.int32, device=DEVICE),
                                                             BLOCK_SIZE)
    real_list, real_groups, real_usage = _host_buffers(tables(real_positions), real_positions, with_block_ends=False)

    generator = torch.Generator().manual_seed(0)
    q_heads, kv_heads, head_size = 4, 2, 8
    num_slots = (_paged_attn_stub()._PAD_BLOCK_ID + 1) * BLOCK_SIZE
    key_cache, value_cache = (torch.randn(num_slots, kv_heads, head_size, generator=generator).to(DEVICE)
                              for _ in range(2))
    query = torch.randn(len(host_positions), 1, q_heads * head_size, generator=generator).to(DEVICE)

    shifted = _decode_attention(query, key_cache, value_cache, host_list, groups, usage)
    expected = _decode_attention(query, key_cache, value_cache, real_list, real_groups, real_usage)
    assert not shifted.isnan().any()
    torch.testing.assert_close(shifted, expected, atol=1e-4, rtol=1e-4)


def test_shifted_slots_use_the_block_of_the_earlier_position():
    runner = types.SimpleNamespace(_PAD_BLOCK_ID=99, _resolve_block=lambda b: b)
    block_table = torch.tensor([[10, 11, 12], [20, 21, 22], [0, 0, 0]], dtype=torch.int32)
    positions = torch.tensor([[5, 6], [8, 9], [0, 0]])
    real_tokens = torch.tensor([[True, True], [True, False], [False, False]])
    slot_mapping = torch.tensor([[45, 46], [88, 396], [397, 398]])
    shifted = HPUModelRunner._shifted_slot_mapping(runner, slot_mapping, positions, real_tokens, block_table, 2, 1,
                                                   BLOCK_SIZE)
    # 4 -> block 11, 5 -> block 11, 7 -> block 21; padding keeps its dummy slots.
    assert shifted.tolist() == [[44, 45], [87, 396], [397, 398]]


def test_shift_counts_rejected_drafts_of_the_previous_step_only():
    runner = types.SimpleNamespace(device=DEVICE,
                                   input_batch=types.SimpleNamespace(prev_req_id_to_index={
                                       "a": 0,
                                       "b": 1,
                                       "p": 2
                                   }),
                                   _async_spec_prev_num_drafts=np.array([1, 1, 0], dtype=np.int32),
                                   _async_spec_prev_num_valid=torch.tensor([2, 1, 1], dtype=torch.int32, device=DEVICE),
                                   _mamba_num_accepted={"c": 2})
    # a accepted its draft, b rejected it, p finished its prefill, c skipped
    # the previous step (exact counts, accepted tokens from the host).
    delta, num_valid = HPUModelRunner._async_spec_shift(runner, ["a", "b", "p", "c"], 5)
    assert delta.cpu().tolist() == [0, 1, 0, 0, 0]
    assert num_valid.cpu().tolist() == [2, 1, 1, 2, 1]


def test_input_ids_take_the_last_valid_token_and_the_drafts():
    runner = types.SimpleNamespace(
        device=DEVICE,
        speculative_config=types.SimpleNamespace(num_speculative_tokens=1),
        # Previous step: b in row 0 rejected its draft, a in row 1 accepted it.
        input_batch=types.SimpleNamespace(prev_req_id_to_index={
            "b": 0,
            "a": 1
        },
                                          prev_sampled_token_ids=torch.tensor([[7, -1], [5, 6]],
                                                                              dtype=torch.int32,
                                                                              device=DEVICE)),
        _async_spec_prev_num_valid=torch.tensor([1, 2], dtype=torch.int32, device=DEVICE),
        _draft_token_ids=torch.tensor([[70], [60]], device=DEVICE))
    # The host only has placeholders for a and b; "new" did not run before.
    token_ids = torch.tensor([[-1], [-1], [-1], [-1], [5], [-1]], dtype=torch.int32, device=DEVICE)
    HPUModelRunner._scatter_async_spec_input_ids(runner, token_ids, ["a", "b", "new"], 2, [2, 2, 1])
    assert token_ids.view(-1).cpu().tolist() == [6, 60, 7, 70, 5, -1]


def test_spec_rows_pad_single_tokens():
    spec = torch.tensor([[1, 2], [3, -1], [9, 9]], dtype=torch.int32)
    assert HPUModelRunner._as_spec_rows(spec, 2, 2).tolist() == [[1, 2], [3, -1]]
    single = torch.tensor([4, 5, 6])
    assert HPUModelRunner._as_spec_rows(single, 2, 2).tolist() == [[4, -1], [5, -1]]
