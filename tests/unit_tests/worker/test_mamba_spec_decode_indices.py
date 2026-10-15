# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Mamba2 speculative-decoding state indices of HPUModelRunner.

With K speculative tokens every request owns 1 + K mamba state blocks per cache
group. A decode step resumes from block num_accepted - 1 of the request's
previous step and writes the state after query token t to block t.
"""

import types

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available

from vllm_gaudi.v1.worker.hpu_input_batch import CachedRequestState, InputBatch
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

NUM_SPEC = 1
MAX_MODEL_LEN = 1024
ATTN_BLOCK_SIZE = 16
PAD_BLOCK_ID = 99


def _make_request(req_id: str, mamba_blocks: list[int]) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[0] * 4,
        sampling_params=SamplingParams(),
        pooling_params=None,
        mm_features=[],
        block_ids=([40, 41], mamba_blocks),
        generator=None,
        num_computed_tokens=4,
        output_token_ids=[],
    )


def _make_input_batch(max_num_blocks_per_req=None) -> InputBatch:
    # Group 0: attention. Group 1: mamba without prefix caching, whose block
    # size is max_model_len (one state block per request without spec decode).
    return InputBatch(
        max_num_reqs=4,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=torch.device("hpu"),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
        block_sizes=[ATTN_BLOCK_SIZE, MAX_MODEL_LEN],
        kernel_block_sizes=[ATTN_BLOCK_SIZE, MAX_MODEL_LEN],
        max_num_blocks_per_req=max_num_blocks_per_req,
    )


def _make_runner_stub(requests: list[CachedRequestState]):
    batch = _make_input_batch(max_num_blocks_per_req=[MAX_MODEL_LEN // ATTN_BLOCK_SIZE, 1 + NUM_SPEC])
    for req in requests:
        batch.add_request(req)
    runner = types.SimpleNamespace(input_batch=batch,
                                   _mamba_num_accepted={},
                                   _compact_gdn_group_ids=set(),
                                   _MAMBA_PAD_BLOCK_ID=PAD_BLOCK_ID)
    runner.prepare_mamba_state_idxs = types.MethodType(HPUModelRunner.prepare_mamba_state_idxs, runner)
    return runner


def test_mamba_rows_hold_one_block_per_speculative_position():
    """The scheduler gives a mamba group 1 + K blocks per request."""
    with pytest.raises(ValueError):
        _make_input_batch().add_request(_make_request("a", [1, 2]))
    batch = _make_input_batch(max_num_blocks_per_req=[MAX_MODEL_LEN // ATTN_BLOCK_SIZE, 1 + NUM_SPEC])
    batch.add_request(_make_request("a", [1, 2]))
    assert batch.block_table[1].get_cpu_tensor()[0, :2].tolist() == [1, 2]


def test_spec_decode_resumes_from_last_accepted_slot():
    runner = _make_runner_stub([_make_request("a", [1, 2]), _make_request("b", [3, 4]), _make_request("c", [5, 6])])
    # "a" accepted its draft, "b" did not, "c" has not decoded yet.
    runner._mamba_num_accepted.update({"a": 2, "b": 1})

    load, store, conv_offsets = HPUModelRunner._prepare_mamba_spec_decode_idxs(runner, [0, 1, 2],
                                                                               target_bs=4,
                                                                               num_tokens=1 + NUM_SPEC)

    assert load.shape == (2, 4) and store.shape == (2, 4, 1 + NUM_SPEC)
    assert load[1].tolist() == [2, 3, 5, PAD_BLOCK_ID]
    assert store[1].tolist() == [[1, 2], [3, 4], [5, 6], [PAD_BLOCK_ID, PAD_BLOCK_ID]]
    assert conv_offsets.tolist() == [1, 0, 0, 0]


def test_step_without_drafts_still_resumes_from_last_accepted_slot():
    """A step that schedules no drafts must not fall back to slot 0."""
    runner = _make_runner_stub([_make_request("a", [1, 2]), _make_request("b", [3, 4])])
    runner._mamba_num_accepted.update({"a": 2, "b": 1})

    load, store, conv_offsets = HPUModelRunner._prepare_mamba_spec_decode_idxs(runner, [0, 1],
                                                                               target_bs=2,
                                                                               num_tokens=1)

    assert load[1].tolist() == [2, 3]
    assert store[1].tolist() == [[1], [3]]
    assert conv_offsets.tolist() == [1, 0]
