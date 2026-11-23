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
                                   _mamba_spec_group_ids=frozenset({1}),
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


def test_async_load_slots_follow_the_device_accepted_counts():
    """Async scheduling knows the last step's accepted tokens only on the device;
    the load slots and conv offsets must match the synchronous ones."""
    runner = _make_runner_stub([_make_request("a", [1, 2]), _make_request("b", [3, 4]), _make_request("c", [5, 6])])
    runner._mamba_num_accepted.update({"a": 2, "b": 1})
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=NUM_SPEC)
    runner.device = torch.device("hpu")
    sync_load, _, sync_conv_offsets = HPUModelRunner._prepare_mamba_spec_decode_idxs(runner, [0, 1, 2],
                                                                                     target_bs=4,
                                                                                     num_tokens=1 + NUM_SPEC)

    num_valid = torch.tensor([2, 1, 1, 1], dtype=torch.int32, device="hpu")
    load, conv_offsets = HPUModelRunner._async_mamba_spec_load_idxs(runner, [0, 1, 2], 4, num_valid)

    assert load.cpu().tolist() == sync_load.tolist()
    assert conv_offsets.cpu().tolist() == sync_conv_offsets.tolist()


def _dummy_step(max_num_tokens: int, batch_size: int) -> tuple[dict, dict]:
    """The scheduled tokens and drafts of the warmup's dummy decode step with one draft per request."""
    runner = types.SimpleNamespace(max_num_tokens=max_num_tokens,
                                   use_contiguous_pa=False,
                                   attn_block_size=128,
                                   max_model_len=32768,
                                   speculative_config=types.SimpleNamespace(num_speculative_tokens=NUM_SPEC,
                                                                            use_eagle=lambda: True))
    for name in ("_generate_seq_lengths", "_prepare_dummy_scenario"):
        setattr(runner, name, types.MethodType(getattr(HPUModelRunner, name), runner))

    def add(requests, num_scheduled_tokens, num_computed_tokens, total_tokens, scheduled_tokens, is_prompt, block_id):
        requests.append(types.SimpleNamespace(req_id=f"r{len(requests)}"))
        num_scheduled_tokens[requests[-1].req_id] = scheduled_tokens

    step: dict = {}
    runner._add_dummy_request = add
    runner._execute_dummy_scenario = lambda requests, scheduled, spec_decode_tokens=None: step.update(
        scheduled=scheduled, drafts=spec_decode_tokens)
    runner._prepare_dummy_scenario(None, (batch_size, 1, 8 * batch_size), decode_num_tokens=1 + NUM_SPEC)
    return step["scheduled"], step["drafts"]


def test_spec_decode_warmup_step_fits_max_num_batched_tokens():
    """A step schedules at most max_num_batched_tokens: with 256 of them, 200 requests leave room for 56 drafts,
    as the scheduler would give; the padded step keeps 1 + K tokens per request, the shape the warmup compiles."""
    scheduled, drafts = _dummy_step(256, 200)
    assert len(scheduled) == 200 and sum(scheduled.values()) == 256
    assert len(drafts) == 56 and all(len(d) == NUM_SPEC for d in drafts.values())
    # A budget that fits every draft keeps the step as it was.
    scheduled, drafts = _dummy_step(4096, 200)
    assert sum(scheduled.values()) == 200 * (1 + NUM_SPEC) and len(drafts) == 200


def test_spec_decode_warmup_skips_shapes_without_room_for_a_draft():
    """256 requests of 1 + K tokens cannot occur with 256 tokens a step: each request takes at least one."""
    calls = []
    runner = types.SimpleNamespace(max_num_tokens=256,
                                   use_contiguous_pa=False,
                                   speculative_config=types.SimpleNamespace(num_speculative_tokens=NUM_SPEC),
                                   bucketing_manager=types.SimpleNamespace(seed_decode_buckets=[(128, 1, 0), (256, 1,
                                                                                                              0)]))
    runner._prepare_dummy_scenario = lambda prompt_cfg, decode_cfg, decode_num_tokens=1: calls.append(
        (decode_cfg[0], decode_num_tokens))
    HPUModelRunner._warmup_mamba_spec_decode_bucket(runner, (512, 1, 4096))
    assert calls == []
    HPUModelRunner._warmup_mamba_spec_decode_bucket(runner, (256, 1, 4096))
    assert calls == [(256, 1), (128, 1 + NUM_SPEC)]
