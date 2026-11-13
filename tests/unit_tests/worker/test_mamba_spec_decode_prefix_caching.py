# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Mamba2 speculative-decoding state slots of HPUModelRunner with prefix caching.

In the "align" cache mode a mamba block table is indexed by position: column j
holds the state after (j + 1) * block_size tokens. With K speculative tokens a
decode step writes its 1 + K states to the columns from the one of its last
query token on, resumes from the slots of the previous step, and copies a
block boundary state it crossed to the boundary's column.
"""

import types

import numpy as np
import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_gaudi.v1.worker.hpu_input_batch import CachedRequestState, InputBatch
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

NUM_SPEC = 1
MAX_MODEL_LEN = 1024
BLOCK_SIZE = 16
PAD_BLOCK_ID = 99
WIDTH = 4  # conv kernel width
STATE_LEN = WIDTH - 1


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


def _make_runner_stub(rows: dict[str, list[int]], num_computed: list[int]):
    # Group 0: attention. Group 1: mamba in the "align" cache mode, one block
    # table column per block of the sequence plus the speculative ones.
    batch = InputBatch(
        max_num_reqs=4,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=torch.device("hpu"),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
        block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE, BLOCK_SIZE],
        max_num_blocks_per_req=[MAX_MODEL_LEN // BLOCK_SIZE, MAX_MODEL_LEN // BLOCK_SIZE + NUM_SPEC],
    )
    for req_id, mamba_blocks in rows.items():
        batch.add_request(_make_request(req_id, mamba_blocks))
    batch.num_computed_tokens_cpu[:len(num_computed)] = num_computed
    mamba_spec = MambaSpec(block_size=BLOCK_SIZE,
                           shapes=((STATE_LEN + NUM_SPEC, 1), (1, )),
                           dtypes=(torch.float32, torch.float32),
                           mamba_cache_mode="align",
                           num_speculative_blocks=NUM_SPEC)
    attn_spec = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.bfloat16)
    runner = types.SimpleNamespace(input_batch=batch,
                                   cache_config=types.SimpleNamespace(mamba_block_size=BLOCK_SIZE),
                                   kv_cache_config=types.SimpleNamespace(kv_cache_groups=[
                                       types.SimpleNamespace(kv_cache_spec=attn_spec, layer_names=["attn"]),
                                       types.SimpleNamespace(kv_cache_spec=mamba_spec, layer_names=["mixer"]),
                                   ]),
                                   device=torch.device("hpu"),
                                   _mamba_num_accepted={},
                                   _mamba_state_col={},
                                   _compact_gdn_group_ids=set(),
                                   _MAMBA_PAD_BLOCK_ID=PAD_BLOCK_ID)
    for name in ("prepare_mamba_state_idxs", "_mamba_spec_window_cols", "_prepare_mamba_spec_decode_idxs",
                 "_copy_mamba_spec_boundary_states", "_drop_stale_mamba_blocks"):
        setattr(runner, name, types.MethodType(getattr(HPUModelRunner, name), runner))
    return runner


def test_decode_slots_move_along_the_block_table():
    # "a" decoded tokens 45-46 into the slots at columns 2-3 and accepted
    # both; it now runs tokens 47-48, so its slots start at column 3. "b" has
    # just finished a 20-token prefill, whose state is at column 1.
    runner = _make_runner_stub({"a": [0, 0, 7, 8, 9], "b": [0, 5, 6]}, num_computed=[47, 20])
    runner._mamba_state_col["a"] = 2
    runner._mamba_num_accepted.update({"a": 2, "b": 1})

    cols = runner._mamba_spec_window_cols([0, 1], np.array([47, 20]), [1 + NUM_SPEC] * 2)
    load, store, conv_offsets = runner._prepare_mamba_spec_decode_idxs([0, 1], 3, 1 + NUM_SPEC, cols)

    assert cols == ([2, 1], [3, 1])
    assert runner._mamba_state_col == {"a": 3, "b": 1}
    # The recurrent state after the last accepted token, then one slot per query token.
    assert load[1].tolist() == [8, 5, PAD_BLOCK_ID]
    assert store[1].tolist() == [[8, 9], [5, 6], [PAD_BLOCK_ID, PAD_BLOCK_ID]]
    # The conv window is read from the first slot of the previous step.
    assert runner.prepare_mamba_state_idxs([0, 1], cols[0], 3)[1].tolist() == [7, 5, PAD_BLOCK_ID]
    assert conv_offsets.tolist() == [1, 0, 0]


def test_prefill_skips_stale_columns():
    """The KV cache manager moved the scratch block of column 2 to column 3 and
    nulled column 2, but the worker's row keeps its id in both."""
    runner = _make_runner_stub({"a": [0, 5, 6, 6, 9]}, num_computed=[64])
    # A chunk ending in column 3 writes the boundary states of columns 1 and 2,
    # and its final state to column 3.
    blocks = torch.tensor([5, 6, 6], dtype=torch.int32)
    runner._drop_stale_mamba_blocks(blocks, 1, 0, 1)
    assert blocks.tolist() == [5, PAD_BLOCK_ID, 6]


class _FakeMixer:

    def __init__(self, conv_state, ssm_state):
        self.kv_cache = (conv_state, ssm_state)
        self.conv_kernel_size = WIDTH


@pytest.fixture
def mamba_states(monkeypatch):
    monkeypatch.setattr("vllm_gaudi.ops.hpu_mamba_mixer2.HPUMambaMixer2", _FakeMixer)
    conv_state = torch.randn(12, STATE_LEN + NUM_SPEC, 1)
    ssm_state = torch.randn(12, 1)
    return conv_state, ssm_state


def test_boundary_states_are_copied_to_their_columns(mamba_states):
    conv_state, ssm_state = mamba_states
    # "a" runs tokens 31-32 in the slots at columns 2-3: the state after its
    # first token, at 32 tokens, belongs in column 1. "b" runs tokens 30-31 in
    # the slots at columns 1-2 and accepts both: the state after 32 tokens,
    # in the second slot, belongs in column 1, the first slot. "c" crosses no
    # boundary.
    runner = _make_runner_stub({"a": [0, 4, 5, 6], "b": [0, 7, 8], "c": [0, 9, 10]}, num_computed=[31, 30, 20])
    runner._mamba_state_col.update({"a": 2, "b": 1, "c": 1})
    runner._mamba_num_accepted.update({"a": 1, "b": 2, "c": 2})
    hpu_conv, hpu_ssm = conv_state.to("hpu"), ssm_state.to("hpu")
    runner.vllm_config = types.SimpleNamespace(compilation_config=types.SimpleNamespace(
        static_forward_context={"mixer": _FakeMixer(hpu_conv, hpu_ssm)}))
    num_scheduled = {req_id: 1 + NUM_SPEC for req_id in "abc"}
    drafts = {req_id: [1] * NUM_SPEC for req_id in "abc"}
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens=num_scheduled, scheduled_spec_decode_tokens=drafts)

    runner._copy_mamba_spec_boundary_states(scheduler_output, 3)

    expected_conv, expected_ssm = conv_state.clone(), ssm_state.clone()
    expected_ssm[4] = ssm_state[5]
    expected_conv[4, :STATE_LEN] = conv_state[5, :STATE_LEN]
    expected_ssm[7] = ssm_state[8]
    expected_conv[7, :STATE_LEN] = conv_state[7, 1:1 + STATE_LEN]
    torch.testing.assert_close(hpu_ssm.cpu(), expected_ssm)
    torch.testing.assert_close(hpu_conv.cpu(), expected_conv)
    # "b" resumes from the first slot, which now holds its state.
    assert runner._mamba_num_accepted == {"a": 1, "b": 1, "c": 2}


def _sync_boundary_decision(num_running, num_valid, src, block_size):
    """The decisions of _copy_mamba_spec_boundary_states for one row: (copy, bias, dest, resume)."""
    aligned = (num_running + num_valid - 1) // block_size * block_size
    if aligned < num_running:
        return False, 0, None, num_valid
    bias, dest = aligned - num_running, aligned // block_size - 1
    if dest == src:
        return bias != 0, bias, dest, 1
    return True, bias, dest, num_valid


def test_boundary_plan_matches_the_synchronous_decisions():
    rng = np.random.default_rng(0)
    num_running = rng.integers(1, 6 * BLOCK_SIZE, 400)
    num_valid = rng.integers(1, NUM_SPEC + 2, 400)
    # The slots start at the column of the step's last query token, give or take the async shift.
    src = (num_running + NUM_SPEC - 1) // BLOCK_SIZE + rng.integers(-1, 1, 400)
    do_copy, bias, dest, resume = HPUModelRunner._mamba_spec_boundary_plan(torch.tensor(num_running),
                                                                           torch.tensor(num_valid), torch.tensor(src),
                                                                           BLOCK_SIZE)
    for i in range(400):
        copy, ref_bias, ref_dest, ref_resume = _sync_boundary_decision(int(num_running[i]), int(num_valid[i]),
                                                                       int(src[i]), BLOCK_SIZE)
        assert bool(do_copy[i]) == copy and int(resume[i]) == ref_resume
        if copy:
            assert int(bias[i]) == ref_bias and int(dest[i]) == ref_dest


ASYNC_PAD_BLOCK_ID = 11  # a spare block: the async copies write the rows without a copy there


def _async_copy(conv_state, ssm_state, rows, num_computed, state_cols, delta, num_valid):
    runner = _make_runner_stub(rows, num_computed=num_computed)
    runner._mamba_state_col.update(state_cols)
    runner._MAMBA_PAD_BLOCK_ID = ASYNC_PAD_BLOCK_ID
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=NUM_SPEC)
    hpu_conv, hpu_ssm = conv_state.to("hpu"), ssm_state.to("hpu")
    runner.vllm_config = types.SimpleNamespace(compilation_config=types.SimpleNamespace(
        static_forward_context={"mixer": _FakeMixer(hpu_conv, hpu_ssm)}))
    runner._async_spec_delta = torch.tensor(delta, dtype=torch.int32, device="hpu")
    runner._mamba_spec_boundary_plan = HPUModelRunner._mamba_spec_boundary_plan
    num_scheduled = {req_id: 1 + NUM_SPEC for req_id in rows}
    drafts = {req_id: [1] * NUM_SPEC for req_id in rows}
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens=num_scheduled, scheduled_spec_decode_tokens=drafts)
    resume = HPUModelRunner._copy_mamba_spec_boundary_states_async(
        runner, scheduler_output, len(rows), torch.tensor(num_valid, dtype=torch.int32, device="hpu"))
    real = [b for b in range(conv_state.shape[0]) if b != ASYNC_PAD_BLOCK_ID]
    return hpu_conv.cpu()[real], hpu_ssm.cpu()[real], resume.cpu().tolist(), real


def test_async_boundary_copies_match_the_synchronous_ones(mamba_states):
    """test_boundary_states_are_copied_to_their_columns with the accepted tokens on the device."""
    conv_state, ssm_state = mamba_states
    conv, ssm, resume, real = _async_copy(conv_state, ssm_state, {
        "a": [0, 4, 5, 6],
        "b": [0, 7, 8],
        "c": [0, 9, 10]
    }, [31, 30, 20], {
        "a": 2,
        "b": 1,
        "c": 1
    }, [0, 0, 0], [1, 2, 2])
    expected_conv, expected_ssm = conv_state.clone(), ssm_state.clone()
    expected_ssm[4] = ssm_state[5]
    expected_conv[4, :STATE_LEN] = conv_state[5, :STATE_LEN]
    expected_ssm[7] = ssm_state[8]
    expected_conv[7, :STATE_LEN] = conv_state[7, 1:1 + STATE_LEN]
    torch.testing.assert_close(ssm, expected_ssm[real])
    torch.testing.assert_close(conv, expected_conv[real])
    assert resume == [1, 1, 2]


def test_async_boundary_copies_use_the_counts_the_host_is_ahead_by(mamba_states):
    """"b" rejected the draft of its previous step, which the host counted as accepted: it has 30
    tokens, not 31, so this step ran its tokens 31-32 in the slots the host chose for 32-33 (columns
    2-3). Both are accepted: the state after 32 tokens, in the second slot, belongs in column 1."""
    conv_state, ssm_state = mamba_states
    conv, ssm, resume, real = _async_copy(conv_state, ssm_state, {
        "b": [0, 7, 8, 3],
        "c": [0, 9, 10]
    }, [31, 20], {
        "b": 2,
        "c": 1
    }, [1, 0], [2, 2])
    expected_conv, expected_ssm = conv_state.clone(), ssm_state.clone()
    expected_ssm[7] = ssm_state[3]
    expected_conv[7, :STATE_LEN] = conv_state[8, 1:1 + STATE_LEN]
    torch.testing.assert_close(ssm, expected_ssm[real])
    torch.testing.assert_close(conv, expected_conv[real])
    # The boundary column is not the first slot, so "b" resumes after its last accepted token.
    assert resume == [2, 2]


def test_async_loads_resume_from_the_previous_steps_slots():
    """test_decode_slots_move_along_the_block_table with the accepted counts on the device."""
    runner = _make_runner_stub({"a": [0, 0, 7, 8, 9], "b": [0, 5, 6]}, num_computed=[47, 20])
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=NUM_SPEC)
    # The previous slots start at columns 2 ("a") and 1 ("b", its prefill's last column); the third row is padding.
    num_valid = torch.tensor([2, 1, 1], dtype=torch.int32, device="hpu")
    load, conv_offsets = HPUModelRunner._async_mamba_spec_load_idxs(runner, [0, 1], 3, num_valid, [2, 1])
    assert load[1].cpu().tolist() == [8, 5, PAD_BLOCK_ID]
    assert conv_offsets.cpu().tolist() == [1, 0, 0]
