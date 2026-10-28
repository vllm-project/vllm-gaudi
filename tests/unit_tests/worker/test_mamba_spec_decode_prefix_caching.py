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
