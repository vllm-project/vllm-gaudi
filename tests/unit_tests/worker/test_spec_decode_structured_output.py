# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for structured output with speculative decoding in HPUModelRunner.

The scheduler sends every structured output request a block of the same number
of mask rows, one per position the step samples: its drafts, then the bonus
token. The runner puts each row on that position's logits row, which for a
decode of a speculative step is request * (max_num_draft_tokens + 1) on, and
masks them before the drafts are verified.
"""

import dataclasses
import types

import numpy as np
import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

VOCAB = 64  # two int32 bitmask words


def _bitmask(*allowed_per_row):
    """One bitmask row per argument, allowing the given token ids."""
    rows = np.zeros((len(allowed_per_row), VOCAB // 32), dtype=np.uint32)
    for row, allowed in zip(rows, allowed_per_row):
        for token in allowed:
            row[token // 32] |= np.uint32(1 << (token % 32))
    return rows.view(np.int32)


def test_bitmask_blocks_land_on_each_requests_rows():
    # Blocks of 2 rows (one draft + the bonus), in the scheduler's order: b, then a.
    output = GrammarOutput(["b", "a"], _bitmask([1], [2], [3], [4]))
    # a: a decode with a draft at rows 0-1; b: a prefill sampling one token at row 4.
    bitmask, rows = HPUModelRunner._sorted_grammar_bitmask(output, 6, {"a": (0, 2), "b": (4, 1)})
    assert sorted(rows) == [0, 1, 4]
    assert (bitmask[0] == _bitmask([3])[0]).all() and (bitmask[1] == _bitmask([4])[0]).all()
    # b uses the first row of its block only.
    assert (bitmask[4] == _bitmask([1])[0]).all()
    # Other rows allow everything.
    assert (bitmask[[2, 3, 5]] == -1).all()


def test_requests_without_logits_rows_are_skipped():
    output = GrammarOutput(["chunk", "a"], _bitmask([1], [1], [2], [2]))
    bitmask, rows = HPUModelRunner._sorted_grammar_bitmask(output, 2, {"a": (0, 1)})
    assert rows == [0]
    assert (bitmask[0] == _bitmask([2])[0]).all() and (bitmask[1] == -1).all()


def test_speculative_decodes_own_their_drafts_and_bonus_rows():
    # Two logits rows per request; request 1 has no draft this step.
    metadata = dataclasses.replace(SpecDecodeMetadata.make_dummy([[5], [], [7]], device=torch.device("cpu")),
                                   logits_indices=torch.arange(6))
    assert HPUModelRunner._spec_decode_logits_rows(["r0", "r1", "r2"], metadata) == {
        "r0": (0, 2),
        "r1": (2, 1),
        "r2": (4, 2),
    }


def test_masks_apply_only_to_the_rows_of_structured_requests():
    runner = types.SimpleNamespace(device="cpu", _sorted_grammar_bitmask=HPUModelRunner._sorted_grammar_bitmask)
    logits = torch.zeros(4, VOCAB, dtype=torch.bfloat16)
    output = GrammarOutput(["a"], _bitmask([3], [5]))
    HPUModelRunner.apply_grammar_bitmask(runner, output, logits, {"a": (0, 2)})
    assert logits[0, 3] == 0 and torch.isinf(logits[0, :3]).all() and torch.isinf(logits[0, 4:]).all()
    assert logits[1, 5] == 0 and torch.isinf(logits[1]).sum() == VOCAB - 1
    assert (logits[2:] == 0).all()


def test_drafts_keep_the_requests_of_the_step_that_proposed_them():
    # The batch changed after the proposal (async scheduling takes the drafts a step later).
    runner = types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=["x", "y"]),
                                   _draft_token_ids=torch.tensor([[5], [6]]),
                                   _draft_token_req_ids=["y", "z"])
    out = HPUModelRunner.take_draft_token_ids(runner)
    assert out.req_ids == ["y", "z"] and out.draft_token_ids == [[5], [6]]
    assert runner._draft_token_ids is None and runner._draft_token_req_ids is None
