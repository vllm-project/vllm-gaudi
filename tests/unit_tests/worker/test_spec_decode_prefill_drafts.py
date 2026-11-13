# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the drafter inputs HPUModelRunner builds for prefill batches.

The drafter runs over every prefill chunk, also chunks that do not finish
their prompt, so that its own KV covers every prompt position. Its input at
position i is token i + 1; after a request's last scheduled token comes the
first token of its next chunk, or the token just sampled when the chunk
finishes the prompt.
"""

import types

import numpy as np
import torch

from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

SEQ_LEN = 8  # padded prefill length of the batch


def _runner(use_merged_prefill: bool = False):
    # "chunked": 10-token prompt, first chunk of 6 tokens scheduled.
    # "last": 8-token prompt, 4 tokens computed, the remaining 4 scheduled; the
    # token it just sampled (500) already sits after the prompt.
    token_ids_cpu = np.zeros((2, 16), dtype=np.int32)
    token_ids_cpu[0, :10] = np.arange(100, 110)
    token_ids_cpu[1, :9] = list(range(200, 208)) + [500]
    input_batch = types.SimpleNamespace(
        req_id_to_index={
            "chunked": 0,
            "last": 1
        },
        num_computed_tokens_cpu=np.array([0, 4], dtype=np.int32),
        num_prompt_tokens=np.array([10, 8], dtype=np.int32),
        num_tokens_no_spec=np.array([10, 9], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    runner = types.SimpleNamespace(input_batch=input_batch, use_merged_prefill=use_merged_prefill)
    runner._prefill_draft_last_tokens = types.MethodType(HPUModelRunner._prefill_draft_last_tokens, runner)
    return runner


def test_partial_chunk_is_followed_by_its_next_prompt_token():
    last, nxt, finishes = _runner()._prefill_draft_last_tokens(["chunked"], [6], SEQ_LEN)
    assert last == [5]
    assert nxt == [106]
    assert finishes == [False]


def test_final_chunk_is_followed_by_the_sampled_token():
    last, nxt, finishes = _runner()._prefill_draft_last_tokens(["last"], [4], SEQ_LEN)
    assert last == [3]
    assert nxt == [500]
    assert finishes == [True]


def test_padded_batch_rows_start_at_multiples_of_the_padded_length():
    last, nxt, _ = _runner()._prefill_draft_last_tokens(["chunked", "last"], [6, 4], SEQ_LEN)
    assert last == [5, SEQ_LEN + 3]
    assert nxt == [106, 500]


def test_merged_prefill_requests_are_packed_back_to_back():
    last, nxt, _ = _runner(use_merged_prefill=True)._prefill_draft_last_tokens(["chunked", "last"], [6, 4], 16)
    assert last == [5, 6 + 3]
    assert nxt == [106, 500]


def test_unknown_request_gets_a_placeholder_token():
    last, nxt, finishes = _runner()._prefill_draft_last_tokens(["dummy"], [3], SEQ_LEN)
    assert last == [2]
    assert nxt == [0]
    assert finishes == [False]


def test_shift_puts_the_next_token_after_each_last_scheduled_position():
    token_ids = torch.tensor([[100, 101, 102, 103, 104, 105, -1, -1], [204, 205, 206, 207, -1, -1, -1, -1]],
                             dtype=torch.int32)
    shifted = HPUModelRunner._shift_prefill_token_ids(token_ids, torch.tensor([5, SEQ_LEN + 3]),
                                                      torch.tensor([106, 500], dtype=torch.int32))
    assert shifted.tolist() == [[101, 102, 103, 104, 105, 106, -1, -1], [205, 206, 207, 500, -1, -1, -1, -1]]
    assert token_ids[0, 0] == 100  # the target's input ids stay untouched
