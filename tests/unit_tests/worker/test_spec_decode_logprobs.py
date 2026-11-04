# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the logprobs of speculative decoding steps in HPUModelRunner.

A decode request of a speculative step generates its accepted drafts plus one
token, so its logprobs take that many rows; the scheduler finds a request's
rows through cu_num_generated_tokens. Steps without drafts keep one row per
request.
"""

import dataclasses
import types

import torch

from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner


def _runner(req_id_to_index):
    runner = types.SimpleNamespace(input_batch=types.SimpleNamespace(req_id_to_index=req_id_to_index))
    runner._build_spec_logprobs_output = types.MethodType(HPUModelRunner._build_spec_logprobs_output, runner)
    return runner


def _logprobs(token_ids):
    """One row per token: the token itself, logprob -token, rank token."""
    ids = torch.tensor(token_ids, dtype=torch.int64).view(-1, 1)
    return LogprobsTensors(ids, -ids.float(), ids.view(-1))


def test_steps_without_drafts_keep_one_row_per_request():
    runner = _runner({"a": 0, "b": 1})
    out = HPUModelRunner._build_logprobs_output(runner, [(["b", "a"], _logprobs([20, 10]))], 2)
    assert out.cu_num_generated_tokens is None
    assert out.logprob_token_ids[:, 0].tolist() == [10, 20]


def test_speculative_step_gives_each_request_its_generated_tokens():
    # Decodes "a" (draft accepted: 2 tokens) and "b" (draft rejected: 1 token), padded to 3 rows of
    # 2 positions; a prefill "c" generates one token; index 2 has no request this step.
    runner = _runner({"a": 0, "b": 1, "c": 3})
    sampled = torch.tensor([[11, 12], [21, PLACEHOLDER_TOKEN_ID], [31, PLACEHOLDER_TOKEN_ID]])
    spec = (["a", "b"], _logprobs([11, 12, 21, 0, 31, 0]), sampled)
    out = HPUModelRunner._build_logprobs_output(runner, [(["c"], _logprobs([41]))], 4, spec)
    assert out.cu_num_generated_tokens == [0, 2, 3, 3, 4]
    assert out.logprob_token_ids[:, 0].tolist() == [11, 12, 21, 41]
    # What the scheduler takes for each request (slice_request with its number of new tokens).
    assert out.slice_request(0, 2).logprob_token_ids[:, 0].tolist() == [11, 12]
    assert out.slice_request(1, 1).logprob_token_ids[:, 0].tolist() == [21]
    assert out.slice_request(3, 1).logprob_token_ids[:, 0].tolist() == [41]
    assert out.slice_request(3, 1).logprobs[:, 0].tolist() == [-41.0]


def test_speculative_step_without_other_logprobs():
    runner = _runner({"a": 0})
    sampled = torch.tensor([[11, 12]])
    out = HPUModelRunner._build_logprobs_output(runner, [(["a"], None)], 1, (["a"], _logprobs([11, 12]), sampled))
    assert out.cu_num_generated_tokens == [0, 2]
    assert out.sampled_token_ranks.tolist() == [11, 12]


def test_spec_token_ids_are_each_requests_own_drafts():
    # Two draft rows per request; request 1 has one draft and request 2 none (padding).
    metadata = dataclasses.replace(SpecDecodeMetadata.make_dummy([[5, 6], [7], []], device=torch.device("cpu")),
                                   draft_token_ids=torch.tensor([5, 6, 7, 0, 0, 0]))
    assert HPUModelRunner._spec_token_ids(metadata) == [[5, 6], [7], []]
