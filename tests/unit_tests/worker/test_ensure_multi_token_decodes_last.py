# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hpu_model_runner.ensure_multi_token_decodes_last.

Covers the routing invariant introduced for KV-offload + async spec-decode
(PR #1401, originally #1264): when speculative decoding is disabled, a decode
request with more than one scheduled token (a resumed/catch-up request from
e.g. OffloadingConnector requeue) must be sorted to the end of the decode
region so that `_get_prompts_and_decodes` routes it through the prefill path,
avoiding bucket overflow / Habana workspace OOM.
"""

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available

from vllm_gaudi.v1.worker.hpu_input_batch import (CachedRequestState, InputBatch)
from vllm_gaudi.v1.worker.hpu_model_runner import ensure_multi_token_decodes_last


def _make_request(req_id: str, prompt_len: int, num_computed_tokens: int) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[0] * prompt_len,
        sampling_params=SamplingParams(),
        pooling_params=None,
        mm_features=[],
        block_ids=([], ),
        generator=None,
        num_computed_tokens=num_computed_tokens,
        output_token_ids=[],
    )


def _make_input_batch(reqs: list[CachedRequestState]) -> InputBatch:
    batch = InputBatch(
        max_num_reqs=max(len(reqs), 1),
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device=torch.device("hpu"),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    for i, req in enumerate(reqs):
        assigned = batch.add_request(req)
        assert assigned == i
    return batch


def test_multi_token_decode_sorted_to_end_of_decode_region():
    """[1-tok decode, multi-tok decode, 1-tok decode, prompt] should become
    [1-tok decode, 1-tok decode, multi-tok decode, prompt]."""
    reqs = [
        # 1-tok decode: num_computed >= num_prompt
        _make_request("d0", prompt_len=4, num_computed_tokens=4),
        # multi-tok catch-up decode (num_scheduled_tokens > 1)
        _make_request("d_multi", prompt_len=4, num_computed_tokens=4),
        # another 1-tok decode
        _make_request("d1", prompt_len=4, num_computed_tokens=5),
        # prompt: num_computed < num_prompt
        _make_request("p0", prompt_len=8, num_computed_tokens=2),
    ]
    batch = _make_input_batch(reqs)
    scheduled = {"d0": 1, "d_multi": 5, "d1": 1, "p0": 8}

    ensure_multi_token_decodes_last(batch, scheduled)

    # Expected layout: 1-tok decodes first, then multi-tok decode, then prompt.
    assert list(batch.req_ids[:batch.num_reqs]) == ["d0", "d1", "d_multi", "p0"]
    # Decode region (first 3) preserves the prompt boundary.
    for i in range(3):
        assert batch.num_computed_tokens_cpu[i] >= batch.num_prompt_tokens[i]
    # Prompt stays last.
    assert batch.num_computed_tokens_cpu[3] < batch.num_prompt_tokens[3]


def test_no_op_when_only_single_token_decodes():
    reqs = [
        _make_request("d0", prompt_len=4, num_computed_tokens=4),
        _make_request("d1", prompt_len=4, num_computed_tokens=5),
        _make_request("p0", prompt_len=8, num_computed_tokens=2),
    ]
    batch = _make_input_batch(reqs)
    scheduled = {"d0": 1, "d1": 1, "p0": 8}
    original_order = list(batch.req_ids[:batch.num_reqs])

    ensure_multi_token_decodes_last(batch, scheduled)

    assert list(batch.req_ids[:batch.num_reqs]) == original_order


def test_no_op_when_only_multi_token_decodes():
    """All decodes are multi-token: order of decode region is preserved."""
    reqs = [
        _make_request("d0", prompt_len=4, num_computed_tokens=4),
        _make_request("d1", prompt_len=4, num_computed_tokens=5),
        _make_request("p0", prompt_len=8, num_computed_tokens=2),
    ]
    batch = _make_input_batch(reqs)
    scheduled = {"d0": 3, "d1": 4, "p0": 8}
    original_order = list(batch.req_ids[:batch.num_reqs])

    ensure_multi_token_decodes_last(batch, scheduled)

    # Both d0 and d1 are multi-tok; write_pos never advances, no swaps occur.
    assert list(batch.req_ids[:batch.num_reqs]) == original_order


def test_decode_only_batch_no_prompt():
    """No prompt in the batch: decode_end == num_reqs."""
    reqs = [
        _make_request("d_multi", prompt_len=4, num_computed_tokens=4),
        _make_request("d0", prompt_len=4, num_computed_tokens=4),
        _make_request("d1", prompt_len=4, num_computed_tokens=5),
    ]
    batch = _make_input_batch(reqs)
    scheduled = {"d_multi": 7, "d0": 1, "d1": 1}

    ensure_multi_token_decodes_last(batch, scheduled)

    assert list(batch.req_ids[:batch.num_reqs]) == ["d0", "d1", "d_multi"]


def test_prompt_only_batch_unchanged():
    """No decodes: function should be a no-op."""
    reqs = [
        _make_request("p0", prompt_len=8, num_computed_tokens=2),
        _make_request("p1", prompt_len=8, num_computed_tokens=0),
    ]
    batch = _make_input_batch(reqs)
    scheduled = {"p0": 6, "p1": 8}
    original_order = list(batch.req_ids[:batch.num_reqs])

    ensure_multi_token_decodes_last(batch, scheduled)

    assert list(batch.req_ids[:batch.num_reqs]) == original_order


def test_missing_req_id_treated_as_single_token():
    """Defensive: scheduled_tokens.get(req_id, 1) defaults to 1 if missing."""
    reqs = [
        _make_request("d0", prompt_len=4, num_computed_tokens=4),
        _make_request("d_multi", prompt_len=4, num_computed_tokens=4),
    ]
    batch = _make_input_batch(reqs)
    # d_multi is the only key; d0 absent -> treated as 1-tok decode.
    scheduled = {"d_multi": 3}

    ensure_multi_token_decodes_last(batch, scheduled)

    assert list(batch.req_ids[:batch.num_reqs]) == ["d0", "d_multi"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
