# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hpu_model_runner.sort_prompts_by_context.

A request scheduled later in a step can get prefix-cache hits on blocks that
another request computes in that same step. Prompts run as separate forwards in
row order, so the prompt rows are sorted by num_computed_tokens to make every
writer run before its readers.
"""

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available

from vllm_gaudi.v1.worker.hpu_input_batch import (CachedRequestState, InputBatch)
from vllm_gaudi.v1.worker.hpu_model_runner import sort_prompts_by_context


def _make_request(req_id: str, prompt_len: int, num_computed_tokens: int, token: int = 0) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[token] * prompt_len,
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


def _req_ids(batch: InputBatch) -> list[str]:
    return list(batch.req_ids[:batch.num_reqs])


def test_reader_moved_after_writer():
    """A prompt with a prefix-cache hit sitting above the prompt that computes
    the shared prefix is moved after it; the decode row is untouched."""
    reqs = [
        _make_request("d0", prompt_len=4, num_computed_tokens=4, token=1),
        _make_request("reader", prompt_len=40, num_computed_tokens=32, token=2),
        _make_request("writer", prompt_len=40, num_computed_tokens=0, token=3),
    ]
    batch = _make_input_batch(reqs)

    sort_prompts_by_context(batch, first_prompt=1)

    assert _req_ids(batch) == ["d0", "writer", "reader"]
    # Per-row state follows the request.
    assert list(batch.num_computed_tokens_cpu[:3]) == [4, 0, 32]
    assert batch.token_ids_cpu[1, 0] == 3
    assert batch.token_ids_cpu[2, 0] == 2
    for i, req_id in enumerate(_req_ids(batch)):
        assert batch.req_id_to_index[req_id] == i


def test_sort_is_ascending_and_stable():
    reqs = [
        _make_request("p48", prompt_len=64, num_computed_tokens=48),
        _make_request("p0a", prompt_len=64, num_computed_tokens=0),
        _make_request("p16", prompt_len=64, num_computed_tokens=16),
        _make_request("p0b", prompt_len=64, num_computed_tokens=0),
    ]
    batch = _make_input_batch(reqs)

    sort_prompts_by_context(batch, first_prompt=0)

    assert _req_ids(batch) == ["p0a", "p0b", "p16", "p48"]


def test_rows_before_first_prompt_untouched():
    reqs = [
        _make_request("d0", prompt_len=4, num_computed_tokens=9),
        _make_request("d1", prompt_len=4, num_computed_tokens=5),
        _make_request("p0", prompt_len=8, num_computed_tokens=2),
    ]
    batch = _make_input_batch(reqs)

    sort_prompts_by_context(batch, first_prompt=2)

    assert _req_ids(batch) == ["d0", "d1", "p0"]


def test_no_prompts_is_no_op():
    reqs = [
        _make_request("d0", prompt_len=4, num_computed_tokens=9),
        _make_request("d1", prompt_len=4, num_computed_tokens=5),
    ]
    batch = _make_input_batch(reqs)

    sort_prompts_by_context(batch, first_prompt=batch.num_reqs)

    assert _req_ids(batch) == ["d0", "d1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
