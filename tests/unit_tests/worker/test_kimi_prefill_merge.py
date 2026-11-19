# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merge decision of fresh prefills for hybrid models, gated by HPU_KIMI_PREFILL_MERGE."""

import pytest

from tests.unit_tests.worker.test_hpu_mm_prefill_batching import _make_batch_contents, _make_runner_stub
from vllm_gaudi.v1.worker import hpu_model_runner


def _hybrid_runner(flag, kimi=True, prefix_caching=False, max_bs=2, merged_prefill=False):
    runner = _make_runner_stub(max_prefill_batch_size=max_bs)
    runner.num_mamba_like_layers = 4
    runner.is_kimi_k3 = kimi
    runner.use_prefix_caching = prefix_caching
    runner.use_merged_prefill = merged_prefill
    hpu_model_runner.gaudi_envs.HPU_KIMI_PREFILL_MERGE = flag
    return runner


@pytest.fixture(autouse=True)
def _restore_flag():
    saved = getattr(hpu_model_runner.gaudi_envs, "HPU_KIMI_PREFILL_MERGE", False)
    yield
    hpu_model_runner.gaudi_envs.HPU_KIMI_PREFILL_MERGE = saved


def _pair():
    return _make_batch_contents("a", 128), _make_batch_contents("b", 128)


def test_flag_off_keeps_hybrid_unmerged():
    assert _hybrid_runner(False)._can_merge_prefill_contents(*_pair()) is False


def test_flag_on_merges_kimi_without_prefix_caching():
    assert _hybrid_runner(True)._can_merge_prefill_contents(*_pair()) is True


def test_flag_on_respects_prefill_batch_limit():
    assert _hybrid_runner(True, max_bs=1)._can_merge_prefill_contents(*_pair()) is False


def test_flag_on_blocks_other_hybrids():
    assert _hybrid_runner(True, kimi=False)._can_merge_prefill_contents(*_pair()) is False


def test_flag_on_blocks_prefix_caching():
    assert _hybrid_runner(True, prefix_caching=True)._can_merge_prefill_contents(*_pair()) is False


def test_flag_on_blocks_request_with_history():
    lhs, rhs = _pair()
    rhs.context_lens = [64]
    assert _hybrid_runner(True)._can_merge_prefill_contents(lhs, rhs) is False


def test_flag_on_blocks_merged_prefill():
    assert _hybrid_runner(True, merged_prefill=True)._can_merge_prefill_contents(*_pair()) is False
