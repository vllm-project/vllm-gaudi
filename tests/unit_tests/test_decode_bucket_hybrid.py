# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2026 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""
Regression tests for decode bucket generation and warmup in hybrid models.

Hybrid models (e.g., Qwen3.5) have block_size != attn_block_size:
- block_size=640: unified page size for KV cache management
- attn_block_size=128: HPU kernel page size used by paged attention

The decode path (_create_decode_input_data) computes num_blocks using
attn_block_size. Therefore:
1. Decode buckets MUST be generated in attn_block_size units.
2. Warmup seq_lengths MUST produce the correct sum(num_blocks) to match
   the target bucket after find_decode_bucket lookup.
3. For non-contiguous PA, _generate_seq_lengths MUST NOT cap num_blocks
   at kv_cache_config.num_blocks (physical pool), because runtime can
   exceed this via prefix-sharing.

Regression: f24f3f9d introduced a formula for max_decode_blocks using
block_size instead of attn_block_size, and added a physical-pool cap
that prevented large decode buckets from being warmed.
"""

import math

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_gaudi.extension.bucketing.common import (
    HPUBucketingManager,
    find_equal_or_closest_greater_config,
)
from vllm_gaudi.extension.bucketing.exponential import ExponentialBucketingStrategy
from vllm_gaudi.extension.runtime import get_config, clear_config
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

# --- Qwen3.5 hybrid model parameters ---
_QWEN35_BLOCK_SIZE = 640  # unified page size (5 * 128)
_QWEN35_ATTN_BLOCK_SIZE = 128  # HPU kernel page size
_QWEN35_MAX_MODEL_LEN = 262144
_QWEN35_MAX_NUM_SEQS = 45
_QWEN35_NUM_HPU_BLOCKS = 15405  # physical blocks in attn_block_size units


@pytest.fixture(autouse=True)
def default_config(monkeypatch):
    """Reset singleton and pin bucketing config for deterministic tests."""
    # Reset singleton to prevent state leakage between tests
    HPUBucketingManager._instance = None
    # Pin bucketing strategy to avoid env-dependent behavior in CI
    monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", "exp")
    monkeypatch.delenv("VLLM_EXPONENTIAL_BUCKETING", raising=False)
    clear_config()
    get_config()
    yield
    HPUBucketingManager._instance = None
    clear_config()


class _MockConfig:
    """Lightweight mock for get_config()."""

    def __init__(self, **kwargs):
        defaults = dict(
            prefix_caching=False,
            use_contiguous_pa=False,
            merged_prefill=False,
            VLLM_PROMPT_BS_BUCKET_MIN=None,
            VLLM_PROMPT_BS_BUCKET_STEP=None,
            VLLM_PROMPT_BS_BUCKET_MAX=None,
            VLLM_PROMPT_SEQ_BUCKET_MIN=None,
            VLLM_PROMPT_SEQ_BUCKET_STEP=None,
            VLLM_PROMPT_SEQ_BUCKET_MAX=None,
            VLLM_DECODE_BS_BUCKET_MIN=None,
            VLLM_DECODE_BS_BUCKET_STEP=None,
            VLLM_DECODE_BS_BUCKET_MAX=None,
            VLLM_DECODE_BLOCK_BUCKET_MIN=None,
            VLLM_DECODE_BLOCK_BUCKET_STEP=None,
            VLLM_DECODE_BLOCK_BUCKET_MAX=None,
            VLLM_PROMPT_QUERY_BUCKET_MIN=None,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            object.__setattr__(self, k, v)


def _make_bucketing_manager(block_size, max_model_len, max_num_seqs, num_hpu_blocks):
    """Create a minimally-configured HPUBucketingManager."""
    mgr = HPUBucketingManager.__new__(HPUBucketingManager)
    mgr.block_size = block_size
    mgr.max_model_len = max_model_len
    mgr.max_num_seqs = max_num_seqs
    mgr.max_num_prefill_seqs = 1
    mgr.num_hpu_blocks = num_hpu_blocks
    mgr.max_num_batched_tokens = 131072
    mgr.initialized = True
    mgr.mamba_chunk_size = None
    mgr.mamba_chunk_size_is_explicit = False
    mgr.num_speculative_tokens = None
    mgr.use_sliding_window = False
    mgr.fallback_bs_base_step = 2
    mgr.fallback_seq_base_step = 32
    mgr.fallback_blocks_base_step = 32
    mgr._fallback_max_ctx = 0
    return mgr


class _MockModelRunner:
    """Minimal mock of HPUModelRunner for _generate_seq_lengths testing."""

    def __init__(self, use_contiguous_pa, num_blocks, max_model_len, speculative_config=None):
        self.use_contiguous_pa = use_contiguous_pa
        self.kv_cache_config = SimpleNamespace(num_blocks=num_blocks)
        self.max_model_len = max_model_len
        self.speculative_config = speculative_config


def _generate_seq_lengths(runner, num_samples, num_blocks, block_size):
    """Call HPUModelRunner._generate_seq_lengths via unbound method."""
    return HPUModelRunner._generate_seq_lengths(runner, num_samples, num_blocks, block_size)


# =============================================================================
# Test 1: Decode bucket generation uses attn_block_size for hybrid models
# =============================================================================


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_hybrid_decode_buckets_use_attn_block_size(mock_exp_config):
    """Decode buckets for hybrid model must be generated using attn_block_size.

    When block_size=640 is incorrectly used:
      max_decode_blocks = ceil(262144/640)*45 = 18450
    When attn_block_size=128 is correctly used:
      max_decode_blocks = ceil(262144/128)*45 = 92160

    The warmup_model() scopes bucketing_manager.block_size to attn_block_size
    before calling generate_decode_buckets(). This test verifies that using
    the correct block_size produces the right max.
    """
    mock_exp_config.return_value = _MockConfig(use_contiguous_pa=False)

    # With WRONG block_size (640) — the bug
    mgr_wrong = _make_bucketing_manager(
        block_size=_QWEN35_BLOCK_SIZE,
        max_model_len=_QWEN35_MAX_MODEL_LEN,
        max_num_seqs=_QWEN35_MAX_NUM_SEQS,
        num_hpu_blocks=_QWEN35_NUM_HPU_BLOCKS,
    )
    mgr_wrong.generate_decode_buckets()
    wrong_max_ctx = max(ctx for _, _, ctx in mgr_wrong.decode_buckets)

    # With CORRECT block_size (128) — the fix
    mgr_correct = _make_bucketing_manager(
        block_size=_QWEN35_ATTN_BLOCK_SIZE,
        max_model_len=_QWEN35_MAX_MODEL_LEN,
        max_num_seqs=_QWEN35_MAX_NUM_SEQS,
        num_hpu_blocks=_QWEN35_NUM_HPU_BLOCKS,
    )
    mgr_correct.generate_decode_buckets()
    correct_max_ctx = max(ctx for _, _, ctx in mgr_correct.decode_buckets)

    expected_max = math.ceil(_QWEN35_MAX_MODEL_LEN / _QWEN35_ATTN_BLOCK_SIZE) * _QWEN35_MAX_NUM_SEQS
    wrong_expected = math.ceil(_QWEN35_MAX_MODEL_LEN / _QWEN35_BLOCK_SIZE) * _QWEN35_MAX_NUM_SEQS

    assert wrong_max_ctx <= wrong_expected, (
        f"Wrong block_size should produce max_ctx <= {wrong_expected}, got {wrong_max_ctx}")
    assert correct_max_ctx <= expected_max, (
        f"Correct block_size should produce max_ctx <= {expected_max}, got {correct_max_ctx}")
    assert correct_max_ctx > wrong_max_ctx, (
        f"attn_block_size={_QWEN35_ATTN_BLOCK_SIZE} should produce larger buckets than "
        f"block_size={_QWEN35_BLOCK_SIZE}: {correct_max_ctx} vs {wrong_max_ctx}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_hybrid_decode_buckets_cover_runtime_scenarios(mock_exp_config):
    """Decode buckets must cover all runtime-reachable configurations.

    For 28 seqs at max context: 28 * ceil(262144/128) = 28 * 2048 = 57344.
    A bucket >= 57344 must exist for batch_size=28.
    """
    mock_exp_config.return_value = _MockConfig(use_contiguous_pa=False)

    mgr = _make_bucketing_manager(
        block_size=_QWEN35_ATTN_BLOCK_SIZE,
        max_model_len=_QWEN35_MAX_MODEL_LEN,
        max_num_seqs=_QWEN35_MAX_NUM_SEQS,
        num_hpu_blocks=_QWEN35_NUM_HPU_BLOCKS,
    )
    mgr.generate_decode_buckets()

    # For each batch size, the max reachable ctx is bs * max_blocks_per_seq
    max_blocks_per_seq = math.ceil(_QWEN35_MAX_MODEL_LEN / _QWEN35_ATTN_BLOCK_SIZE)

    # Check that large decode scenarios are covered
    test_cases = [
        (28, 37620),  # Real case from the bug report
        (45, 92160),  # Maximum: all seqs at max_model_len
        (1, 2048),  # Single seq at max context
    ]
    for bs, target_ctx in test_cases:
        # Verify target is reachable (within physical limits for the batch)
        max_reachable = bs * max_blocks_per_seq
        assert target_ctx <= max_reachable, (f"Test case ({bs}, {target_ctx}) exceeds reachable max {max_reachable}")

        # Verify a covering bucket exists
        found = find_equal_or_closest_greater_config(mgr.decode_buckets, (bs, 1, target_ctx))
        assert found is not None, (f"No decode bucket found >= ({bs}, 1, {target_ctx}). "
                                   f"Max bucket for bs={bs}: "
                                   f"{max((ctx for b, _, ctx in mgr.decode_buckets if b >= bs), default='NONE')}")


# =============================================================================
# Test 2: _generate_seq_lengths does NOT cap for non-contiguous PA
# =============================================================================


class TestGenerateSeqLengthsNonContiguousPA:
    """Verify _generate_seq_lengths behavior for non-contiguous PA."""

    def test_no_cap_when_num_blocks_exceeds_physical(self):
        """num_blocks > kv_cache_config.num_blocks should NOT be capped.

        This is the key regression: capping prevents large decode buckets
        from being warmed, causing 'not warmed-up' warnings at runtime.
        """
        runner = _MockModelRunner(
            use_contiguous_pa=False,
            num_blocks=_QWEN35_NUM_HPU_BLOCKS,  # 15405
            max_model_len=_QWEN35_MAX_MODEL_LEN,
        )
        target_blocks = 37620  # Much larger than num_blocks=15405

        seq_lengths = _generate_seq_lengths(runner, 28, target_blocks, _QWEN35_ATTN_BLOCK_SIZE)

        # Verify total blocks from seq_lengths matches target
        total_blocks = sum(math.ceil((sl + 1) / _QWEN35_ATTN_BLOCK_SIZE) for sl in seq_lengths)
        assert total_blocks == target_blocks, (
            f"Expected total_blocks={target_blocks}, got {total_blocks}. "
            f"Non-contiguous PA must not cap at kv_cache_config.num_blocks={_QWEN35_NUM_HPU_BLOCKS}")

    def test_max_model_len_still_bounds_per_seq(self):
        """Individual seq_lengths must still be clamped by max_model_len."""
        runner = _MockModelRunner(
            use_contiguous_pa=False,
            num_blocks=_QWEN35_NUM_HPU_BLOCKS,
            max_model_len=_QWEN35_MAX_MODEL_LEN,
        )
        # Large bucket: 1 seq with 92160 blocks (way beyond max_model_len/block_size=2048)
        seq_lengths = _generate_seq_lengths(runner, 1, 92160, _QWEN35_ATTN_BLOCK_SIZE)

        assert len(seq_lengths) == 1
        assert seq_lengths[0] <= _QWEN35_MAX_MODEL_LEN - 1, (
            f"seq_length {seq_lengths[0]} exceeds max_model_len-1={_QWEN35_MAX_MODEL_LEN - 1}")

    @pytest.mark.parametrize("batch_size,target_blocks", [
        (28, 37620),
        (45, 92160),
        (14, 20000),
        (1, 2048),
    ])
    def test_warmup_roundtrip_targets_correct_bucket(self, batch_size, target_blocks):
        """Verify warmup roundtrip: seq_lengths -> num_blocks -> find_decode_bucket.

        The warmup path generates seq_lengths from the target bucket, then the
        runtime decode path recomputes num_blocks from those seq_lengths. The
        resulting sum(num_blocks) must find the same bucket via find_decode_bucket.
        """
        runner = _MockModelRunner(
            use_contiguous_pa=False,
            num_blocks=_QWEN35_NUM_HPU_BLOCKS,
            max_model_len=_QWEN35_MAX_MODEL_LEN,
        )
        max_blocks_per_seq = math.ceil(_QWEN35_MAX_MODEL_LEN / _QWEN35_ATTN_BLOCK_SIZE)

        # Skip unreachable buckets (can't produce these at runtime either)
        max_reachable = batch_size * max_blocks_per_seq
        if target_blocks > max_reachable:
            pytest.skip(f"Bucket ({batch_size}, 1, {target_blocks}) is unreachable "
                        f"(max={max_reachable})")

        seq_lengths = _generate_seq_lengths(runner, batch_size, target_blocks, _QWEN35_ATTN_BLOCK_SIZE)

        # Simulate _create_decode_input_data's num_blocks computation
        num_blocks_per_req = [math.ceil((sl + 1) / _QWEN35_ATTN_BLOCK_SIZE) for sl in seq_lengths]
        total_blocks_after_roundtrip = sum(num_blocks_per_req)

        # The roundtrip total should equal the target (for reachable buckets)
        assert total_blocks_after_roundtrip == target_blocks, (
            f"Roundtrip mismatch for bucket ({batch_size}, 1, {target_blocks}): "
            f"got sum(num_blocks)={total_blocks_after_roundtrip}. "
            f"Warmup will target wrong bucket!")


# =============================================================================
# Test 3: _generate_seq_lengths DOES cap for contiguous PA
# =============================================================================


class TestGenerateSeqLengthsContiguousPA:
    """Verify _generate_seq_lengths caps correctly for contiguous PA."""

    def test_caps_at_physical_blocks(self):
        """For contiguous PA, num_blocks MUST be capped at kv_cache_config.num_blocks.

        This is because contiguous PA uses block_id = num_blocks - 1 as the
        contiguous allocation base, which must be a valid physical block.
        """
        runner = _MockModelRunner(
            use_contiguous_pa=True,
            num_blocks=_QWEN35_NUM_HPU_BLOCKS,  # 15405
            max_model_len=_QWEN35_MAX_MODEL_LEN,
        )
        target_blocks = 37620  # Larger than physical

        seq_lengths = _generate_seq_lengths(runner, 28, target_blocks, _QWEN35_ATTN_BLOCK_SIZE)

        # Total blocks should be capped at num_blocks
        total_blocks = sum(math.ceil((sl + 1) / _QWEN35_ATTN_BLOCK_SIZE) for sl in seq_lengths)
        assert total_blocks <= _QWEN35_NUM_HPU_BLOCKS, (
            f"Contiguous PA: total_blocks={total_blocks} exceeds physical "
            f"num_blocks={_QWEN35_NUM_HPU_BLOCKS}. block_id would overflow!")


# =============================================================================
# Test 4: End-to-end decode bucket max formula for hybrid models
# =============================================================================


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_hybrid_max_decode_blocks_formula(mock_exp_config):
    """Verify max_decode_blocks = ceil(max_model_len/attn_block_size) * max_num_seqs.

    For Qwen3.5: ceil(262144/128) * 45 = 2048 * 45 = 92160.
    This must NOT use block_size=640 which gives ceil(262144/640)*45 = 18450.
    """
    mock_exp_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    # Using attn_block_size=128 (correct)
    _, _, block_cfg = strategy.get_decode_cfgs(
        max_num_seqs=_QWEN35_MAX_NUM_SEQS,
        block_size=_QWEN35_ATTN_BLOCK_SIZE,
        max_num_batched_tokens=131072,
        max_model_len=_QWEN35_MAX_MODEL_LEN,
        max_blocks=_QWEN35_NUM_HPU_BLOCKS,
    )
    expected_max = math.ceil(_QWEN35_MAX_MODEL_LEN / _QWEN35_ATTN_BLOCK_SIZE) * _QWEN35_MAX_NUM_SEQS
    assert block_cfg[2] == expected_max, (
        f"max_decode_blocks should be {expected_max} with attn_block_size={_QWEN35_ATTN_BLOCK_SIZE}, "
        f"got {block_cfg[2]}")

    # Using block_size=640 (wrong — would produce 18450)
    _, _, block_cfg_wrong = strategy.get_decode_cfgs(
        max_num_seqs=_QWEN35_MAX_NUM_SEQS,
        block_size=_QWEN35_BLOCK_SIZE,
        max_num_batched_tokens=131072,
        max_model_len=_QWEN35_MAX_MODEL_LEN,
        max_blocks=_QWEN35_NUM_HPU_BLOCKS,
    )
    wrong_max = math.ceil(_QWEN35_MAX_MODEL_LEN / _QWEN35_BLOCK_SIZE) * _QWEN35_MAX_NUM_SEQS
    assert block_cfg_wrong[2] == wrong_max, (
        f"With block_size=640, max_decode_blocks should be {wrong_max}, got {block_cfg_wrong[2]}")
    assert expected_max > wrong_max, (f"attn_block_size formula ({expected_max}) must produce larger max than "
                                      f"block_size formula ({wrong_max})")


# =============================================================================
# Test 5: Verify the bug scenario — bucket (28, 1, 37620) IS reachable
# =============================================================================


def test_bucket_37620_reachable_at_runtime():
    """Bucket (28, 1, 37620) is reachable: 28 seqs averaging ~1344 blocks each.

    Each seq has context_len ≈ 171903 tokens → ceil(171904/128) = 1344 blocks.
    Sum across 28 seqs ≈ 37620. This is within max_model_len per seq.
    """
    attn_block_size = _QWEN35_ATTN_BLOCK_SIZE
    max_blocks_per_seq = math.ceil(_QWEN35_MAX_MODEL_LEN / attn_block_size)  # 2048
    batch_size = 28
    target_blocks = 37620

    # Each seq needs target_blocks/batch_size ≈ 1344 blocks
    blocks_per_seq = target_blocks / batch_size  # 1343.57
    tokens_per_seq = blocks_per_seq * attn_block_size  # ~171977

    assert tokens_per_seq < _QWEN35_MAX_MODEL_LEN, (f"Scenario requires {tokens_per_seq:.0f} tokens/seq which exceeds "
                                                    f"max_model_len={_QWEN35_MAX_MODEL_LEN}")
    assert blocks_per_seq <= max_blocks_per_seq, (f"Scenario requires {blocks_per_seq:.1f} blocks/seq which exceeds "
                                                  f"max_blocks_per_seq={max_blocks_per_seq}")


# =============================================================================
# Test 6: Regression test — with old cap, bucket warmup targets wrong bucket
# =============================================================================


def test_old_cap_causes_wrong_bucket_warmup():
    """Demonstrate that capping at kv_cache_config.num_blocks causes warmup
    to target the wrong bucket, producing 'not warmed-up' warnings.

    With cap: _generate_seq_lengths(28, min(15405, 37620)=15405, 128)
    → sum(num_blocks) ≈ 15405 → find_decode_bucket finds a smaller bucket.
    """
    runner_capped = _MockModelRunner(
        use_contiguous_pa=True,  # simulate old buggy behavior (cap always)
        num_blocks=_QWEN35_NUM_HPU_BLOCKS,
        max_model_len=_QWEN35_MAX_MODEL_LEN,
    )
    target_bucket_ctx = 37620

    # With cap (simulates old behavior)
    seq_lengths_capped = _generate_seq_lengths(runner_capped, 28, target_bucket_ctx, _QWEN35_ATTN_BLOCK_SIZE)
    total_capped = sum(math.ceil((sl + 1) / _QWEN35_ATTN_BLOCK_SIZE) for sl in seq_lengths_capped)

    # Without cap (correct behavior for non-contiguous PA)
    runner_uncapped = _MockModelRunner(
        use_contiguous_pa=False,
        num_blocks=_QWEN35_NUM_HPU_BLOCKS,
        max_model_len=_QWEN35_MAX_MODEL_LEN,
    )
    seq_lengths_uncapped = _generate_seq_lengths(runner_uncapped, 28, target_bucket_ctx, _QWEN35_ATTN_BLOCK_SIZE)
    total_uncapped = sum(math.ceil((sl + 1) / _QWEN35_ATTN_BLOCK_SIZE) for sl in seq_lengths_uncapped)

    # Capped version misses the target
    assert total_capped < target_bucket_ctx, (f"Capped version should produce fewer blocks than target: "
                                              f"{total_capped} vs {target_bucket_ctx}")
    assert total_capped <= _QWEN35_NUM_HPU_BLOCKS, (
        f"Capped version should be bounded by num_blocks={_QWEN35_NUM_HPU_BLOCKS}")

    # Uncapped version hits the target exactly
    assert total_uncapped == target_bucket_ctx, (f"Uncapped version should produce exactly {target_bucket_ctx} blocks, "
                                                 f"got {total_uncapped}")
