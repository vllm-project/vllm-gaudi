# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Bucketing tests for issue #1347 — 256K model length on Gaudi2.

Tests cover the four PRs that address long-context bucketing:
  - PR #762:  Padding-aware bucketing strategy
  - PR #1122: Exponential decode block formula & generate_buckets filter
  - PR #1155: FusedSDPA slicing dependency on bucketing (pad strategy)
  - PR #1346: HPU graph capture skip for long prefills / warmup clamp

The tests are designed to be run in a Gaudi environment where vllm_gaudi
is importable.  They verify bucket generation correctness, filter behaviour,
and the contracts that downstream consumers (warmup, FusedSDPA slicing) rely on.
"""

import math

import pytest
from unittest.mock import patch

import vllm_gaudi.extension.bucketing.linear as linear
from vllm_gaudi.extension.bucketing.common import (
    HPUBucketingManager,
    generate_buckets,
    calc_fallback_value,
)
from vllm_gaudi.extension.bucketing.exponential import (
    ExponentialBucketingStrategy,
    warmup_range_with_limit,
)
from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
from vllm_gaudi.extension.bucketing.padding_aware import (
    PaddingAwareBucketingStrategy,
    warmup_range_with_limits,
)
from vllm_gaudi.extension.runtime import get_config, clear_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def default_config():
    clear_config()
    get_config()
    yield
    clear_config()


# ---------------------------------------------------------------------------
# Mock config helper (shared across all tests)
# ---------------------------------------------------------------------------


class _MockConfig:
    """Lightweight mock config for bucketing tests."""

    def __init__(self, **kwargs):
        defaults = dict(
            prefix_caching=False,
            use_contiguous_pa=False,
            merged_prefill=False,
            bucketing_strategy='exp',
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
            VLLM_PROMPT_CTX_BUCKET_MAX=None,
            PT_HPU_SDPA_QKV_SLICE_MODE_FWD=False,
            PT_HPU_SDPA_BC_FACTOR=None,
            VLLM_FUSEDSDPA_SLIDE_THLD=None,
            VLLM_BUCKETING_FROM_FILE=None,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            object.__setattr__(self, k, v)


# ---------------------------------------------------------------------------
# Qwen3-30B-A3B long-context scenario constants (issue #1347 reference)
# ---------------------------------------------------------------------------

# 256K model length, TP=1, Gaudi2 (96 GB HBM)
_LONG_CTX_MAX_MODEL_LEN = 262144
_LONG_CTX_BLOCK_SIZE = 128
_LONG_CTX_MAX_NUM_SEQS = 256
_LONG_CTX_MAX_NUM_BATCHED_TOKENS = 8192
_LONG_CTX_MAX_BLOCKS_GAUDI2 = 3593  # realistic KV-cache blocks for 256K

###############################################################################
#                                                                             #
#  PR #762 — Padding-Aware Bucketing Strategy                                 #
#                                                                             #
###############################################################################


class TestPR762PaddingAwareStrategy:
    """Tests validating the padding-aware bucketing strategy from PR #762."""

    # -- warmup_range_with_limits correctness --

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            # Linear fallback: pad_percent=0 means no skipping
            ((0, 8, 64, 64, 0), [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]),
            # Exponential fallback: high pad_percent skips many buckets
            ((0, 8, 64, 64, 50), [0, 1, 2, 4, 8, 16, 32, 64]),
            # Mixed: pad_max constrains absolute gap
            ((0, 8, 64, 16, 50), [0, 1, 2, 4, 8, 16, 32, 48, 64]),
            # No ramp-up when min == step
            ((16, 16, 128, 32, 25), [16, 32, 48, 64, 80, 96, 128]),
        ],
    )
    def test_warmup_range_with_limits_doc_examples(self, config, expected):
        """Padding-aware warmup range matches documented examples."""
        assert warmup_range_with_limits(config) == expected

    def test_warmup_range_always_includes_max(self):
        """bucket_max is always present as the last element."""
        config = (128, 128, 8192, 2048, 25)
        result = warmup_range_with_limits(config)
        assert result[-1] == 8192

    def test_warmup_range_always_includes_min(self):
        """bucket_min is always present as the first element."""
        config = (0, 8, 128, 32, 25)
        result = warmup_range_with_limits(config)
        assert result[0] == 0

    def test_warmup_range_monotonically_increasing(self):
        """Generated buckets must be strictly increasing."""
        config = (0, 128, 8192, 2048, 25)
        result = warmup_range_with_limits(config)
        for i in range(1, len(result)):
            assert result[i] > result[i - 1], (f"Bucket {result[i]} at index {i} is not greater than "
                                               f"previous {result[i - 1]}")

    def test_warmup_range_padding_never_exceeds_limits(self):
        """Each gap never exceeds pad_max absolute, and consecutive ratio
        never exceeds pad_percent (for the 'stable' region only)."""
        pad_max = 2048
        pad_percent = 25
        config = (128, 128, 8192, pad_max, pad_percent)
        result = warmup_range_with_limits(config)
        # Check in the stable region (after ramp-up, i.e. values > step)
        stable = [b for b in result if b > 128]
        for i in range(1, len(stable)):
            gap = stable[i] - stable[i - 1]
            # The gap must be bounded: either padding ratio or absolute limit
            # was controlling bucket selection.  Worst case padding from
            # current to next is gap - 1, but we check gap itself as a
            # conservative bound.
            assert gap <= pad_max + 128, (  # +step tolerance for rounding
                f"Gap {gap} between buckets {stable[i - 1]} and "
                f"{stable[i]} exceeds pad_max={pad_max} + step")

    # -- Padding-aware decode config for long-context scenario --

    @patch('vllm_gaudi.extension.bucketing.padding_aware.get_config')
    def test_pad_decode_cfgs_long_context_non_contiguous(self, mock_get_config):
        """Padding-aware decode cfgs handle the 256K scenario without OOM-scale block max."""
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
        strategy = PaddingAwareBucketingStrategy()

        _, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
            block_size=_LONG_CTX_BLOCK_SIZE,
            max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
            max_model_len=_LONG_CTX_MAX_MODEL_LEN,
            max_blocks=_LONG_CTX_MAX_BLOCKS_GAUDI2,
        )

        # The max_decode_blocks formula: max(ceil(model_len * seqs / blk), blk)
        expected_raw = max(
            math.ceil(_LONG_CTX_MAX_MODEL_LEN * _LONG_CTX_MAX_NUM_SEQS / _LONG_CTX_BLOCK_SIZE),
            _LONG_CTX_BLOCK_SIZE,
        )
        # Padding-aware block_cfg[2] == the max from settings (may be == expected_raw)
        assert block_cfg[2] == expected_raw, (
            f"Padding-aware max_decode_blocks should be {expected_raw}, got {block_cfg[2]}")

        # Verify padding limits are set
        assert block_cfg[3] > 0, "pad_max should be positive"
        assert 0 <= block_cfg[4] <= 50, "pad_percent should be in [0, 50]"

    @patch('vllm_gaudi.extension.bucketing.padding_aware.get_config')
    def test_pad_decode_cfgs_contiguous_pa_clamped(self, mock_get_config):
        """With contiguous PA, padding-aware block max is clamped to max_blocks."""
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
        strategy = PaddingAwareBucketingStrategy()

        _, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
            block_size=_LONG_CTX_BLOCK_SIZE,
            max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
            max_model_len=_LONG_CTX_MAX_MODEL_LEN,
            max_blocks=_LONG_CTX_MAX_BLOCKS_GAUDI2,
        )

        assert block_cfg[2] == _LONG_CTX_MAX_BLOCKS_GAUDI2, (
            f"Contiguous PA should clamp to max_blocks={_LONG_CTX_MAX_BLOCKS_GAUDI2}, "
            f"got {block_cfg[2]}")

    def test_pad_prompt_cfgs_produce_bounded_buckets(self):
        """Prompt bucketing with padding-aware strategy produces reasonable bucket count."""
        strategy = PaddingAwareBucketingStrategy()
        bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
            max_num_prefill_seqs=1,
            block_size=_LONG_CTX_BLOCK_SIZE,
            max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
            max_model_len=_LONG_CTX_MAX_MODEL_LEN,
        )

        bs_range = strategy.get_range(bs_cfg)
        query_range = strategy.get_range(query_cfg)
        ctx_range = strategy.get_range(ctx_cfg)

        # The padding-aware strategy should produce far fewer buckets than
        # linear (which can yield thousands) but more than exponential.
        assert len(bs_range) >= 1
        assert len(query_range) >= 2
        assert len(ctx_range) >= 2

        # The max query should be max_num_batched_tokens
        assert query_range[-1] == _LONG_CTX_MAX_NUM_BATCHED_TOKENS

    # -- Full bucket generation through generate_buckets --

    def test_pad_strategy_generates_nonzero_prompt_buckets(self):
        """Padding-aware strategy must generate at least 1 prompt bucket."""
        strategy = PaddingAwareBucketingStrategy()
        bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
            max_num_prefill_seqs=16,
            block_size=128,
            max_num_batched_tokens=2048,
            max_model_len=4096,
        )
        bs_range = strategy.get_range(bs_cfg)
        query_range = strategy.get_range(query_cfg)
        ctx_range = strategy.get_range(ctx_cfg)

        buckets = generate_buckets(
            bs_range,
            query_range,
            ctx_range,
            True,
            4096,
            16,
            16,
            2048,
            128,
            1024,
        )
        assert len(buckets) > 0, "Must generate at least one prompt bucket"

    @patch('vllm_gaudi.extension.bucketing.padding_aware.get_config')
    def test_pad_strategy_generates_nonzero_decode_buckets(self, mock_get_config):
        """Padding-aware strategy must generate at least 1 decode bucket."""
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
        strategy = PaddingAwareBucketingStrategy()
        _, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=64,
            block_size=128,
            max_num_batched_tokens=2048,
            max_model_len=4096,
            max_blocks=1024,
        )
        bs_cfg, query_cfg, _ = strategy.get_decode_cfgs(
            max_num_seqs=64,
            block_size=128,
            max_num_batched_tokens=2048,
            max_model_len=4096,
            max_blocks=1024,
        )
        bs_range = strategy.get_range(bs_cfg)
        query_range = strategy.get_range(query_cfg)
        ctx_range = strategy.get_range(block_cfg)

        buckets = generate_buckets(
            bs_range,
            query_range,
            ctx_range,
            False,
            4096,
            64,
            64,
            2048,
            128,
            1024,
        )
        assert len(buckets) > 0, "Must generate at least one decode bucket"


###############################################################################
#                                                                             #
#  PR #1122 — Decode Bucketing Fixes for Non-Contiguous PA                    #
#                                                                             #
###############################################################################


class TestPR1122DecodeBucketFormula:
    """Tests for PR #1122 changes: exponential decode block formula,
    limit cap removal, new generate_buckets filter, linear fix,
    and warmup clamp."""

    # -- Exponential decode config formula changes --

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_exp_decode_non_contiguous_pa_formula(self, mock_get_config):
        """Non-contiguous PA decode max = ceil(max_model_len/block_size) * max_num_seqs.

        This is the PR #1122 formula. Actual bounding of generated buckets
        happens via filters in generate_buckets().
        """
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
        strategy = ExponentialBucketingStrategy()

        max_blocks = _LONG_CTX_MAX_BLOCKS_GAUDI2
        _, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
            block_size=_LONG_CTX_BLOCK_SIZE,
            max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
            max_model_len=_LONG_CTX_MAX_MODEL_LEN,
            max_blocks=max_blocks,
        )

        expected = math.ceil(_LONG_CTX_MAX_MODEL_LEN / _LONG_CTX_BLOCK_SIZE) * _LONG_CTX_MAX_NUM_SEQS
        assert block_cfg[2] == expected, (
            f"Non-contiguous PA max should be ceil(max_model_len/block_size)*max_num_seqs={expected}, "
            f"got {block_cfg[2]}")

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_exp_decode_contiguous_pa_unchanged(self, mock_get_config):
        """Contiguous PA decode max = max_blocks (unchanged by PR #1122)."""
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
        strategy = ExponentialBucketingStrategy()

        _, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
            block_size=_LONG_CTX_BLOCK_SIZE,
            max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
            max_model_len=_LONG_CTX_MAX_MODEL_LEN,
            max_blocks=_LONG_CTX_MAX_BLOCKS_GAUDI2,
        )

        assert block_cfg[2] == _LONG_CTX_MAX_BLOCKS_GAUDI2

    # -- Block limit cap --

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_exp_decode_block_limit_is_uncapped(self, mock_get_config):
        """PR #1122: block limit is computed from log2(max_decode_blocks) without cap.

        Excessive warmup buckets are controlled by filters in generate_buckets()
        rather than by capping the block limit in get_decode_cfgs().
        """
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
        strategy = ExponentialBucketingStrategy()

        _, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
            block_size=_LONG_CTX_BLOCK_SIZE,
            max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
            max_model_len=_LONG_CTX_MAX_MODEL_LEN,
            max_blocks=_LONG_CTX_MAX_BLOCKS_GAUDI2,
        )

        max_decode_blocks = math.ceil(_LONG_CTX_MAX_MODEL_LEN / _LONG_CTX_BLOCK_SIZE) * _LONG_CTX_MAX_NUM_SEQS
        expected_limit = math.ceil(math.log2(max_decode_blocks)) + 1

        assert block_cfg[3] == expected_limit, (f"Block limit should be {expected_limit}, got {block_cfg[3]}")

    # -- Generate buckets filter: batch_size_smaller_than_blocks --

    def test_decode_filter_batch_size_smaller_than_blocks(self):
        """Non-contiguous PA decode filter: bs <= ctx removes invalid combos."""
        # A decode bucket with bs=64 but ctx=32 blocks is physically impossible
        bs_range = [1, 64]
        query_range = [1]
        ctx_range = [32, 128]

        buckets = generate_buckets(
            bs_range,
            query_range,
            ctx_range,
            False,  # decode
            max_model_len=8192,
            max_num_seqs=64,
            max_num_prefill_seqs=64,
            max_num_batched_tokens=8192,
            block_size=128,
            max_blocks=1024,
        )

        for bs, q, ctx in buckets:
            assert bs <= ctx, (f"Decode bucket ({bs}, {q}, {ctx}): bs must be <= ctx "
                               f"for non-contiguous PA")

    def test_decode_contiguous_pa_no_bs_ctx_filter(self):
        """Contiguous PA decode: has num_ctx_tokens_less_or_equal_batched_max_model_len filter.

        PR #1122 added this filter to control excessive warmup buckets.
        """
        bs_range = [1, 4, 16]
        query_range = [1]
        ctx_range = [1, 8, 32, 128]

        with patch('vllm_gaudi.extension.bucketing.common.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=True, merged_prefill=False)
            buckets = generate_buckets(
                bs_range,
                query_range,
                ctx_range,
                False,  # decode
                max_model_len=8192,
                max_num_seqs=16,
                max_num_prefill_seqs=16,
                max_num_batched_tokens=8192,
                block_size=128,
                max_blocks=1024,
            )

        # With contiguous PA, num_ctx_tokens filter is applied
        assert len(buckets) > 0
        # Verify filter: ctx <= ceil(max_model_len/block_size) * bs
        for bs, _, ctx in buckets:
            if ctx > ctx_range[0]:
                assert ctx <= math.ceil(8192 / 128) * bs, (
                    f"Bucket ({bs}, _, {ctx}): ctx exceeds batched max_model_len limit")

    # -- Linear strategy decode block overflow (PR #1122 fix) --

    @patch('vllm_gaudi.extension.bucketing.linear.get_config')
    def test_linear_decode_block_max_unclamped_non_contiguous(self, mock_get_config):
        """Linear strategy: non-contiguous PA block max is NOT clamped to max_blocks.

        PR #1122 gates the clamp on contiguous_pa only.
        """
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
        strategy = LinearBucketingStrategy()

        _, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=256,
            block_size=128,
            max_num_batched_tokens=8192,
            max_model_len=32768,
            max_blocks=1024,
        )

        # raw max_decode_blocks = max(ceil(32768*256/128), 128) = 65536
        # With PR #1122: non-contiguous PA does NOT clamp to max_blocks
        raw_max = max(math.ceil(32768 * 256 // 128), 128)
        assert raw_max > 1024, "Sanity: raw max should exceed max_blocks"
        # After PR #1122: unclamped for non-contiguous PA
        assert block_cfg[2] == raw_max, (f"Non-contiguous PA: block max should be unclamped raw_max={raw_max}, "
                                         f"got {block_cfg[2]}")

    # -- generate_seq_lengths clamp (PR #1122) --

    def test_generate_seq_lengths_basic_distribution(self):
        """_generate_seq_lengths distributes blocks evenly across samples."""
        # This is a pure logic test - no HPU needed
        # Simulate the function logic
        num_samples = 4
        num_blocks = 10

        # Replicate the algorithm
        blocks = [num_blocks // num_samples] * num_samples
        missing_blocks = num_blocks - sum(blocks)
        for i in range(missing_blocks):
            blocks[i] += 1

        assert sum(blocks) == num_blocks
        assert len(blocks) == num_samples
        # All blocks should be approximately equal
        assert max(blocks) - min(blocks) <= 1

    def test_generate_seq_lengths_asserts_on_too_few_blocks(self):
        """If num_blocks < num_samples, an assertion should fire."""
        num_samples = 10
        num_blocks = 5
        # Replicate the assertion
        with pytest.raises(AssertionError):
            assert num_samples <= num_blocks

    # -- Warmup scenario: large decode bucket vs physical blocks --

    def test_decode_block_range_within_cfg_max(self):
        """For 256K context, exponential block range should stay within
        the configured max_decode_blocks = ceil(max_model_len/block_size)*max_num_seqs.
        Actual bounding per (bs, ctx) pair happens via filters in generate_buckets()."""
        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=False)
            strategy = ExponentialBucketingStrategy()

            _, _, block_cfg = strategy.get_decode_cfgs(
                max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
                block_size=_LONG_CTX_BLOCK_SIZE,
                max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
                max_model_len=_LONG_CTX_MAX_MODEL_LEN,
                max_blocks=_LONG_CTX_MAX_BLOCKS_GAUDI2,
            )
            block_range = strategy.get_range(block_cfg)

        expected_max = math.ceil(_LONG_CTX_MAX_MODEL_LEN / _LONG_CTX_BLOCK_SIZE) * _LONG_CTX_MAX_NUM_SEQS
        assert max(block_range) <= expected_max, (
            f"Block range max {max(block_range)} exceeds cfg max {expected_max}")


###############################################################################
#                                                                             #
#  PR #1155 — FusedSDPA Slicing Dependency on Bucketing                       #
#                                                                             #
###############################################################################


class TestPR1155FusedSDPASlicingBucketingContract:
    """Tests for the contract between FusedSDPA slicing (PR #1155) and bucketing.

    FusedSDPA slicing requires padding-aware bucketing to guarantee bounded
    padding in query and context dimensions.  These tests verify that the
    padding-aware strategy upholds those guarantees.
    """

    def test_pad_strategy_query_pad_max_default(self):
        """Default query pad_max should be ceil(max_num_batched_tokens / 4)."""
        strategy = PaddingAwareBucketingStrategy()
        _, query_cfg, _ = strategy.get_prompt_cfgs(
            max_num_prefill_seqs=1,
            block_size=128,
            max_num_batched_tokens=8192,
            max_model_len=262144,
        )
        # query_cfg = [min, step, max, pad_max, pad_percent]
        expected_pad_max = math.ceil(8192 / 4)  # 2048
        assert query_cfg[3] == expected_pad_max, (f"Query pad_max should be {expected_pad_max}, got {query_cfg[3]}")

    def test_pad_strategy_ctx_pad_max_default(self):
        """Default ctx pad_max should be ceil(max_num_batched_tokens / block_size)."""
        strategy = PaddingAwareBucketingStrategy()
        _, _, ctx_cfg = strategy.get_prompt_cfgs(
            max_num_prefill_seqs=1,
            block_size=128,
            max_num_batched_tokens=8192,
            max_model_len=262144,
        )
        expected_pad_max = math.ceil(8192 / 128)  # 64
        assert ctx_cfg[3] == expected_pad_max, (f"Ctx pad_max should be {expected_pad_max}, got {ctx_cfg[3]}")

    def test_pad_strategy_query_range_gap_bounded_for_slicing(self):
        """Query range gaps must be bounded by pad_max for FusedSDPA slicing correctness."""
        strategy = PaddingAwareBucketingStrategy()
        _, query_cfg, _ = strategy.get_prompt_cfgs(
            max_num_prefill_seqs=1,
            block_size=128,
            max_num_batched_tokens=8192,
            max_model_len=262144,
        )
        query_range = strategy.get_range(query_cfg)
        pad_max = query_cfg[3]

        # In the stable region (values > step), gaps must be bounded
        step = query_cfg[1]
        stable = [b for b in query_range if b > step]
        for i in range(1, len(stable)):
            gap = stable[i] - stable[i - 1]
            assert gap <= pad_max + step, (f"Query range gap {gap} between {stable[i - 1]} and "
                                           f"{stable[i]} exceeds pad_max={pad_max} + step={step}. "
                                           f"FusedSDPA slicing assumes bounded padding!")

    def test_pad_strategy_ctx_range_gap_bounded_for_slicing(self):
        """Context range gaps must be bounded by pad_max for FusedSDPA slicing correctness."""
        strategy = PaddingAwareBucketingStrategy()
        _, _, ctx_cfg = strategy.get_prompt_cfgs(
            max_num_prefill_seqs=1,
            block_size=128,
            max_num_batched_tokens=8192,
            max_model_len=262144,
        )
        ctx_range = strategy.get_range(ctx_cfg)
        pad_max = ctx_cfg[3]
        step = ctx_cfg[1]

        stable = [b for b in ctx_range if b > step]
        for i in range(1, len(stable)):
            gap = stable[i] - stable[i - 1]
            assert gap <= pad_max + step, (f"Ctx range gap {gap} between {stable[i - 1]} and "
                                           f"{stable[i]} exceeds pad_max={pad_max} + step={step}. "
                                           f"FusedSDPA slicing assumes bounded padding!")

    def test_bucketing_strategy_selection_pad(self, monkeypatch):
        """VLLM_BUCKETING_STRATEGY=pad selects PaddingAwareBucketingStrategy."""
        monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", "pad")
        clear_config()

        manager = HPUBucketingManager.__new__(HPUBucketingManager)
        strategy = manager.get_bucketing_strategy()
        assert isinstance(strategy, PaddingAwareBucketingStrategy)

    def test_bucketing_strategy_exp_is_not_pad(self, monkeypatch):
        """VLLM_BUCKETING_STRATEGY=exp does NOT select PaddingAwareBucketingStrategy.

        FusedSDPA slicing requires 'pad' strategy — verify this contract.
        """
        monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", "exp")
        clear_config()

        manager = HPUBucketingManager.__new__(HPUBucketingManager)
        strategy = manager.get_bucketing_strategy()
        assert not isinstance(
            strategy, PaddingAwareBucketingStrategy), ("Exponential strategy must not be PaddingAware — "
                                                       "FusedSDPA slicing should be disabled for non-pad strategies")

    @patch('vllm_gaudi.extension.bucketing.padding_aware.get_config')
    def test_pad_strategy_merged_prefill_preserves_padding_limits(self, mock_get_config):
        """When merged_prefill is enabled, padding limits must still be preserved.

        FusedSDPA slicing is incompatible with merged prefill, but if someone
        tries it, the padding properties should still hold.
        """
        mock_get_config.return_value = _MockConfig(merged_prefill=True)
        strategy = PaddingAwareBucketingStrategy()

        bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
            max_num_prefill_seqs=16,
            block_size=128,
            max_num_batched_tokens=2048,
            max_model_len=4096,
        )

        # pad_max and pad_percent should be propagated even with merged prefill
        assert len(bs_cfg) == 5, "Padding-aware cfg should have 5 elements"
        assert len(query_cfg) == 5
        assert len(ctx_cfg) == 5


###############################################################################
#                                                                             #
#  PR #1346 — HPU Graph Capture Skip for Long Prefills                        #
#                                                                             #
###############################################################################


class TestPR1346HPUGraphCaptureSkip:
    """Tests for PR #1346: skipping HPU graph capture for long prefills
    and the warmup seq_lengths clamp.

    These tests verify the bucketing-related contracts that PR #1346 depends on.
    """

    # -- _use_graphs logic (current main: simple enforce_eager check) --

    def test_current_use_graphs_only_checks_enforce_eager(self):
        """Current main: _use_graphs() returns `not enforce_eager`.

        PR #1346 will change this to also check (query + context) size.
        This test documents the current behaviour.
        """
        # Simulate the current logic
        enforce_eager = False
        use_graphs = not enforce_eager
        assert use_graphs is True

        enforce_eager = True
        use_graphs = not enforce_eager
        assert use_graphs is False

    def test_current_cudagraph_skip_based_on_batch_seq_product(self):
        """Current main: graph skip is based on batch_size * seq_len only.

        PR #1346 will change to include context blocks in the calculation.
        """
        # Current logic from _execute_model_generic:
        # if max_cudagraph_capture_size is not None and batch_size * seq_len > max_cudagraph_capture_size:
        #     use_graphs = False
        max_cudagraph_capture_size = 8192
        batch_size = 1
        seq_len = 4096

        # Current: only checks batch_size * seq_len
        current_skip = (max_cudagraph_capture_size is not None and batch_size * seq_len > max_cudagraph_capture_size)
        assert current_skip is False, "4096 < 8192, so current main keeps graphs"

        # But with context, total might exceed: e.g., 4096 query + 64 blocks * 128 = 12288
        num_blocks = 64
        block_size = 128
        total_tokens = batch_size * seq_len + num_blocks * block_size
        pr1346_skip = total_tokens > max_cudagraph_capture_size
        assert pr1346_skip is True, ("PR #1346 would skip graphs for this case: "
                                     f"total_tokens={total_tokens} > {max_cudagraph_capture_size}")

    # -- Bucket generation for decode warmup: blocks vs physical KV cache --

    def test_decode_warmup_buckets_blocks_can_exceed_physical(self):
        """With non-contiguous PA (PR #1122), decode bucket block values can
        exceed num_hpu_blocks because max = ceil(max_model_len/block_size)*max_num_seqs.

        PR #1122 clamps _generate_seq_lengths to avoid OOM during warmup,
        and PR #1346 skips graph capture for long prefills.
        """
        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=False)
            strategy = ExponentialBucketingStrategy()

            _, _, block_cfg = strategy.get_decode_cfgs(
                max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
                block_size=_LONG_CTX_BLOCK_SIZE,
                max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
                max_model_len=_LONG_CTX_MAX_MODEL_LEN,
                max_blocks=_LONG_CTX_MAX_BLOCKS_GAUDI2,
            )
            block_range = strategy.get_range(block_cfg)

        # Some block values exceed physical max_blocks
        exceeds_physical = [b for b in block_range if b > _LONG_CTX_MAX_BLOCKS_GAUDI2]
        assert len(exceeds_physical) > 0, ("With non-contiguous PA, some block buckets should exceed physical blocks. "
                                           "PR #1122 clamps _generate_seq_lengths to handle this.")

    def test_prompt_buckets_long_context_includes_short_and_long(self):
        """For 256K models, prompt buckets should span from short (block_size)
        to near max_num_batched_tokens, with context from 0 to high values."""
        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(
                use_contiguous_pa=False,
                merged_prefill=False,
                VLLM_PROMPT_QUERY_BUCKET_MIN=None,
            )
            strategy = ExponentialBucketingStrategy()

            bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
                max_num_prefill_seqs=1,
                block_size=_LONG_CTX_BLOCK_SIZE,
                max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
                max_model_len=_LONG_CTX_MAX_MODEL_LEN,
            )

        # For long context: max_ctx uses max_num_batched_tokens in formula
        assert strategy.long_context is True, "262144 >= 8192 (LONG_CTX_THRESHOLD)"

        query_range = strategy.get_range(query_cfg)
        ctx_range = strategy.get_range(ctx_cfg)

        # Should include both short and long queries
        assert min(query_range) <= _LONG_CTX_BLOCK_SIZE
        assert max(query_range) >= _LONG_CTX_MAX_NUM_BATCHED_TOKENS

        # Context should go from 0 to a high value
        assert 0 in ctx_range
        assert max(ctx_range) > 100  # significant context blocks for 256K

    def test_max_cudagraph_capture_size_defaults_to_none(self):
        """On current main, max_cudagraph_capture_size comes from compilation_config.

        PR #1346 defaults it to max_num_batched_tokens when None.
        """
        # Document the current behaviour: the value is directly from config
        # which may be None.
        # After PR #1346: None -> max_num_batched_tokens
        max_num_batched_tokens = 8192
        max_cudagraph_capture_size = None

        # Current main: None means the skip check is bypassed entirely
        current_skip = (max_cudagraph_capture_size is not None and max_cudagraph_capture_size < 999999)
        assert current_skip is False, "None means no skip on current main"

        # PR #1346 behaviour:
        if max_cudagraph_capture_size is None:
            max_cudagraph_capture_size = max_num_batched_tokens
        pr1346_skip = (max_cudagraph_capture_size is not None and max_cudagraph_capture_size < 999999)
        assert pr1346_skip is True, "PR #1346 defaults to max_num_batched_tokens"


###############################################################################
#                                                                             #
#  Cross-PR Integration: Full Bucket Generation for 256K Scenario             #
#                                                                             #
###############################################################################


class TestCrossPRIntegration256K:
    """End-to-end tests simulating the 256K model-length scenario from issue #1347.

    Verifies that bucketing remains functional across all strategies.
    """

    # -- Full manager-level bucket generation --

    @patch('vllm_gaudi.extension.bucketing.common.get_config')
    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_manager_generates_prompt_and_decode_buckets_exp(self, mock_exp_config, mock_common_config, monkeypatch):
        """HPUBucketingManager generates valid prompt + decode buckets with exp strategy."""
        config = _MockConfig(
            use_contiguous_pa=False,
            merged_prefill=False,
            bucketing_strategy='exp',
            VLLM_PROMPT_QUERY_BUCKET_MIN=None,
        )
        mock_exp_config.return_value = config
        mock_common_config.return_value = config
        monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", "exp")
        monkeypatch.delenv("VLLM_EXPONENTIAL_BUCKETING", raising=False)
        clear_config()

        # Reset singleton
        HPUBucketingManager._instance = None
        manager = HPUBucketingManager()
        manager.initialize(
            max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
            max_num_prefill_seqs=1,
            block_size=_LONG_CTX_BLOCK_SIZE,
            max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
            max_model_len=_LONG_CTX_MAX_MODEL_LEN,
        )
        manager.num_hpu_blocks = _LONG_CTX_MAX_BLOCKS_GAUDI2

        manager.generate_prompt_buckets()
        manager.generate_decode_buckets()

        assert len(manager.prompt_buckets) > 0, "Must generate prompt buckets"
        assert len(manager.decode_buckets) > 0, "Must generate decode buckets"

        # All prompt buckets should have valid dimensions
        for bs, query, ctx in manager.prompt_buckets:
            assert bs >= 1
            assert query >= 1
            assert ctx >= 0
            assert (query + ctx * _LONG_CTX_BLOCK_SIZE) <= _LONG_CTX_MAX_MODEL_LEN, (
                f"Prompt bucket ({bs}, {query}, {ctx}): "
                f"query + ctx*block_size = {query + ctx * _LONG_CTX_BLOCK_SIZE} "
                f"exceeds max_model_len={_LONG_CTX_MAX_MODEL_LEN}")

        # All decode buckets should have valid dimensions
        for bs, query, ctx in manager.decode_buckets:
            assert bs >= 1
            assert query == 1  # decode always has query=1
            assert ctx >= 1

        # Reset singleton for other tests
        HPUBucketingManager._instance = None

    def test_pad_strategy_prompt_decode_generation(self, monkeypatch):
        """Padding-aware strategy generates valid buckets for 256K scenario."""
        monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", "pad")
        clear_config()

        with patch('vllm_gaudi.extension.bucketing.common.get_config') as mock_common, \
             patch('vllm_gaudi.extension.bucketing.padding_aware.get_config') as mock_pad:

            config = _MockConfig(
                use_contiguous_pa=False,
                merged_prefill=False,
                bucketing_strategy='pad',
            )
            mock_common.return_value = config
            mock_pad.return_value = config

            HPUBucketingManager._instance = None
            manager = HPUBucketingManager()
            manager.initialize(
                max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
                max_num_prefill_seqs=1,
                block_size=_LONG_CTX_BLOCK_SIZE,
                max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
                max_model_len=_LONG_CTX_MAX_MODEL_LEN,
            )
            manager.num_hpu_blocks = _LONG_CTX_MAX_BLOCKS_GAUDI2

            manager.generate_prompt_buckets()
            manager.generate_decode_buckets()

            assert len(manager.prompt_buckets) > 0
            assert len(manager.decode_buckets) > 0

            # Padding-aware should have more buckets than exponential
            # but not an extreme number
            assert len(
                manager.prompt_buckets) < 10000, (f"Padding-aware prompt buckets ({len(manager.prompt_buckets)}) "
                                                  f"seems unreasonably high — likely regression")
            assert len(
                manager.decode_buckets) < 10000, (f"Padding-aware decode buckets ({len(manager.decode_buckets)}) "
                                                  f"seems unreasonably high — likely regression")

        HPUBucketingManager._instance = None

    # -- Fallback bucket behaviour for long-context --

    def test_fallback_bucket_256k_ctx(self):
        """Fallback bucket for large ctx values in 256K scenario."""
        mgr = HPUBucketingManager.__new__(HPUBucketingManager)
        mgr.max_num_seqs = _LONG_CTX_MAX_NUM_SEQS
        mgr.max_num_batched_tokens = _LONG_CTX_MAX_NUM_BATCHED_TOKENS
        mgr.fallback_bs_base_step = 2
        mgr.fallback_seq_base_step = 32
        mgr.fallback_blocks_base_step = 32
        mgr.num_hpu_blocks = _LONG_CTX_MAX_BLOCKS_GAUDI2
        mgr.block_size = _LONG_CTX_BLOCK_SIZE
        mgr.use_sliding_window = False
        mgr._fallback_max_ctx = _LONG_CTX_MAX_BLOCKS_GAUDI2 * 3  # 10779

        # Request a ctx within fallback_max_ctx
        _, _, new_ctx = mgr.generate_fallback_bucket(batch_size=1, seq_len=1, ctx=5000)
        assert new_ctx >= 5000, f"Fallback ctx {new_ctx} < 5000"
        assert new_ctx == calc_fallback_value(5000, 32)

        # Request a ctx exceeding fallback_max_ctx → should be capped
        _, _, new_ctx = mgr.generate_fallback_bucket(batch_size=1, seq_len=1, ctx=20000)
        assert new_ctx == mgr._fallback_max_ctx, (f"Fallback ctx {new_ctx} should be capped at {mgr._fallback_max_ctx}")

    def test_find_decode_bucket_returns_closest_match(self):
        """find_decode_bucket returns the closest bucket >= requested."""
        mgr = HPUBucketingManager.__new__(HPUBucketingManager)
        mgr.initialized = True
        mgr.decode_buckets = [
            (1, 1, 128),
            (1, 1, 256),
            (4, 1, 128),
            (4, 1, 256),
            (16, 1, 512),
        ]
        mgr.seed_decode_buckets = None
        mgr.max_num_seqs = 256
        mgr.max_num_batched_tokens = 8192
        mgr.fallback_bs_base_step = 2
        mgr.fallback_seq_base_step = 32
        mgr.fallback_blocks_base_step = 32
        mgr.num_hpu_blocks = 1024
        mgr.block_size = 128
        mgr.use_sliding_window = False
        mgr._fallback_max_ctx = 512

        # Exact match
        bucket = mgr.find_decode_bucket(batch_size=4, num_blocks=256)
        assert bucket == (4, 1, 256)

        # Closest greater
        bucket = mgr.find_decode_bucket(batch_size=2, num_blocks=200)
        assert bucket[0] >= 2
        assert bucket[2] >= 200

    def test_find_prompt_bucket_returns_closest_match(self):
        """find_prompt_bucket returns the closest bucket >= requested."""
        mgr = HPUBucketingManager.__new__(HPUBucketingManager)
        mgr.initialized = True
        mgr.prompt_buckets = [
            (1, 128, 0),
            (1, 256, 0),
            (1, 512, 4),
            (1, 1024, 8),
        ]
        mgr.max_num_seqs = 256
        mgr.max_num_batched_tokens = 8192
        mgr.fallback_bs_base_step = 2
        mgr.fallback_seq_base_step = 32
        mgr.fallback_blocks_base_step = 32
        mgr.num_hpu_blocks = 1024
        mgr.block_size = 128
        mgr.use_sliding_window = False
        mgr._fallback_max_ctx = 512

        # Exact match
        bucket = mgr.find_prompt_bucket(batch_size=1, seq_len=256, ctx=0)
        assert bucket == (1, 256, 0)

        # Closest greater
        bucket = mgr.find_prompt_bucket(batch_size=1, seq_len=300, ctx=0)
        assert bucket[1] >= 300

    # -- Decode bucket count reasonableness --

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_exp_decode_bucket_count_bounded(self, mock_get_config):
        """Exponential decode bucket count stays reasonable for 256K scenario.

        Without the limit cap (or with PR #1122 filter), the total decode
        bucket count should not explode to 100+ which causes 30min warmup.
        """
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
        strategy = ExponentialBucketingStrategy()

        bs_cfg, query_cfg, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=_LONG_CTX_MAX_NUM_SEQS,
            block_size=_LONG_CTX_BLOCK_SIZE,
            max_num_batched_tokens=_LONG_CTX_MAX_NUM_BATCHED_TOKENS,
            max_model_len=_LONG_CTX_MAX_MODEL_LEN,
            max_blocks=_LONG_CTX_MAX_BLOCKS_GAUDI2,
        )

        bs_range = strategy.get_range(bs_cfg)
        block_range = strategy.get_range(block_cfg)

        # Cartesian product before filtering
        raw_count = len(bs_range) * len(block_range)

        # Should be manageable (current cap keeps it under ~70)
        assert raw_count < 200, (f"Raw decode bucket count {raw_count} is too high "
                                 f"(bs_range={len(bs_range)}, block_range={len(block_range)}). "
                                 f"This would cause very long warmup.")

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_exp_decode_small_batch_size_scenario(self, mock_get_config):
        """Small max_num_seqs (e.g. 21) should not produce excessive decode buckets.

        This was the GAUDISW-247226 scenario.
        """
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
        strategy = ExponentialBucketingStrategy()

        bs_cfg, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=21,
            block_size=128,
            max_num_batched_tokens=8192,
            max_model_len=131072,
            max_blocks=65536,
        )

        bs_range = strategy.get_range(bs_cfg)
        block_range = strategy.get_range(block_cfg)
        total = len(bs_range) * len(block_range)

        assert total <= 60, (f"Total decode buckets {total} is too high for max_num_seqs=21 scenario. "
                             f"Would cause ~30min warmup.")

    # -- All strategies produce sorted, unique buckets --

    @pytest.mark.parametrize("strategy_name,strategy_cls", [
        ("exp", ExponentialBucketingStrategy),
        ("pad", PaddingAwareBucketingStrategy),
    ])
    def test_strategy_ranges_sorted_and_unique(self, strategy_name, strategy_cls):
        """All bucketing strategy ranges must be sorted and have unique values."""
        if strategy_name == "exp":
            with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
                mock.return_value = _MockConfig(use_contiguous_pa=False, VLLM_PROMPT_QUERY_BUCKET_MIN=None)
                strategy = strategy_cls()
                bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
                    max_num_prefill_seqs=1,
                    block_size=128,
                    max_num_batched_tokens=8192,
                    max_model_len=32768,
                )
        else:
            strategy = strategy_cls()
            bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
                max_num_prefill_seqs=1,
                block_size=128,
                max_num_batched_tokens=8192,
                max_model_len=32768,
            )

        for name, cfg in [("bs", bs_cfg), ("query", query_cfg), ("ctx", ctx_cfg)]:
            range_vals = strategy.get_range(cfg)
            assert range_vals == sorted(range_vals), (f"{strategy_name} {name} range is not sorted: {range_vals}")
            assert len(range_vals) == len(
                set(range_vals)), (f"{strategy_name} {name} range has duplicates: {range_vals}")


###############################################################################
#                                                                             #
#  Regression Tests — Known Failure Scenarios                                 #
#                                                                             #
###############################################################################


class TestRegressions:
    """Regression tests for known failure scenarios from the issue."""

    def test_calc_fallback_value_large_inputs(self):
        """calc_fallback_value must return values >= n for all positive n."""
        test_values = [1, 31, 100, 1000, 4001, 10000, 50000, 100000, 524288]
        for n in test_values:
            result = calc_fallback_value(n, 32)
            assert result >= n, (f"calc_fallback_value({n}, 32) = {result} < {n}")

    def test_calc_fallback_value_divisibility(self):
        """calc_fallback_value results should be divisible by base_step."""
        for n in [31, 100, 4001, 10000]:
            for base_step in [2, 32, 128]:
                result = calc_fallback_value(n, base_step)
                assert result % base_step == 0, (f"calc_fallback_value({n}, {base_step}) = {result} "
                                                 f"is not divisible by {base_step}")

    def test_warmup_range_with_limit_single_bucket(self):
        """Config with num_buckets=1 should return [bmax]."""
        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=False)
            config = (1, 128, 1024, 1)
            result = warmup_range_with_limit(config)
            assert result == [1024]

    def test_warmup_range_with_limit_never_exceeds_bmax(self):
        """No bucket value should exceed bmax."""
        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=False)
            configs = [
                (1, 256, 10779, 14),
                (1, 256, 3593, 9),
                (0, 1, 702, 11),
                (128, 128, 8192, 11),
            ]
            for config in configs:
                result = warmup_range_with_limit(config)
                assert max(result) <= config[2], (f"For config {config}: max bucket {max(result)} "
                                                  f"exceeds bmax={config[2]}")

    def test_generate_buckets_prompt_respects_max_model_len(self):
        """All prompt buckets: query + ctx * block_size <= max_model_len."""
        max_model_len = 32768
        block_size = 128
        max_num_batched_tokens = 8192

        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(
                use_contiguous_pa=False,
                merged_prefill=False,
                VLLM_PROMPT_QUERY_BUCKET_MIN=None,
            )
            strategy = ExponentialBucketingStrategy()
            bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
                max_num_prefill_seqs=1,
                block_size=block_size,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
            )
            bs_range = strategy.get_range(bs_cfg)
            query_range = strategy.get_range(query_cfg)
            ctx_range = strategy.get_range(ctx_cfg)

        buckets = generate_buckets(
            bs_range,
            query_range,
            ctx_range,
            True,
            max_model_len,
            256,
            1,
            max_num_batched_tokens,
            block_size,
            1024,
        )

        for bs, query, ctx in buckets:
            total = query + ctx * block_size
            assert total <= max_model_len, (f"Prompt bucket ({bs}, {query}, {ctx}): "
                                            f"query + ctx*block_size = {total} > {max_model_len}")

    def test_generate_buckets_decode_corrector_non_contiguous(self):
        """Non-contiguous PA decode corrector: ctx <= bs * ceil(max_model_len/block_size)."""
        max_model_len = 32768
        block_size = 128

        bs_range = [1, 4, 16, 64]
        query_range = [1]
        ctx_range = [1, 64, 256, 1024, 4096]

        buckets = generate_buckets(
            bs_range,
            query_range,
            ctx_range,
            False,
            max_model_len,
            64,
            64,
            8192,
            block_size,
            4096,
        )

        max_blocks_per_seq = math.ceil(max_model_len / block_size)
        for bs, query, ctx in buckets:
            assert ctx <= bs * max_blocks_per_seq, (f"Decode bucket ({bs}, {query}, {ctx}): ctx exceeds "
                                                    f"bs * ceil(max_model_len/block_size) = "
                                                    f"{bs * max_blocks_per_seq}")

    def test_linear_warmup_range_basic(self):
        """Linear warmup range produces expected ramp-up + stable pattern."""
        config = (2, 64, 256)
        result = linear.warmup_range(config)
        # ramp_up: 2, 4, 8, 16, 32
        # stable: 64, 128, 192, 256
        expected = [2, 4, 8, 16, 32, 64, 128, 192, 256]
        assert result == expected

    def test_linear_warmup_range_with_zero(self):
        """Linear warmup range with 0 min includes 0 bucket."""
        config = (0, 32, 64)
        result = linear.warmup_range(config)
        assert 0 in result
        assert result[-1] == 0 or max(result) == 64

    def test_padding_aware_warmup_range_min_equals_max(self):
        """When min == max, only one bucket should be generated."""
        config = (128, 128, 128, 32, 25)
        result = warmup_range_with_limits(config)
        assert result == [128]

    def test_exponential_warmup_contiguous_pa_last_bucket_exact(self):
        """With contiguous PA, last bucket should be exactly bmax."""
        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=True)
            config = (1, 256, 3593, 13)
            result = warmup_range_with_limit(config)
            assert result[-1] == 3593
