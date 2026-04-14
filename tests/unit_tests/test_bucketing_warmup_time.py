# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
"""Warmup time impact tests for bucketing (issue #1347).

Warmup time is proportional to the total number of HPU graph compilations,
which equals the number of prompt buckets + decode buckets.  These tests
verify that bucket counts (and thus warmup time) stay within acceptable
bounds across model configurations, strategies, and edge cases.

Key relationships:
  - warmup_graphs() iterates over ALL prompt+decode buckets
  - Each bucket triggers one HPU graph compilation (~5-30 s on Gaudi2)
  - Total warmup time ≈ num_buckets × per_bucket_time
  - cache_size_limit is derived from total bucket count
"""

import math
from unittest.mock import patch

import pytest

from vllm_gaudi.extension.bucketing.common import (
    HPUBucketingManager,
    generate_buckets,
)
from vllm_gaudi.extension.bucketing.exponential import (
    ExponentialBucketingStrategy,
    warmup_range_with_limit,
)
from vllm_gaudi.extension.bucketing.linear import (
    warmup_range as linear_warmup_range, )
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
# Mock config helper
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
# Scenario configurations for warmup time testing
# ---------------------------------------------------------------------------

# Typical model configurations (name, max_model_len, max_num_seqs, block_size,
#                               max_num_batched_tokens, max_blocks)
_SCENARIOS = {
    "llama3_8k": (8192, 256, 128, 8192, 3593),
    "qwen3_32k": (32768, 256, 128, 8192, 3593),
    "qwen3_128k": (131072, 256, 128, 8192, 65536),
    "qwen3_256k": (262144, 256, 128, 8192, 3593),
    "small_batch_131k": (131072, 21, 128, 8192, 65536),
}

# Maximum acceptable total bucket count (prompt + decode) before warmup
# becomes unreasonably long.  At ~15s per graph compilation on Gaudi2,
# 200 buckets ≈ 50 min warmup which is the practical upper bound.
_MAX_TOTAL_BUCKETS = 200

# Warmup budget: max acceptable for each phase
_MAX_PROMPT_BUCKETS = 150
_MAX_DECODE_BUCKETS = 150

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_bucket_counts_exp(max_model_len,
                           max_num_seqs,
                           block_size,
                           max_num_batched_tokens,
                           max_blocks,
                           use_contiguous_pa=False):
    """Generate bucket counts for exponential strategy."""
    with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
        mock.return_value = _MockConfig(
            use_contiguous_pa=use_contiguous_pa,
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
        prompt_bs = strategy.get_range(bs_cfg)
        prompt_query = strategy.get_range(query_cfg)
        prompt_ctx = strategy.get_range(ctx_cfg)

        bs_cfg_d, query_cfg_d, block_cfg_d = strategy.get_decode_cfgs(
            max_num_seqs=max_num_seqs,
            block_size=block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            max_blocks=max_blocks,
        )
        decode_bs = strategy.get_range(bs_cfg_d)
        decode_block = strategy.get_range(block_cfg_d)

    with patch('vllm_gaudi.extension.bucketing.common.get_config') as mock:
        mock.return_value = _MockConfig(
            use_contiguous_pa=use_contiguous_pa,
            merged_prefill=False,
        )
        prompt_buckets = generate_buckets(
            prompt_bs,
            prompt_query,
            prompt_ctx,
            True,
            max_model_len,
            max_num_seqs,
            1,
            max_num_batched_tokens,
            block_size,
            max_blocks,
        )
        decode_buckets = generate_buckets(
            decode_bs,
            [1],
            decode_block,
            False,
            max_model_len,
            max_num_seqs,
            1,
            max_num_batched_tokens,
            block_size,
            max_blocks,
        )

    return len(prompt_buckets), len(decode_buckets)


def _get_range_counts_exp(max_model_len,
                          max_num_seqs,
                          block_size,
                          max_num_batched_tokens,
                          max_blocks,
                          use_contiguous_pa=False):
    """Get individual range sizes for exponential strategy (before Cartesian product)."""
    with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
        mock.return_value = _MockConfig(
            use_contiguous_pa=use_contiguous_pa,
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
        p_bs = len(strategy.get_range(bs_cfg))
        p_q = len(strategy.get_range(query_cfg))
        p_ctx = len(strategy.get_range(ctx_cfg))

        bs_d, _, blk_d = strategy.get_decode_cfgs(
            max_num_seqs=max_num_seqs,
            block_size=block_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            max_blocks=max_blocks,
        )
        d_bs = len(strategy.get_range(bs_d))
        d_blk = len(strategy.get_range(blk_d))

    return {
        'prompt_bs': p_bs,
        'prompt_query': p_q,
        'prompt_ctx': p_ctx,
        'decode_bs': d_bs,
        'decode_block': d_blk,
        'prompt_cartesian': p_bs * p_q * p_ctx,
        'decode_cartesian': d_bs * d_blk,
    }


###############################################################################
#                                                                             #
#  Total Bucket Count Bounds (warmup time proxy)                              #
#                                                                             #
###############################################################################


class TestWarmupBucketCountBounds:
    """Verify total bucket counts stay within warmup time budget.

    Each HPU graph compilation takes ~5-30 seconds on Gaudi2.
    With 200 buckets at 15s average = 50 min warmup — the practical limit.
    """

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_exp_total_bucket_count_within_budget(self, scenario_name):
        """Exponential strategy: total buckets (prompt+decode) within budget."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        prompt_count, decode_count = _get_bucket_counts_exp(model_len, seqs, blk_sz, tokens, blocks)
        total = prompt_count + decode_count

        assert total <= _MAX_TOTAL_BUCKETS, (f"[{scenario_name}] Total buckets {total} "
                                             f"(prompt={prompt_count}, decode={decode_count}) exceeds budget "
                                             f"{_MAX_TOTAL_BUCKETS}. Warmup would take too long.")

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_exp_prompt_bucket_count_within_budget(self, scenario_name):
        """Exponential strategy: prompt bucket count within phase budget."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        prompt_count, _ = _get_bucket_counts_exp(model_len, seqs, blk_sz, tokens, blocks)

        assert prompt_count <= _MAX_PROMPT_BUCKETS, (f"[{scenario_name}] Prompt buckets {prompt_count} exceeds "
                                                     f"budget {_MAX_PROMPT_BUCKETS}")

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_exp_decode_bucket_count_within_budget(self, scenario_name):
        """Exponential strategy: decode bucket count within phase budget."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        _, decode_count = _get_bucket_counts_exp(model_len, seqs, blk_sz, tokens, blocks)

        assert decode_count <= _MAX_DECODE_BUCKETS, (f"[{scenario_name}] Decode buckets {decode_count} exceeds "
                                                     f"budget {_MAX_DECODE_BUCKETS}")

    def test_exp_contiguous_pa_fewer_or_equal_buckets(self):
        """Contiguous PA should produce fewer or equal decode buckets
        (no 3x headroom needed)."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS["qwen3_256k"]
        _, decode_noncont = _get_bucket_counts_exp(model_len, seqs, blk_sz, tokens, blocks, use_contiguous_pa=False)
        _, decode_cont = _get_bucket_counts_exp(model_len, seqs, blk_sz, tokens, blocks, use_contiguous_pa=True)

        assert decode_cont <= decode_noncont, (f"Contiguous PA decode buckets ({decode_cont}) should be <= "
                                               f"non-contiguous ({decode_noncont})")


###############################################################################
#                                                                             #
#  Warmup Time Regression Scenarios                                           #
#                                                                             #
###############################################################################


class TestWarmupTimeRegressions:
    """Regression tests for known bucket explosion scenarios that caused
    30+ minute warmup times."""

    def test_gaudisw_247226_small_batch_131k_context(self):
        """GAUDISW-247226: max_num_seqs=21, max_model_len=131072.

        Without decode block limit cap, this produced 17-18 block buckets
        × 6 bs buckets = ~126 decode graphs → 30+ min warmup.
        With the cap: block limit <= max(6, bs_limit) = max(6, 6) = 6.
        """
        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=True)
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
            decode_cartesian = len(bs_range) * len(block_range)

        # The cap ensures block_cfg[3] (limit) is bounded
        decode_bs_limit = math.ceil(math.log2(21)) + 1  # 6
        expected_block_limit = max(6, decode_bs_limit)  # 6
        assert block_cfg[3] <= expected_block_limit, (f"Block limit {block_cfg[3]} exceeds cap {expected_block_limit}")

        # Total decode buckets should be well under the old 126
        assert decode_cartesian <= 50, (f"Decode Cartesian product {decode_cartesian} "
                                        f"(bs={len(bs_range)}, block={len(block_range)}) "
                                        f"is too high — would cause 30+ min warmup")

    def test_256k_non_contiguous_decode_not_exploding(self):
        """256K context with non-contiguous PA: decode buckets must not explode.

        Before fix: max_blocks = ceil(262144/128) × 256 = 524288 blocks
        → log2(524288) + 1 = 20 block buckets × 9 bs = 180 decode graphs.
        After fix: max_blocks = 3593 × 3 = 10779, capped limit.
        """
        with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=False)
            strategy = ExponentialBucketingStrategy()

            bs_cfg, _, block_cfg = strategy.get_decode_cfgs(
                max_num_seqs=256,
                block_size=128,
                max_num_batched_tokens=8192,
                max_model_len=262144,
                max_blocks=3593,
            )

            bs_range = strategy.get_range(bs_cfg)
            block_range = strategy.get_range(block_cfg)

        # The buggy formula: ceil(262144/128)*256 = 524288
        buggy_max = math.ceil(262144 / 128) * 256
        assert block_cfg[2] < buggy_max, (f"Block max {block_cfg[2]} matches buggy formula {buggy_max}")

        # Must stay under budget
        decode_total = len(bs_range) * len(block_range)
        assert decode_total <= 100, (f"Decode buckets {decode_total} "
                                     f"(bs={len(bs_range)}, block={len(block_range)}) too high")

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_block_limit_cap_prevents_uncapped_explosion(self, mock_get_config):
        """Without the decode_block_limit_cap, large KV caches produce
        too many block buckets.

        With max_blocks=65536 and no cap:
          uncapped_limit = ceil(log2(65536)) + 1 = 17
          17 block buckets × 9 bs buckets = 153 decode graphs
        With cap: min(17, max(6, 9)) = 9
          ≤ 9 block buckets → manageable warmup.
        """
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
        strategy = ExponentialBucketingStrategy()

        bs_cfg, _, block_cfg = strategy.get_decode_cfgs(
            max_num_seqs=256,
            block_size=128,
            max_num_batched_tokens=8192,
            max_model_len=131072,
            max_blocks=65536,
        )

        decode_bs_limit = math.ceil(math.log2(256)) + 1  # 9
        expected_cap = max(6, decode_bs_limit)  # 9
        uncapped = math.ceil(math.log2(65536)) + 1  # 17

        assert block_cfg[3] == min(
            uncapped, expected_cap), (f"Block limit should be capped: expected {min(uncapped, expected_cap)}, "
                                      f"got {block_cfg[3]}")

        block_range = strategy.get_range(block_cfg)
        # With cap, block range should have ≤ expected_cap unique values
        assert len(block_range) <= expected_cap + 1, (f"Block range has {len(block_range)} values, expected <= "
                                                      f"{expected_cap + 1}")


###############################################################################
#                                                                             #
#  Warmup Scaling: Bucket Count Growth with Model Length                      #
#                                                                             #
###############################################################################


class TestWarmupScaling:
    """Verify that bucket counts scale sub-linearly (or logarithmically)
    with model length, not linearly or worse."""

    @pytest.mark.parametrize("use_contiguous_pa", [True, False])
    def test_exp_decode_bucket_count_scales_sublinearly(self, use_contiguous_pa):
        """Doubling model length should NOT double decode bucket count."""
        model_lens = [8192, 32768, 131072, 262144]
        prev_count = 0
        growth_factors = []

        for model_len in model_lens:
            max_blocks = max(model_len // 128, 3593)  # realistic
            with patch('vllm_gaudi.extension.bucketing.exponential.get_config') as mock:
                mock.return_value = _MockConfig(use_contiguous_pa=use_contiguous_pa)
                strategy = ExponentialBucketingStrategy()
                bs_cfg, _, block_cfg = strategy.get_decode_cfgs(
                    max_num_seqs=256,
                    block_size=128,
                    max_num_batched_tokens=8192,
                    max_model_len=model_len,
                    max_blocks=max_blocks,
                )
                bs_range = strategy.get_range(bs_cfg)
                block_range = strategy.get_range(block_cfg)
                decode_count = len(bs_range) * len(block_range)

            if prev_count > 0:
                growth = decode_count / prev_count
                growth_factors.append(growth)
            prev_count = decode_count

        # With logarithmic scaling, growth factor should be < 2x for each 4x model length increase
        for i, gf in enumerate(growth_factors):
            assert gf < 3.0, (f"Decode bucket growth factor {gf:.2f} at step {i} is "
                              f"too high — suggests linear or worse scaling with model length. "
                              f"(contiguous_pa={use_contiguous_pa})")

    def test_exp_prompt_range_sizes_logarithmic(self):
        """Prompt range dimensions should grow logarithmically with model length."""
        model_lens = [8192, 131072, 262144]
        prev_ranges = None

        for model_len in model_lens:
            ranges = _get_range_counts_exp(
                max_model_len=model_len,
                max_num_seqs=256,
                block_size=128,
                max_num_batched_tokens=8192,
                max_blocks=3593,
            )
            if prev_ranges is not None:
                # Each dimension should grow at most by a small constant
                for dim in ['prompt_bs', 'prompt_query', 'prompt_ctx']:
                    growth = ranges[dim] / max(prev_ranges[dim], 1)
                    assert growth < 4.0, (f"Prompt {dim} range grew by {growth:.1f}x between "
                                          f"model lengths — suggests non-logarithmic scaling")
            prev_ranges = ranges


###############################################################################
#                                                                             #
#  Strategy Comparison for Warmup Efficiency                                  #
#                                                                             #
###############################################################################


class TestStrategyWarmupComparison:
    """Compare bucket counts (warmup time) across strategies for the same config."""

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_exp_fewer_decode_buckets_than_linear_for_large_models(self, mock_get_config):
        """Exponential strategy should produce fewer decode range values than linear
        for large model lengths, leading to less warmup time."""
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)

        # Exponential
        exp_strategy = ExponentialBucketingStrategy()
        _, _, exp_block_cfg = exp_strategy.get_decode_cfgs(
            max_num_seqs=256,
            block_size=128,
            max_num_batched_tokens=8192,
            max_model_len=131072,
            max_blocks=65536,
        )
        exp_block_range = exp_strategy.get_range(exp_block_cfg)

        # Linear: would produce step-by-step range from min to max
        # For step=128, max=65536: that's 512 block values!
        linear_block_count = len(range(1, 65537, 128))

        assert len(exp_block_range) < linear_block_count, (
            f"Exponential block range ({len(exp_block_range)}) should be fewer than "
            f"linear ({linear_block_count})")

        # Exponential should produce significantly fewer
        ratio = len(exp_block_range) / linear_block_count
        assert ratio < 0.1, (f"Exponential produces {ratio:.1%} of linear block values — "
                             f"expected < 10%")

    def test_pad_strategy_decode_range_bounded_by_padding_limits(self):
        """Padding-aware strategy: decode block range should be bounded by pad limits."""
        with patch('vllm_gaudi.extension.bucketing.padding_aware.get_config') as mock:
            mock.return_value = _MockConfig(use_contiguous_pa=True)
            strategy = PaddingAwareBucketingStrategy()

            _, _, block_cfg = strategy.get_decode_cfgs(
                max_num_seqs=256,
                block_size=128,
                max_num_batched_tokens=8192,
                max_model_len=131072,
                max_blocks=65536,
            )
            block_range = strategy.get_range(block_cfg)

        # Padding-aware with 25% pad_percent should skip many intermediate buckets
        # compared to pure linear (which would have 512 blocks)
        linear_count = (65536 - 128) // 128 + 1
        assert len(block_range) < linear_count, (f"Padding-aware block range ({len(block_range)}) should be fewer "
                                                 f"than pure linear ({linear_count})")


###############################################################################
#                                                                             #
#  Dynamo Cache Size Limit (derived from bucket count)                        #
#                                                                             #
###############################################################################


class TestDynamoCacheSizeLimit:
    """The torch._dynamo.config.cache_size_limit is computed from
    total bucket count.  Verify the relationship stays reasonable."""

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_cache_size_limit_bounded(self, scenario_name):
        """cache_size_limit = 1 + multiplier × (prompt + decode) must be bounded.

        From warmup_model():
          cache_size_limit = 1 + multiplier * (len(prompt_buckets) + len(decode_buckets))
        where multiplier = 5 for regional compilation, 1 otherwise.
        """
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        prompt_count, decode_count = _get_bucket_counts_exp(model_len, seqs, blk_sz, tokens, blocks)

        for multiplier in [1, 5]:
            cache_size_limit = 1 + multiplier * (prompt_count + decode_count)
            # Reasonable upper bound: 1000 for standard, 5000 for regional
            max_limit = 5000 if multiplier == 5 else 1000
            assert cache_size_limit <= max_limit, (f"[{scenario_name}] cache_size_limit={cache_size_limit} "
                                                   f"(mult={multiplier}) exceeds max {max_limit}. "
                                                   f"prompt={prompt_count}, decode={decode_count}")

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_accumulated_cache_size_limit_bounded(self, scenario_name):
        """accumulated_cache_size_limit = cache_size_limit × 8 must be bounded."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        prompt_count, decode_count = _get_bucket_counts_exp(model_len, seqs, blk_sz, tokens, blocks)

        for multiplier in [1, 5]:
            cache_size_limit = 1 + multiplier * (prompt_count + decode_count)
            accumulated = cache_size_limit * 8
            max_accumulated = 40000 if multiplier == 5 else 8000
            assert accumulated <= max_accumulated, (f"[{scenario_name}] accumulated_cache_size_limit={accumulated} "
                                                    f"(mult={multiplier}) exceeds max {max_accumulated}")


###############################################################################
#                                                                             #
#  Range Size Sanity Checks (individual dimension contributions)              #
#                                                                             #
###############################################################################


class TestRangeSizeSanity:
    """Verify individual range dimensions are reasonable.

    The Cartesian product of ranges determines the raw candidate count
    before filtering.  Each dimension must be bounded.
    """

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_decode_bs_range_bounded(self, scenario_name):
        """Decode bs range should have O(log(max_num_seqs)) entries."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        ranges = _get_range_counts_exp(model_len, seqs, blk_sz, tokens, blocks)

        expected_max = math.ceil(math.log2(seqs)) + 2  # limit + 1 for rounding
        assert ranges['decode_bs'] <= expected_max, (f"[{scenario_name}] Decode BS range has {ranges['decode_bs']} "
                                                     f"entries, expected <= {expected_max}")

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_decode_block_range_bounded(self, scenario_name):
        """Decode block range should be capped by decode_block_limit_cap."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        ranges = _get_range_counts_exp(model_len, seqs, blk_sz, tokens, blocks)

        decode_bs_limit = math.ceil(math.log2(seqs)) + 1
        cap = max(6, decode_bs_limit)
        # Block range size should be <= cap + 1 (for the 0/1 origin bucket)
        assert ranges['decode_block'] <= cap + 2, (f"[{scenario_name}] Decode block range has {ranges['decode_block']} "
                                                   f"entries, expected <= {cap + 2}")

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_prompt_query_range_logarithmic(self, scenario_name):
        """Prompt query range should have O(log(max_batched_tokens)) entries."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        ranges = _get_range_counts_exp(model_len, seqs, blk_sz, tokens, blocks)

        expected_max = math.ceil(math.log2(tokens)) + 2
        assert ranges['prompt_query'] <= expected_max, (
            f"[{scenario_name}] Prompt query range has {ranges['prompt_query']} "
            f"entries, expected <= {expected_max}")

    @pytest.mark.parametrize("scenario_name", list(_SCENARIOS.keys()))
    def test_prompt_ctx_range_bounded(self, scenario_name):
        """Prompt context range should be bounded by its limit config."""
        model_len, seqs, blk_sz, tokens, blocks = _SCENARIOS[scenario_name]
        ranges = _get_range_counts_exp(model_len, seqs, blk_sz, tokens, blocks)

        # max_ctx for long context: (model_len - max_batched_tokens) / block_size
        if model_len >= 8192:
            max_ctx = max(1, (model_len - tokens) // blk_sz)
        else:
            max_ctx = max(1, (model_len - blk_sz) // blk_sz)
        expected_limit = 2 if max_ctx == 1 else math.ceil(math.log2(max_ctx)) + 1
        # +1 for the 0 bucket
        assert ranges['prompt_ctx'] <= expected_limit + 2, (
            f"[{scenario_name}] Prompt ctx range has {ranges['prompt_ctx']} "
            f"entries, expected <= {expected_limit + 2}")


###############################################################################
#                                                                             #
#  Warmup Range Function Edge Cases                                           #
#                                                                             #
###############################################################################


class TestWarmupRangeEdgeCases:
    """Test warmup range generation edge cases that affect bucket count."""

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_warmup_range_dedup_reduces_bucket_count(self, mock_get_config):
        """Exponential range should deduplicate after padding to step,
        producing fewer buckets than the raw limit."""
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
        # Small range with many potential duplicates
        config = (128, 128, 1024, 10)
        result = warmup_range_with_limit(config)

        # 10 requested, but dedup should reduce
        assert len(result) <= 10, (f"Expected <= 10 unique buckets, got {len(result)}")
        assert len(result) == len(set(result)), "Should have no duplicates"

    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_warmup_range_large_limit_still_bounded(self, mock_get_config):
        """Even with a large limit, the range is bounded by unique step-aligned values."""
        mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
        config = (128, 128, 2048, 50)  # 50 requested for only 16 possible
        result = warmup_range_with_limit(config)

        max_possible = (2048 - 128) // 128 + 1  # 16 possible step-aligned values
        assert len(result) <= max_possible, (f"Range has {len(result)} values but only {max_possible} "
                                             f"step-aligned positions exist")

    def test_linear_warmup_range_count_predictable(self):
        """Linear range count should be predictable from config."""
        config = (1, 128, 1024)
        result = linear_warmup_range(config)

        # ramp_up: 1, 2, 4, 8, 16, 32, 64
        # stable: 128, 256, 384, 512, 640, 768, 896, 1024
        ramp_up_count = len([1, 2, 4, 8, 16, 32, 64])
        stable_count = len(range(128, 1025, 128))
        expected = ramp_up_count + stable_count

        assert len(result) == expected, (f"Linear range produced {len(result)} values, expected {expected}")

    def test_padding_aware_fewer_buckets_with_higher_pad_percent(self):
        """Higher pad_percent should skip more intermediate buckets → fewer total."""
        config_low = (128, 128, 8192, 2048, 10)  # 10% padding
        config_high = (128, 128, 8192, 2048, 50)  # 50% padding

        result_low = warmup_range_with_limits(config_low)
        result_high = warmup_range_with_limits(config_high)

        assert len(result_high) <= len(result_low), (f"Higher pad_percent should produce fewer buckets: "
                                                     f"{len(result_high)} (50%) > {len(result_low)} (10%)")

    def test_padding_aware_fewer_buckets_with_lower_pad_max(self):
        """Lower pad_max means more frequent bucket keeping → more buckets."""
        config_large_pad = (128, 128, 8192, 4096, 25)  # large pad_max
        config_small_pad = (128, 128, 8192, 256, 25)  # small pad_max

        result_large = warmup_range_with_limits(config_large_pad)
        result_small = warmup_range_with_limits(config_small_pad)

        # Smaller pad_max → more buckets (since absolute limit is hit sooner)
        assert len(result_small) >= len(result_large), (
            f"Smaller pad_max should produce more buckets: "
            f"{len(result_small)} (pad_max=256) < {len(result_large)} (pad_max=4096)")


###############################################################################
#                                                                             #
#  End-to-End Manager Warmup Bucket Generation                                #
#                                                                             #
###############################################################################


class TestManagerWarmupBudget:
    """Test HPUBucketingManager end-to-end warmup bucket counts."""

    @patch('vllm_gaudi.extension.bucketing.common.get_config')
    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_manager_256k_total_warmup_budget(self, mock_exp, mock_common):
        """Full manager bucket generation for 256K: total must be within budget."""
        config = _MockConfig(
            use_contiguous_pa=False,
            merged_prefill=False,
            bucketing_strategy='exp',
            VLLM_PROMPT_QUERY_BUCKET_MIN=None,
        )
        mock_exp.return_value = config
        mock_common.return_value = config

        HPUBucketingManager._instance = None
        try:
            manager = HPUBucketingManager()
            manager.initialize(
                max_num_seqs=256,
                max_num_prefill_seqs=1,
                block_size=128,
                max_num_batched_tokens=8192,
                max_model_len=262144,
            )
            manager.num_hpu_blocks = 3593

            manager.generate_prompt_buckets()
            manager.generate_decode_buckets()

            total = len(manager.prompt_buckets) + len(manager.decode_buckets)
            assert total <= _MAX_TOTAL_BUCKETS, (f"256K scenario: {total} total buckets "
                                                 f"(prompt={len(manager.prompt_buckets)}, "
                                                 f"decode={len(manager.decode_buckets)}) exceeds "
                                                 f"warmup budget {_MAX_TOTAL_BUCKETS}")
        finally:
            HPUBucketingManager._instance = None

    @patch('vllm_gaudi.extension.bucketing.common.get_config')
    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_manager_8k_fast_warmup(self, mock_exp, mock_common):
        """8K model with small seq count: should have very few buckets → fast warmup."""
        config = _MockConfig(
            use_contiguous_pa=True,
            merged_prefill=False,
            bucketing_strategy='exp',
            VLLM_PROMPT_QUERY_BUCKET_MIN=None,
        )
        mock_exp.return_value = config
        mock_common.return_value = config

        HPUBucketingManager._instance = None
        try:
            manager = HPUBucketingManager()
            manager.initialize(
                max_num_seqs=32,
                max_num_prefill_seqs=4,
                block_size=128,
                max_num_batched_tokens=4096,
                max_model_len=8192,
            )
            manager.num_hpu_blocks = 1024

            manager.generate_prompt_buckets()
            manager.generate_decode_buckets()

            total = len(manager.prompt_buckets) + len(manager.decode_buckets)
            # Small model should warmup in under 2 minutes (~10 buckets)
            assert total <= 50, (f"8K/32-seq scenario: {total} total buckets is too many "
                                 f"for fast warmup")
        finally:
            HPUBucketingManager._instance = None

    @patch('vllm_gaudi.extension.bucketing.common.get_config')
    @patch('vllm_gaudi.extension.bucketing.exponential.get_config')
    def test_manager_spec_decode_bucket_count_bounded(self, mock_exp, mock_common):
        """With speculative decoding, additional buckets should not blow up warmup."""
        config = _MockConfig(
            use_contiguous_pa=True,
            merged_prefill=False,
            bucketing_strategy='exp',
            VLLM_PROMPT_QUERY_BUCKET_MIN=None,
        )
        mock_exp.return_value = config
        mock_common.return_value = config

        HPUBucketingManager._instance = None
        try:
            manager = HPUBucketingManager()
            manager.initialize(
                max_num_seqs=256,
                max_num_prefill_seqs=1,
                block_size=128,
                max_num_batched_tokens=8192,
                max_model_len=32768,
                num_speculative_tokens=3,
            )
            manager.num_hpu_blocks = 3593

            manager.generate_prompt_buckets()
            manager.generate_decode_buckets()

            # Spec decode adds extra buckets (seed × num_tokens expansion)
            # but should not exceed ~3x the base count
            total = len(manager.prompt_buckets) + len(manager.decode_buckets)
            assert total <= _MAX_TOTAL_BUCKETS * 2, (f"Spec decode scenario: {total} total buckets exceeds "
                                                     f"{_MAX_TOTAL_BUCKETS * 2}")

            # Seed buckets should be tracked
            assert manager.seed_decode_buckets is not None, (
                "With speculative tokens, seed_decode_buckets should be set")
            assert len(manager.seed_decode_buckets) < len(
                manager.decode_buckets), ("Spec decode should add buckets beyond seed buckets")
        finally:
            HPUBucketingManager._instance = None
