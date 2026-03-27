###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import pytest
from unittest.mock import patch

import vllm_gaudi.extension.bucketing.linear as linear
from vllm_gaudi.extension.bucketing.common import generate_buckets, calc_fallback_value
from vllm_gaudi.extension.bucketing.exponential import (ExponentialBucketingStrategy, warmup_range_with_limit)
from vllm_gaudi.extension.runtime import get_config, clear_config


@pytest.fixture(autouse=True)
def default_config():
    clear_config()
    get_config()
    yield
    clear_config()


def test_read_bucket_settings(monkeypatch):
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_MIN", "1")
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_STEP", "16")
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_MAX", "64")
    config = linear.read_bucket_settings("prompt", "bs", min=1, step=32, max=128)
    assert config == [1, 16, 64]


def test_read_bucket_settings_empty_flags():
    config = linear.read_bucket_settings("prompt", "bs", min=1, step=32, max=128)
    assert config == [1, 32, 128]


def test_warmup_range():
    config = (2, 64, 128)
    result = linear.warmup_range(config)
    assert result == [2, 4, 8, 16, 32, 64, 128]


def test_warmup_range_with_one():
    config = (1, 64, 128)
    result = linear.warmup_range(config)
    assert result == [1, 2, 4, 8, 16, 32, 64, 128]


def test_generate_prompt_buckets():
    max_num_batched_tokens = 2048
    block_size = 64
    max_model_len = 2048
    max_blocks = 1024
    bs = 16
    prompt_bs = 16
    bs_range = [1, 2, 4, 8, 16]
    query_range = [512, 1024]
    ctx_range = [0, 1, 2, 3, 4]
    buckets = generate_buckets(bs_range, query_range, ctx_range, True, max_model_len, bs, prompt_bs,
                               max_num_batched_tokens, block_size, max_blocks)
    assert len(buckets) == 25


def test_generate_decode_buckets():
    max_num_batched_tokens = 131072
    max_model_len = 2048
    max_blocks = 1024
    block_size = 128
    bs = 64
    prompt_bs = 64
    bs_range = [1, 2, 4, 8, 16, 32, 64]
    blocks_range = [128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]
    buckets = generate_buckets(bs_range, [1, 1, 1], blocks_range, False, max_model_len, bs, prompt_bs,
                               max_num_batched_tokens, block_size, max_blocks)
    assert len(buckets) == 18
    assert all(ctx <= bs * (max_model_len // block_size) for bs, _, ctx in buckets)


# --- Exponential bucketing tests ---


class _MockConfig:
    """Lightweight mock config for exponential bucketing tests."""

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


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_decode_cfgs_non_contiguous_pa_bounded(mock_get_config):
    """max_decode_blocks should be max_blocks * 3 when use_contiguous_pa=False.

    The 3x multiplier accounts for prefix-cache block sharing: the same
    physical block can appear in multiple sequences' block tables, so total
    block references may exceed num_hpu_blocks.
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    max_blocks = 3593
    block_size = 128
    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=256,
                                               block_size=block_size,
                                               max_num_batched_tokens=131072,
                                               max_model_len=91964,
                                               max_blocks=max_blocks)

    expected_max = max_blocks * 3  # 10779
    assert block_cfg[2] == expected_max, (f"Expected max_decode_blocks={expected_max}, got {block_cfg[2]}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_decode_cfgs_contiguous_pa_uses_max_blocks(mock_get_config):
    """max_decode_blocks should be max_blocks when use_contiguous_pa=True."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
    strategy = ExponentialBucketingStrategy()

    max_blocks = 3593
    block_size = 128
    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=256,
                                               block_size=block_size,
                                               max_num_batched_tokens=131072,
                                               max_model_len=91964,
                                               max_blocks=max_blocks)

    assert block_cfg[2] == max_blocks, (f"Expected max_decode_blocks={max_blocks}, got {block_cfg[2]}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_decode_max_never_exceeds_bounded_value(mock_get_config):
    """Regression test: large max_model_len must NOT produce gigantic decode buckets."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    max_model_len = 91964
    block_size = 128
    max_num_seqs = 256
    max_blocks = 3593

    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=max_num_seqs,
                                               block_size=block_size,
                                               max_num_batched_tokens=131072,
                                               max_model_len=max_model_len,
                                               max_blocks=max_blocks)

    # The old (buggy) formula would produce min(91964//128*256, ...) = 183808
    # The fix should give max_blocks * 3 = 10779
    assert block_cfg[2] <= max_blocks * 3, (f"Decode bucket max {block_cfg[2]} exceeds bounded limit "
                                            f"{max_blocks * 3}. Buckets are too large!")
    # Sanity: must not be the old gigantic value
    old_buggy_value = max_model_len // block_size * max_num_seqs
    assert block_cfg[2] < old_buggy_value, (f"Decode bucket max {block_cfg[2]} matches buggy formula output "
                                            f"{old_buggy_value}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_warmup_range_respects_max(mock_get_config):
    """warmup_range_with_limit should not produce values exceeding bmax."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    config = (1, 256, 10779, 14)
    buckets = warmup_range_with_limit(config)
    assert max(buckets) <= 10779, (f"Max bucket {max(buckets)} exceeds configured max 10779")
    assert min(buckets) >= 1


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_warmup_range_contiguous_pa(mock_get_config):
    """warmup_range_with_limit with use_contiguous_pa should set last bucket to bmax exactly."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
    config = (1, 256, 3593, 13)
    buckets = warmup_range_with_limit(config)
    assert buckets[-1] == 3593, (f"Last bucket should be bmax=3593, got {buckets[-1]}")


def test_fallback_bucket_ctx_uses_calc_fallback():
    """generate_fallback_bucket should use calc_fallback_value for ctx, capped at max prepared bucket."""
    from vllm_gaudi.extension.bucketing.common import HPUBucketingManager

    mgr = HPUBucketingManager.__new__(HPUBucketingManager)
    mgr.max_num_seqs = 256
    mgr.max_num_batched_tokens = 131072
    mgr.fallback_bs_base_step = 2
    mgr.fallback_seq_base_step = 32
    mgr.fallback_blocks_base_step = 32
    mgr.num_hpu_blocks = 3593
    mgr.block_size = 128
    mgr.use_sliding_window = False
    mgr._fallback_max_ctx = 10779  # max prepared decode bucket ctx

    # Request a ctx larger than max prepared bucket — should be capped
    _, _, new_ctx = mgr.generate_fallback_bucket(batch_size=64, seq_len=512, ctx=50000)
    assert new_ctx == 10779, (f"Fallback ctx {new_ctx} should be capped at _fallback_max_ctx=10779")

    # Request a ctx within range — should NOT be capped
    _, _, new_ctx = mgr.generate_fallback_bucket(batch_size=64, seq_len=512, ctx=7365)
    expected = calc_fallback_value(7365, 32)  # 7680
    assert new_ctx == expected, (f"Fallback ctx {new_ctx} should equal calc_fallback_value(7365, 32)={expected}")
    assert new_ctx >= 7365, (f"Fallback ctx {new_ctx} should be >= requested 7365")


# --- Scenarios derived from real server logs (Qwen3-32B, TP=2, max-model-len=91964) ---

# Parameters observed in the real run
_REAL_MAX_MODEL_LEN = 91964
_REAL_BLOCK_SIZE = 128
_REAL_MAX_NUM_SEQS = 256
_REAL_MAX_BLOCKS = 3593  # num_hpu_blocks
_REAL_MAX_BATCHED_TOKENS = 2048
_REAL_FIXED_MAX_DECODE_BLOCKS = _REAL_MAX_BLOCKS * 3  # 10779
_REAL_BUGGY_MAX_DECODE_BLOCKS = 183808  # min(91964//128*256, 3593*256//4)


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_decode_cfg_matches_fixed_log(mock_get_config):
    """Verify decode bucket config matches expected values for real scenario.

    With max_blocks * 3: block config should be [1, 256, 10779, 9]
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=_REAL_MAX_NUM_SEQS,
                                               block_size=_REAL_BLOCK_SIZE,
                                               max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
                                               max_model_len=_REAL_MAX_MODEL_LEN,
                                               max_blocks=_REAL_MAX_BLOCKS)

    # Expected: [1, 256, 10779, 9]
    assert block_cfg[0] == 1, f"block min: expected 1, got {block_cfg[0]}"
    assert block_cfg[1] == _REAL_MAX_NUM_SEQS, (f"block step: expected {_REAL_MAX_NUM_SEQS}, got {block_cfg[1]}")
    assert block_cfg[2] == _REAL_FIXED_MAX_DECODE_BLOCKS, (
        f"block max: expected {_REAL_FIXED_MAX_DECODE_BLOCKS}, got {block_cfg[2]}")
    import math
    uncapped_limit = math.ceil(math.log2(_REAL_MAX_BLOCKS * 3)) + 1
    decode_bs_limit = math.ceil(math.log2(_REAL_MAX_NUM_SEQS)) + 1
    expected_limit = min(uncapped_limit, max(6, decode_bs_limit))  # min(15, 9) = 9
    assert block_cfg[3] == expected_limit, (f"block limit: expected {expected_limit}, got {block_cfg[3]}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_decode_cfg_matches_fixed_bs_log(mock_get_config):
    """Verify decode bs config matches the fixed server log output exactly.

    From log: Decode bucket config ... bs:[1, 2, 256, 9]
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    bs_cfg, _, _ = strategy.get_decode_cfgs(max_num_seqs=_REAL_MAX_NUM_SEQS,
                                            block_size=_REAL_BLOCK_SIZE,
                                            max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
                                            max_model_len=_REAL_MAX_MODEL_LEN,
                                            max_blocks=_REAL_MAX_BLOCKS)

    # Expected from log: [1, 2, 256, 9]
    assert list(bs_cfg) == [1, 2, 256, 9], f"bs cfg mismatch: {bs_cfg}"


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_decode_block_range_bounded(mock_get_config):
    """Verify generated decode block range stays within bounds (real scenario).

    Fixed log showed blocks up to 3721. Buggy run had blocks up to 183808.
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=_REAL_MAX_NUM_SEQS,
                                               block_size=_REAL_BLOCK_SIZE,
                                               max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
                                               max_model_len=_REAL_MAX_MODEL_LEN,
                                               max_blocks=_REAL_MAX_BLOCKS)

    block_range = strategy.get_range(block_cfg)

    assert max(block_range) <= _REAL_FIXED_MAX_DECODE_BLOCKS, (
        f"Largest block bucket {max(block_range)} exceeds bounded max "
        f"{_REAL_FIXED_MAX_DECODE_BLOCKS}")
    assert max(block_range) < _REAL_BUGGY_MAX_DECODE_BLOCKS, (
        f"Block range still contains buggy value {_REAL_BUGGY_MAX_DECODE_BLOCKS}")
    # Verify reasonable number of buckets (log showed 13 unique block values)
    assert len(block_range) <= 20, (f"Too many block buckets: {len(block_range)}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_decode_bs_range_matches_log(mock_get_config):
    """Verify decode bs range matches the server log.

    Log showed bs values: 1, 2, 4, 8, 14, 24, 42, 78, 140, 256
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    bs_cfg, _, _ = strategy.get_decode_cfgs(max_num_seqs=_REAL_MAX_NUM_SEQS,
                                            block_size=_REAL_BLOCK_SIZE,
                                            max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
                                            max_model_len=_REAL_MAX_MODEL_LEN,
                                            max_blocks=_REAL_MAX_BLOCKS)

    bs_range = strategy.get_range(bs_cfg)

    expected_bs = [1, 2, 4, 8, 14, 24, 42, 78, 140, 256]
    assert bs_range == expected_bs, (f"BS range mismatch.\nExpected: {expected_bs}\nGot:      {bs_range}")


def test_real_scenario_fallback_ctx_4026_not_truncated():
    """Reproduce real fallback from log: ctx=4026 should produce a bucket >= 4026.

    calc_fallback_value(4026, 32) = 4096, which is >= the requested ctx.
    This prevents HPU graph cache misses that caused OOM.
    """
    from vllm_gaudi.extension.bucketing.common import HPUBucketingManager

    mgr = HPUBucketingManager.__new__(HPUBucketingManager)
    mgr.max_num_seqs = _REAL_MAX_NUM_SEQS
    mgr.max_num_batched_tokens = _REAL_MAX_BATCHED_TOKENS
    mgr.fallback_bs_base_step = 2
    mgr.fallback_seq_base_step = 32
    mgr.fallback_blocks_base_step = 32
    mgr.num_hpu_blocks = _REAL_MAX_BLOCKS
    mgr.block_size = _REAL_BLOCK_SIZE
    mgr.use_sliding_window = False
    mgr._fallback_max_ctx = _REAL_FIXED_MAX_DECODE_BLOCKS

    new_bs, _, new_ctx = mgr.generate_fallback_bucket(batch_size=24, seq_len=1, ctx=4026)

    assert new_ctx >= 4026, (f"Fallback ctx {new_ctx} is smaller than requested 4026 — "
                             f"this would cause HPU graph recompilation and potential OOM.")
    assert new_ctx == calc_fallback_value(4026, 32), (f"Fallback ctx {new_ctx} should equal calc_fallback_value result")
    assert new_bs >= 24, f"Batch size should be >= 24, got {new_bs}"


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_prompt_cfg_matches_log(mock_get_config):
    """Verify prompt bucket config matches the real server log.

    Log: Prompt bucket config ... bs:[1, 2, 1, 1], query:[128, 128, 2048, 11], blocks:[0, 1, 702, 11]
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False,
                                               merged_prefill=False,
                                               VLLM_PROMPT_QUERY_BUCKET_MIN=None)
    strategy = ExponentialBucketingStrategy()

    bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(max_num_prefill_seqs=1,
                                                          block_size=_REAL_BLOCK_SIZE,
                                                          max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
                                                          max_model_len=_REAL_MAX_MODEL_LEN)

    assert list(bs_cfg) == [1, 2, 1, 1], f"prompt bs cfg: {bs_cfg}"
    assert list(query_cfg) == [128, 128, 2048, 11], f"prompt query cfg: {query_cfg}"
    assert list(ctx_cfg) == [0, 1, 702, 11], f"prompt ctx cfg: {ctx_cfg}"


def test_real_scenario_fallback_ctx_3833_not_truncated():
    """Regression test for the OOM crash scenario: 22 seqs with 3833 total block refs.

    Before fix: fallback capped ctx at 3721 < 3833 actual, causing the block_list
    tensor (3833 elements) to not match the bucket (3721), triggering HPU graph
    recompilation at 97.6% KV cache utilization -> OOM.

    After fix: fallback returns ctx >= 3833 (no cap), so the graph matches and
    no recompilation occurs.
    """
    from vllm_gaudi.extension.bucketing.common import HPUBucketingManager

    mgr = HPUBucketingManager.__new__(HPUBucketingManager)
    mgr.max_num_seqs = _REAL_MAX_NUM_SEQS
    mgr.max_num_batched_tokens = _REAL_MAX_BATCHED_TOKENS
    mgr.fallback_bs_base_step = 2
    mgr.fallback_seq_base_step = 32
    mgr.fallback_blocks_base_step = 32
    mgr.num_hpu_blocks = _REAL_MAX_BLOCKS
    mgr.block_size = _REAL_BLOCK_SIZE
    mgr.use_sliding_window = False
    mgr._fallback_max_ctx = _REAL_FIXED_MAX_DECODE_BLOCKS

    new_bs, _, new_ctx = mgr.generate_fallback_bucket(batch_size=22, seq_len=1, ctx=3833)

    assert new_ctx >= 3833, (f"Fallback ctx {new_ctx} < 3833: block_list won't be padded, "
                             f"HPU graph cache miss will trigger recompilation and likely OOM.")
    assert new_ctx == calc_fallback_value(3833, 32), (f"Fallback ctx {new_ctx} should equal calc_fallback_value result")


def test_real_scenario_fallback_ctx_7365_not_truncated():
    """Regression test: 43 seqs with 7365 total block refs (from second benchmark run).

    With the old cap (num_hpu_blocks * 2 + block_size = 7314), the fallback
    bucket would be capped at 7314 < 7365, causing tensor/graph size mismatch.
    Without cap, calc_fallback_value(7365, 32) = 7680, which properly covers
    the actual block_list.
    """
    from vllm_gaudi.extension.bucketing.common import HPUBucketingManager

    mgr = HPUBucketingManager.__new__(HPUBucketingManager)
    mgr.max_num_seqs = _REAL_MAX_NUM_SEQS
    mgr.max_num_batched_tokens = _REAL_MAX_BATCHED_TOKENS
    mgr.fallback_bs_base_step = 2
    mgr.fallback_seq_base_step = 32
    mgr.fallback_blocks_base_step = 32
    mgr.num_hpu_blocks = _REAL_MAX_BLOCKS
    mgr.block_size = _REAL_BLOCK_SIZE
    mgr.use_sliding_window = False
    mgr._fallback_max_ctx = _REAL_FIXED_MAX_DECODE_BLOCKS

    new_bs, _, new_ctx = mgr.generate_fallback_bucket(batch_size=43, seq_len=1, ctx=7365)

    assert new_ctx >= 7365, (f"Fallback ctx {new_ctx} < 7365: tensor/graph size mismatch.")
    assert new_ctx == calc_fallback_value(7365, 32), (f"Fallback ctx {new_ctx} should equal calc_fallback_value result")


def test_real_scenario_fallback_ctx_7408_not_truncated():
    """Regression test: 44 seqs with 7408 total block refs (from first benchmark run).

    Without cap, calc_fallback_value(7408, 32) produces a value >= 7408,
    ensuring pad_list pads up correctly and no graph recompilation occurs.
    """
    from vllm_gaudi.extension.bucketing.common import HPUBucketingManager

    mgr = HPUBucketingManager.__new__(HPUBucketingManager)
    mgr.max_num_seqs = _REAL_MAX_NUM_SEQS
    mgr.max_num_batched_tokens = _REAL_MAX_BATCHED_TOKENS
    mgr.fallback_bs_base_step = 2
    mgr.fallback_seq_base_step = 32
    mgr.fallback_blocks_base_step = 32
    mgr.num_hpu_blocks = _REAL_MAX_BLOCKS
    mgr.block_size = _REAL_BLOCK_SIZE
    mgr.use_sliding_window = False
    mgr._fallback_max_ctx = _REAL_FIXED_MAX_DECODE_BLOCKS

    new_bs, _, new_ctx = mgr.generate_fallback_bucket(batch_size=44, seq_len=1, ctx=7408)

    assert new_ctx >= 7408, (f"Fallback ctx {new_ctx} < 7408: tensor/graph size mismatch.")
    assert new_ctx == calc_fallback_value(7408, 32), (f"Fallback ctx {new_ctx} should equal calc_fallback_value result")


def test_exponential_decode_block_limit_cap(monkeypatch):
    """Verify that the decode block limit is capped to avoid excessive warmup.

    Reproduces the GAUDISW-247226 scenario: max_num_seqs=21 with a large KV
    cache (65536 blocks) previously produced ~126 decode buckets and ~30 min
    warmup.  With the cap the block dimension should have at most 6 exponential
    steps, giving significantly fewer total buckets.
    """
    monkeypatch.setenv("VLLM_EXPONENTIAL_BUCKETING", "true")
    monkeypatch.setenv("VLLM_CONTIGUOUS_PA", "true")
    clear_config()
    get_config()

    strategy = ExponentialBucketingStrategy()
    max_num_seqs = 21
    block_size = 128
    max_num_batched_tokens = 8192
    max_model_len = 131072
    max_blocks = 65536

    bs_cfg, query_cfg, block_cfg = strategy.get_decode_cfgs(max_num_seqs, block_size, max_num_batched_tokens,
                                                            max_model_len, max_blocks)

    bs_range = strategy.get_range(bs_cfg)
    block_range = strategy.get_range(block_cfg)

    # decode_bs_limit = ceil(log2(21)) + 1 = 6
    # cap = max(6, 6) = 6  →  block_limit capped at 6
    assert block_cfg[3] == 6

    # Block range: 6 exponential values + 1 (bmin_origin=1) ≤ 7 unique values
    assert len(block_range) <= 7

    # Total decode buckets (Cartesian product) should be much less than
    # the uncapped ~126.
    total = len(bs_range) * len(block_range)
    assert total <= 50
