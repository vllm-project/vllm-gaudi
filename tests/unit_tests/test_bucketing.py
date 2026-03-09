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
from vllm_gaudi.extension.bucketing.exponential import (
    ExponentialBucketingStrategy, warmup_range_with_limit)
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
    """max_decode_blocks should be max_blocks + block_size when use_contiguous_pa=False."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    max_blocks = 3593
    block_size = 128
    _, _, block_cfg = strategy.get_decode_cfgs(
        max_num_seqs=256, block_size=block_size,
        max_num_batched_tokens=131072, max_model_len=91964,
        max_blocks=max_blocks)

    expected_max = max_blocks + block_size  # 3721
    assert block_cfg[2] == expected_max, (
        f"Expected max_decode_blocks={expected_max}, got {block_cfg[2]}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_decode_cfgs_contiguous_pa_uses_max_blocks(mock_get_config):
    """max_decode_blocks should be max_blocks when use_contiguous_pa=True."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
    strategy = ExponentialBucketingStrategy()

    max_blocks = 3593
    block_size = 128
    _, _, block_cfg = strategy.get_decode_cfgs(
        max_num_seqs=256, block_size=block_size,
        max_num_batched_tokens=131072, max_model_len=91964,
        max_blocks=max_blocks)

    assert block_cfg[2] == max_blocks, (
        f"Expected max_decode_blocks={max_blocks}, got {block_cfg[2]}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_decode_max_never_exceeds_bounded_value(mock_get_config):
    """Regression test: large max_model_len must NOT produce gigantic decode buckets."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    max_model_len = 91964
    block_size = 128
    max_num_seqs = 256
    max_blocks = 3593

    _, _, block_cfg = strategy.get_decode_cfgs(
        max_num_seqs=max_num_seqs, block_size=block_size,
        max_num_batched_tokens=131072, max_model_len=max_model_len,
        max_blocks=max_blocks)

    # The old (buggy) formula would produce min(91964//128*256, ...) = 183808
    # The fix should give max_blocks + block_size = 3721
    assert block_cfg[2] <= max_blocks + block_size, (
        f"Decode bucket max {block_cfg[2]} exceeds bounded limit "
        f"{max_blocks + block_size}. Buckets are too large!")
    # Sanity: must not be the old gigantic value
    old_buggy_value = max_model_len // block_size * max_num_seqs
    assert block_cfg[2] < old_buggy_value, (
        f"Decode bucket max {block_cfg[2]} matches buggy formula output "
        f"{old_buggy_value}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_warmup_range_respects_max(mock_get_config):
    """warmup_range_with_limit should not produce values exceeding bmax."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    config = (1, 256, 3721, 13)
    buckets = warmup_range_with_limit(config)
    assert max(buckets) <= 3721, (
        f"Max bucket {max(buckets)} exceeds configured max 3721")
    assert min(buckets) >= 1


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_exponential_warmup_range_contiguous_pa(mock_get_config):
    """warmup_range_with_limit with use_contiguous_pa should set last bucket to bmax exactly."""
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
    config = (1, 256, 3593, 13)
    buckets = warmup_range_with_limit(config)
    assert buckets[-1] == 3593, (
        f"Last bucket should be bmax=3593, got {buckets[-1]}")


def test_fallback_bucket_ctx_capped():
    """generate_fallback_bucket should cap ctx at num_hpu_blocks + block_size."""
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

    # Request a ctx far larger than num_hpu_blocks
    _, _, new_ctx = mgr.generate_fallback_bucket(batch_size=64, seq_len=512, ctx=50000)
    assert new_ctx <= mgr.num_hpu_blocks + mgr.block_size, (
        f"Fallback ctx {new_ctx} exceeds cap {mgr.num_hpu_blocks + mgr.block_size}")


# --- Scenarios derived from real server logs (Qwen3-32B, TP=2, max-model-len=91964) ---

# Parameters observed in the real run
_REAL_MAX_MODEL_LEN = 91964
_REAL_BLOCK_SIZE = 128
_REAL_MAX_NUM_SEQS = 256
_REAL_MAX_BLOCKS = 3593  # num_hpu_blocks
_REAL_MAX_BATCHED_TOKENS = 2048
_REAL_FIXED_MAX_DECODE_BLOCKS = _REAL_MAX_BLOCKS + _REAL_BLOCK_SIZE  # 3721
_REAL_BUGGY_MAX_DECODE_BLOCKS = 183808  # min(91964//128*256, 3593*256//4)


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_decode_cfg_matches_fixed_log(mock_get_config):
    """Verify decode bucket config matches the fixed server log output exactly.

    From log: Decode bucket config ... block:[1, 256, 3721, 13]
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    _, _, block_cfg = strategy.get_decode_cfgs(
        max_num_seqs=_REAL_MAX_NUM_SEQS,
        block_size=_REAL_BLOCK_SIZE,
        max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
        max_model_len=_REAL_MAX_MODEL_LEN,
        max_blocks=_REAL_MAX_BLOCKS)

    # Expected from fixed log: [1, 256, 3721, 13]
    assert block_cfg[0] == 1, f"block min: expected 1, got {block_cfg[0]}"
    assert block_cfg[1] == _REAL_MAX_NUM_SEQS, (
        f"block step: expected {_REAL_MAX_NUM_SEQS}, got {block_cfg[1]}")
    assert block_cfg[2] == _REAL_FIXED_MAX_DECODE_BLOCKS, (
        f"block max: expected {_REAL_FIXED_MAX_DECODE_BLOCKS}, got {block_cfg[2]}")
    import math
    expected_limit = math.ceil(math.log2(_REAL_MAX_BLOCKS)) + 1  # 13
    assert block_cfg[3] == expected_limit, (
        f"block limit: expected {expected_limit}, got {block_cfg[3]}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_decode_cfg_matches_fixed_bs_log(mock_get_config):
    """Verify decode bs config matches the fixed server log output exactly.

    From log: Decode bucket config ... bs:[1, 2, 256, 9]
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    bs_cfg, _, _ = strategy.get_decode_cfgs(
        max_num_seqs=_REAL_MAX_NUM_SEQS,
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

    _, _, block_cfg = strategy.get_decode_cfgs(
        max_num_seqs=_REAL_MAX_NUM_SEQS,
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
    assert len(block_range) <= 20, (
        f"Too many block buckets: {len(block_range)}")


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_decode_bs_range_matches_log(mock_get_config):
    """Verify decode bs range matches the server log.

    Log showed bs values: 1, 2, 4, 8, 14, 24, 42, 78, 140, 256
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    bs_cfg, _, _ = strategy.get_decode_cfgs(
        max_num_seqs=_REAL_MAX_NUM_SEQS,
        block_size=_REAL_BLOCK_SIZE,
        max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
        max_model_len=_REAL_MAX_MODEL_LEN,
        max_blocks=_REAL_MAX_BLOCKS)

    bs_range = strategy.get_range(bs_cfg)

    expected_bs = [1, 2, 4, 8, 14, 24, 42, 78, 140, 256]
    assert bs_range == expected_bs, (
        f"BS range mismatch.\nExpected: {expected_bs}\nGot:      {bs_range}")


def test_real_scenario_fallback_ctx_4026_capped_to_3721():
    """Reproduce real fallback from log: ctx=4026 should cap to <= 3721.

    Log: "Decode bucket for (24, 1, 4026) was not prepared. Adding new bucket: (24, 1, 3721)"
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

    new_bs, _, new_ctx = mgr.generate_fallback_bucket(
        batch_size=24, seq_len=1, ctx=4026)

    assert new_ctx <= _REAL_FIXED_MAX_DECODE_BLOCKS, (
        f"Fallback ctx {new_ctx} exceeds cap {_REAL_FIXED_MAX_DECODE_BLOCKS}. "
        f"Log showed it should fall back to 3721.")
    assert new_bs >= 24, f"Batch size should be >= 24, got {new_bs}"


@patch('vllm_gaudi.extension.bucketing.exponential.get_config')
def test_real_scenario_prompt_cfg_matches_log(mock_get_config):
    """Verify prompt bucket config matches the real server log.

    Log: Prompt bucket config ... bs:[1, 2, 1, 1], query:[128, 128, 2048, 11], blocks:[0, 1, 702, 11]
    """
    mock_get_config.return_value = _MockConfig(
        use_contiguous_pa=False, merged_prefill=False,
        VLLM_PROMPT_QUERY_BUCKET_MIN=None)
    strategy = ExponentialBucketingStrategy()

    bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
        max_num_prefill_seqs=1,
        block_size=_REAL_BLOCK_SIZE,
        max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
        max_model_len=_REAL_MAX_MODEL_LEN)

    assert list(bs_cfg) == [1, 2, 1, 1], f"prompt bs cfg: {bs_cfg}"
    assert list(query_cfg) == [128, 128, 2048, 11], f"prompt query cfg: {query_cfg}"
    assert list(ctx_cfg) == [0, 1, 702, 11], f"prompt ctx cfg: {ctx_cfg}"
