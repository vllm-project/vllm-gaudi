###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import math

import pytest
from unittest.mock import patch

import vllm_gaudi.extension.bucketing.linear as linear
import vllm_gaudi.extension.bucketing.padding_aware as padding_aware
from vllm_gaudi.extension.bucketing.common import HPUBucketingManager, generate_buckets, calc_fallback_value
from vllm_gaudi.extension.bucketing.exponential import (ExponentialBucketingStrategy, warmup_range_with_limit)
from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
from vllm_gaudi.extension.bucketing.padding_aware import PaddingAwareBucketingStrategy
from vllm_gaudi.extension.runtime import get_config, clear_config


@pytest.fixture(autouse=True)
def default_config():
    clear_config()
    get_config()
    yield
    clear_config()


@pytest.mark.parametrize(
    ("env_value", "expected_type"),
    [
        ("exp", ExponentialBucketingStrategy),
        ("lin", LinearBucketingStrategy),
        ("pad", PaddingAwareBucketingStrategy),
    ],
)
def test_get_bucketing_strategy_selected_by_env(monkeypatch, env_value, expected_type):
    monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", env_value)
    clear_config()

    manager = HPUBucketingManager.__new__(HPUBucketingManager)
    strategy = manager.get_bucketing_strategy()

    assert isinstance(strategy, expected_type)


def test_get_bucketing_strategy_default_when_env_not_set(monkeypatch):
    monkeypatch.delenv("VLLM_BUCKETING_STRATEGY", raising=False)
    clear_config()

    manager = HPUBucketingManager.__new__(HPUBucketingManager)
    strategy = manager.get_bucketing_strategy()

    assert isinstance(strategy, ExponentialBucketingStrategy)


@patch('vllm_gaudi.extension.bucketing.common.logger')
def test_get_bucketing_strategy_deprecated_env_overrides_to_exponential(mock_logger, monkeypatch):
    monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", "lin")
    monkeypatch.setenv("VLLM_EXPONENTIAL_BUCKETING", "true")
    clear_config()

    manager = HPUBucketingManager.__new__(HPUBucketingManager)
    strategy = manager.get_bucketing_strategy()

    assert isinstance(strategy, ExponentialBucketingStrategy)

    warnings = [call.args[0] for call in mock_logger.return_value.warning.call_args_list]
    assert any("deprecated" in message for message in warnings)
    assert any("Overriding bucketing strategy LinearBucketingStrategy with ExponentialBucketingStrategy" in message
               for message in warnings)


@patch('vllm_gaudi.extension.bucketing.common.logger')
def test_get_bucketing_strategy_deprecated_env_overrides_to_linear(mock_logger, monkeypatch):
    monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", "pad")
    monkeypatch.setenv("VLLM_EXPONENTIAL_BUCKETING", "false")
    clear_config()

    manager = HPUBucketingManager.__new__(HPUBucketingManager)
    strategy = manager.get_bucketing_strategy()

    assert isinstance(strategy, LinearBucketingStrategy)

    warnings = [call.args[0] for call in mock_logger.return_value.warning.call_args_list]
    assert any("deprecated" in message for message in warnings)
    assert any("Overriding bucketing strategy PaddingAwareBucketingStrategy with LinearBucketingStrategy" in message
               for message in warnings)


@patch('vllm_gaudi.extension.bucketing.common.logger')
def test_get_bucketing_strategy_deprecated_env_without_override_logs_only_deprecation(mock_logger, monkeypatch):
    monkeypatch.setenv("VLLM_BUCKETING_STRATEGY", "exp")
    monkeypatch.setenv("VLLM_EXPONENTIAL_BUCKETING", "true")
    clear_config()

    manager = HPUBucketingManager.__new__(HPUBucketingManager)
    strategy = manager.get_bucketing_strategy()

    assert isinstance(strategy, ExponentialBucketingStrategy)

    warnings = [call.args[0] for call in mock_logger.return_value.warning.call_args_list]
    assert any("deprecated" in message for message in warnings)
    assert not any("Overriding bucketing strategy" in message for message in warnings)


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
def test_exponential_decode_cfgs_non_contiguous_pa_unbounded(mock_get_config):
    """max_decode_blocks should be ceil(max_model_len/block_size)*max_num_seqs
    when use_contiguous_pa=False.  Actual bounding of generated buckets
    happens via filters in generate_buckets().
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    max_blocks = 3593
    block_size = 128
    max_model_len = 91964
    max_num_seqs = 256
    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=max_num_seqs,
                                               block_size=block_size,
                                               max_num_batched_tokens=131072,
                                               max_model_len=max_model_len,
                                               max_blocks=max_blocks)

    expected_max = math.ceil(max_model_len / block_size) * max_num_seqs
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
def test_exponential_decode_cfgs_non_contiguous_pa_formula(mock_get_config):
    """Verify non-contiguous PA decode cfg uses ceil(max_model_len/block_size)*max_num_seqs.

    Actual bounding of excessive buckets happens via the
    num_ctx_tokens_less_or_equal_batched_max_model_len filter in generate_buckets().
    """
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

    expected_max = math.ceil(max_model_len / block_size) * max_num_seqs
    assert block_cfg[2] == expected_max, (f"Expected max_decode_blocks={expected_max}, got {block_cfg[2]}")


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
    With non-contiguous PA: block config should be
    [1, 256, ceil(91964/128)*256, ceil(log2(that))+1]
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=_REAL_MAX_NUM_SEQS,
                                               block_size=_REAL_BLOCK_SIZE,
                                               max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
                                               max_model_len=_REAL_MAX_MODEL_LEN,
                                               max_blocks=_REAL_MAX_BLOCKS)

    expected_max = math.ceil(_REAL_MAX_MODEL_LEN / _REAL_BLOCK_SIZE) * _REAL_MAX_NUM_SEQS
    expected_limit = math.ceil(math.log2(expected_max)) + 1
    assert block_cfg[0] == 1, f"block min: expected 1, got {block_cfg[0]}"
    assert block_cfg[1] == _REAL_MAX_NUM_SEQS, (f"block step: expected {_REAL_MAX_NUM_SEQS}, got {block_cfg[1]}")
    assert block_cfg[2] == expected_max, (f"block max: expected {expected_max}, got {block_cfg[2]}")
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
def test_real_scenario_decode_block_range_within_cfg_max(mock_get_config):
    """Verify generated decode block range stays within cfg max (real scenario).

    The block range from get_range() extends up to max_decode_blocks.
    Actual bounding per (bs, ctx) pair happens via filters in generate_buckets().
    """
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=False)
    strategy = ExponentialBucketingStrategy()

    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=_REAL_MAX_NUM_SEQS,
                                               block_size=_REAL_BLOCK_SIZE,
                                               max_num_batched_tokens=_REAL_MAX_BATCHED_TOKENS,
                                               max_model_len=_REAL_MAX_MODEL_LEN,
                                               max_blocks=_REAL_MAX_BLOCKS)

    block_range = strategy.get_range(block_cfg)
    expected_max = math.ceil(_REAL_MAX_MODEL_LEN / _REAL_BLOCK_SIZE) * _REAL_MAX_NUM_SEQS

    assert max(block_range) <= expected_max, (f"Largest block bucket {max(block_range)} exceeds cfg max {expected_max}")


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


def test_exponential_decode_block_limit_uncapped(monkeypatch):
    """Verify that decode block limit is computed from log2(max_decode_blocks).

    With the new approach, excessive warmup buckets are controlled by
    filters in generate_buckets() (num_ctx_tokens_less_or_equal_batched_max_model_len)
    rather than by capping the block limit in get_decode_cfgs().
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

    # max_decode_blocks = min(65536, ceil(131072/128)*21) = min(65536, 21504) = 21504
    expected_max_decode_blocks = min(max_blocks, math.ceil(max_model_len / block_size) * max_num_seqs)
    expected_limit = math.ceil(math.log2(expected_max_decode_blocks)) + 1
    assert block_cfg[2] == expected_max_decode_blocks, (
        f"Expected max_decode_blocks={expected_max_decode_blocks}, got {block_cfg[2]}")
    assert block_cfg[3] == expected_limit, (f"Expected decode_blocks_limit={expected_limit}, got {block_cfg[3]}")
# --- Padding-aware bucketing tests ---


def test_padding_aware_read_bucket_settings_query_seq_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_PROMPT_SEQ_BUCKET_MIN", "64")
    monkeypatch.setenv("VLLM_PROMPT_SEQ_BUCKET_STEP", "128")
    monkeypatch.setenv("VLLM_PROMPT_SEQ_BUCKET_MAX", "1024")
    monkeypatch.setenv("VLLM_PROMPT_SEQ_BUCKET_PAD_MAX", "256")
    monkeypatch.setenv("VLLM_PROMPT_SEQ_BUCKET_PAD_PERCENT", "10")

    config = padding_aware.read_bucket_settings("prompt",
                                                "query",
                                                min=32,
                                                step=32,
                                                max=2048,
                                                pad_max=512,
                                                pad_percent=25)

    assert config == [64, 128, 1024, 256, 10]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ((0, 8, 64, 64, 0), [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]),
        ((0, 8, 64, 64, 50), [0, 1, 2, 4, 8, 16, 32, 64]),
        ((0, 8, 64, 16, 50), [0, 1, 2, 4, 8, 16, 32, 48, 64]),
        ((16, 16, 128, 32, 25), [16, 32, 48, 64, 80, 96, 128]),
    ],
)
def test_padding_aware_warmup_range_with_limits_examples(config, expected):
    assert padding_aware.warmup_range_with_limits(config) == expected


def test_padding_aware_prompt_cfgs_defaults():
    strategy = PaddingAwareBucketingStrategy()

    bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(max_num_prefill_seqs=16,
                                                          block_size=128,
                                                          max_num_batched_tokens=2048,
                                                          max_model_len=4096)

    assert bs_cfg == [1, 1, 16, 4, 25]
    assert query_cfg == [128, 128, 2048, 512, 25]
    assert ctx_cfg == [0, 2, 31, 16, 25]


@patch('vllm_gaudi.extension.bucketing.padding_aware.logger')
@patch('vllm_gaudi.extension.bucketing.padding_aware.get_config')
def test_padding_aware_prompt_cfgs_merged_prefill_overrides_defaults(mock_get_config, mock_logger):
    mock_get_config.return_value = _MockConfig(merged_prefill=True)
    strategy = PaddingAwareBucketingStrategy()

    bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(max_num_prefill_seqs=16,
                                                          block_size=128,
                                                          max_num_batched_tokens=2048,
                                                          max_model_len=4096)

    assert bs_cfg == (1, 1, 1, 4, 25)
    assert query_cfg == (128, 512, 2048, 512, 25)
    assert ctx_cfg == [0, 4, 496, 16, 25]

    info_messages = [call.args[0] for call in mock_logger.return_value.info.call_args_list]
    assert any('Merged prefill is enabled!' in message for message in info_messages)
    assert any('prompt bs cfg: (1, 1, 16, 4, 25) -> (1, 1, 1, 4, 25)' in message for message in info_messages)
    assert any('prompt query cfg: (128, 128, 2048, 512, 25) -> (128, 512, 2048, 512, 25)' in message
               for message in info_messages)
    assert any('prompt ctx cfg: (0, 2, 31, 16, 25) -> [0, 4, 496, 16, 25]' in message for message in info_messages)


@patch('vllm_gaudi.extension.bucketing.padding_aware.get_config')
def test_padding_aware_prompt_cfgs_merged_prefill_preserves_user_padding_limits(mock_get_config, monkeypatch):
    mock_get_config.return_value = _MockConfig(merged_prefill=True)
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_PAD_MAX", "8")
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_PAD_PERCENT", "10")
    monkeypatch.setenv("VLLM_PROMPT_QUERY_BUCKET_STEP", "256")
    monkeypatch.setenv("VLLM_PROMPT_QUERY_BUCKET_PAD_MAX", "640")
    monkeypatch.setenv("VLLM_PROMPT_QUERY_BUCKET_PAD_PERCENT", "15")
    monkeypatch.setenv("VLLM_PROMPT_CTX_BUCKET_PAD_MAX", "32")
    monkeypatch.setenv("VLLM_PROMPT_CTX_BUCKET_PAD_PERCENT", "5")

    strategy = PaddingAwareBucketingStrategy()
    bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(max_num_prefill_seqs=16,
                                                          block_size=128,
                                                          max_num_batched_tokens=2048,
                                                          max_model_len=4096)

    assert bs_cfg == (1, 1, 1, 8, 10)
    assert query_cfg == (128, 1024, 2048, 640, 15)
    assert ctx_cfg == [0, 4, 496, 32, 5]


@patch('vllm_gaudi.extension.bucketing.padding_aware.get_config')
def test_padding_aware_decode_cfgs_contiguous_pa_clamps_block_range(mock_get_config, monkeypatch):
    mock_get_config.return_value = _MockConfig(use_contiguous_pa=True)
    monkeypatch.setenv("VLLM_DECODE_BLOCK_BUCKET_MIN", "4096")
    monkeypatch.setenv("VLLM_DECODE_BLOCK_BUCKET_STEP", "128")
    monkeypatch.setenv("VLLM_DECODE_BLOCK_BUCKET_MAX", "8192")

    strategy = PaddingAwareBucketingStrategy()
    _, _, block_cfg = strategy.get_decode_cfgs(max_num_seqs=64,
                                               block_size=128,
                                               max_num_batched_tokens=2048,
                                               max_model_len=4096,
                                               max_blocks=3593)

    assert block_cfg == [3465, 128, 3593, 899, 25]
