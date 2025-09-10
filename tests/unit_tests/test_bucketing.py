###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import pytest

import vllm_gaudi.extension.bucketing.linear as linear
from vllm_gaudi.extension.bucketing.common import generate_buckets
from vllm_gaudi.extension.runtime import get_config, clear_config


@pytest.fixture(autouse=True)
def default_config():
    get_config(prefix_caching=True)
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
    assert len(buckets) == 40


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
    assert len(buckets) == 15
    assert all(ctx <= bs * (max_model_len // block_size) for bs, _, ctx in buckets)
