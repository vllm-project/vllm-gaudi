###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import pytest

import vllm_gaudi.extension.bucketing.linear as linear
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
    config = linear.read_bucket_settings("prompt",
                                         "bs",
                                         min=1,
                                         step=32,
                                         max=128)
    assert config == [1, 16, 64]


def test_read_bucket_settings_empty_flags():
    config = linear.read_bucket_settings("prompt",
                                         "bs",
                                         min=1,
                                         step=32,
                                         max=128)
    assert config == [1, 32, 128]


def test_warmup_range():
    config = (2, 64, 128)
    result = linear.warmup_range(config)
    assert result == [2, 4, 8, 16, 32, 64, 128]


def test_generate_prompt_buckets():
    bs_bucket_config = (1, 4, 16)
    seq_bucket_config = (512, 512, 1024)
    max_num_batched_tokens = 2048
    block_size = 64
    prefix_caching = False
    buckets, omitted_buckets = linear.generate_prompt_buckets(
        bs_bucket_config, seq_bucket_config, block_size, prefix_caching,
        max_num_batched_tokens)
    assert len(buckets) == 5
    assert len(omitted_buckets) == 7
    assert all(bs * seq <= max_num_batched_tokens for bs, seq, _ in buckets)


def test_generate_decode_buckets():
    bs_bucket_config = [1, 32, 128]
    blocks_bucket_config = [128, 128, 2048]
    max_blocks = 1024
    max_model_len = 131072
    block_size=128
    buckets = linear.generate_decode_buckets(bs_bucket_config,
                                             blocks_bucket_config, max_blocks,
                                             max_model_len, block_size)
    assert len(buckets) == 72
    assert all(blocks <= max_blocks for _, _, blocks in buckets)
