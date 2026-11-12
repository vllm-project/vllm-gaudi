# SPDX-License-Identifier: Apache-2.0
"""VLLM_HPU_MIN_PREFILL_CHUNK rule of HPUAsyncScheduler."""

import pytest
import torch

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler

ENV = "VLLM_HPU_MIN_PREFILL_CHUNK"
BLOCK_SIZE = 16
MNBT = 1024
PROMPT = 923
LONG = 2000
MIN_CHUNK = 400


def _scheduler(cls, max_num_batched_tokens=MNBT, max_model_len=2048, long_prefill_token_threshold=0):
    model_config = ModelConfig(model="facebook/opt-125m",
                               dtype="float16",
                               seed=42,
                               skip_tokenizer_init=True,
                               max_model_len=max_model_len)
    scheduler_config = SchedulerConfig(max_num_seqs=64,
                                       max_num_batched_tokens=max_num_batched_tokens,
                                       max_model_len=max_model_len,
                                       long_prefill_token_threshold=long_prefill_token_threshold,
                                       enable_chunked_prefill=True,
                                       async_scheduling=True,
                                       is_encoder_decoder=False,
                                       watermark=0.0)
    cache_config = CacheConfig(block_size=BLOCK_SIZE, cache_dtype="auto", enable_prefix_caching=False)
    vllm_config = VllmConfig(scheduler_config=scheduler_config,
                             model_config=model_config,
                             cache_config=cache_config,
                             device_config=DeviceConfig("cpu"),
                             parallel_config=ParallelConfig())
    num_blocks = 1024
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer"],
                             FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32))
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    register_all_kvcache_specs(vllm_config)
    return cls(vllm_config=vllm_config,
               kv_cache_config=kv_cache_config,
               block_size=BLOCK_SIZE,
               log_stats=True,
               structured_output_manager=StructuredOutputManager(vllm_config))


def _requests(*num_tokens, prefix="r"):
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    reqs = []
    for i, n in enumerate(num_tokens):
        params = SamplingParams(ignore_eos=True, max_tokens=16)
        params.update_from_generation_config({}, 50256)
        reqs.append(
            Request(request_id=f"{prefix}{i}",
                    prompt_token_ids=[i + 1] * n,
                    sampling_params=params,
                    pooling_params=None,
                    block_hasher=hasher))
    return reqs


def _run(scheduler, num_tokens, steps):
    for req in _requests(*num_tokens):
        scheduler.add_request(req)
    return [dict(scheduler.schedule().num_scheduled_tokens) for _ in range(steps)]


@pytest.fixture
def min_chunk(monkeypatch):

    def _set(value):
        if value is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, str(value))

    return _set


@pytest.mark.parametrize("num_tokens", [(PROMPT, PROMPT), (LONG, PROMPT, 50), (30, 40, PROMPT)])
def test_rule_off_matches_base_scheduler(min_chunk, num_tokens):
    min_chunk(None)
    hpu = _run(_scheduler(HPUAsyncScheduler), num_tokens, steps=4)
    base = _run(_scheduler(AsyncScheduler), num_tokens, steps=4)
    assert hpu == base


def test_small_head_chunk_deferred_next_to_prefill(min_chunk):
    min_chunk(MIN_CHUNK)
    steps = _run(_scheduler(HPUAsyncScheduler), (PROMPT, PROMPT), steps=2)
    assert steps[0] == {"r0": PROMPT}
    assert steps[1] == {"r0": 1, "r1": PROMPT}


def test_short_prompt_fitting_whole_not_deferred(min_chunk):
    min_chunk(MIN_CHUNK)
    steps = _run(_scheduler(HPUAsyncScheduler), (PROMPT, 50), steps=1)
    assert steps[0] == {"r0": PROMPT, "r1": 50}


def test_deferred_next_to_running_continuation(min_chunk):
    min_chunk(MIN_CHUNK)
    steps = _run(_scheduler(HPUAsyncScheduler), (LONG, PROMPT), steps=3)
    assert steps[0] == {"r0": MNBT}
    # The running continuation chunk is never deferred.
    assert steps[1] == {"r0": LONG - MNBT}
    assert steps[2] == {"r0": 1, "r1": PROMPT}


def test_lone_prefill_started_with_small_chunk(min_chunk):
    min_chunk(MNBT + MIN_CHUNK)
    steps = _run(_scheduler(HPUAsyncScheduler), (LONG, ), steps=2)
    assert steps == [{"r0": MNBT}, {"r0": LONG - MNBT}]


def test_lone_prefill_after_decodes_not_deferred(min_chunk):
    min_chunk(MNBT + MIN_CHUNK)
    sched = _scheduler(HPUAsyncScheduler)
    assert _run(sched, (30, 40), steps=2) == [{"r0": 30, "r1": 40}, {"r0": 1, "r1": 1}]
    for req in _requests(LONG, prefix="late"):
        sched.add_request(req)
    # Decodes and prefills of earlier steps do not count as a prefill in this step.
    assert dict(sched.schedule().num_scheduled_tokens) == {"r0": 1, "r1": 1, "late0": MNBT - 2}


def test_running_continuation_not_deferred_after_prefill(min_chunk):
    min_chunk(None)
    sched = _scheduler(HPUAsyncScheduler, long_prefill_token_threshold=600)
    assert _run(sched, (LONG, LONG), steps=1) == [{"r0": 600, "r1": MNBT - 600}]
    sched._min_prefill_chunk = MNBT
    # r1 is RUNNING: its continuation chunk follows r0's prefill and is kept.
    assert dict(sched.schedule().num_scheduled_tokens) == {"r0": 600, "r1": MNBT - 600}


def test_only_prefills_still_running_count(min_chunk):
    min_chunk(MIN_CHUNK)
    sched = _scheduler(HPUAsyncScheduler)
    sched._step_prefill_req_ids = {"dropped"}
    req = _requests(PROMPT)[0]
    sched.add_request(req)
    # A prefill id that never reached self.running must not defer others.
    assert sched._reserve_prefill_lookahead(req, 0, 10) == 10


@pytest.mark.parametrize("value", ["abc", "4k", "1.5"])
def test_invalid_value_disables_rule(min_chunk, value):
    min_chunk(value)
    hpu = _run(_scheduler(HPUAsyncScheduler), (PROMPT, PROMPT), steps=2)
    base = _run(_scheduler(AsyncScheduler), (PROMPT, PROMPT), steps=2)
    assert hpu == base
