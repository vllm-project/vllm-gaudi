# SPDX-License-Identifier: Apache-2.0
"""HPUAsyncScheduler recovery from a failed synchronous KV load (kv_load_failure_policy="recompute").

The step that loads a request's KV also completes its prompt, so under async scheduling the request's first decode
step is scheduled before that step's output, which reports the failure, is processed.
"""
import sys
from pathlib import Path

import pytest
import torch

from vllm.config import (CacheConfig, DeviceConfig, KVTransferConfig, ModelConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler

# The connector is shared with the end-to-end test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "full_tests"))
from kv_load_failure_connector import FAIL_KV_LOAD_KEY, FailingLoadMetadata  # noqa: E402

BLOCK_SIZE = 16
PROMPT = 100  # 6 full blocks come from the connector, 4 tokens are computed in the loading step.
NUM_EXTERNAL = (PROMPT - 1) // BLOCK_SIZE * BLOCK_SIZE


@pytest.fixture
def scheduler():
    model_config = ModelConfig(model="facebook/opt-125m",
                               dtype="float16",
                               seed=42,
                               skip_tokenizer_init=True,
                               max_model_len=1024)
    scheduler_config = SchedulerConfig(max_num_seqs=16,
                                       max_num_batched_tokens=1024,
                                       max_model_len=1024,
                                       enable_chunked_prefill=True,
                                       async_scheduling=True,
                                       is_encoder_decoder=False,
                                       watermark=0.0)
    cache_config = CacheConfig(block_size=BLOCK_SIZE, cache_dtype="auto", enable_prefix_caching=True)
    kv_transfer_config = KVTransferConfig(kv_connector="FailingLoadConnector",
                                          kv_connector_module_path="kv_load_failure_connector",
                                          kv_role="kv_both",
                                          kv_load_failure_policy="recompute")
    vllm_config = VllmConfig(scheduler_config=scheduler_config,
                             model_config=model_config,
                             cache_config=cache_config,
                             kv_transfer_config=kv_transfer_config,
                             device_config=DeviceConfig("cpu"),
                             parallel_config=ParallelConfig())
    num_blocks = 256
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroup.from_layers({
                "layer":
                FullAttentionManager.layer_config(
                    AttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32))
            })
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    return HPUAsyncScheduler(vllm_config=vllm_config,
                             kv_cache_config=kv_cache_config,
                             block_size=BLOCK_SIZE,
                             collect_stats=True,
                             structured_output_manager=StructuredOutputManager(vllm_config))


def _request(req_id: str, fail_kv_load: bool) -> Request:
    init_none_hash(sha256)
    params = SamplingParams(ignore_eos=True,
                            max_tokens=8,
                            extra_args={FAIL_KV_LOAD_KEY: True} if fail_kv_load else None)
    params.update_from_generation_config({}, 50256)
    return Request(request_id=req_id,
                   prompt_token_ids=list(range(1, PROMPT + 1)),
                   sampling_params=params,
                   pooling_params=None,
                   block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256))


def _update(scheduler, scheduler_output, token_id: int, invalid_block_ids: set[int] | None = None) -> dict:
    """Process a step's output: every scheduled request sampled token_id. Returns req_id -> delivered tokens."""
    req_ids = list(scheduler_output.num_scheduled_tokens)
    output = ModelRunnerOutput(req_ids=req_ids,
                               sampled_token_ids=[[token_id] for _ in req_ids],
                               logprobs=None,
                               prompt_logprobs_dict={},
                               pooler_output=[],
                               kv_connector_output=KVConnectorOutput(invalid_block_ids=invalid_block_ids or set()))
    engine_outputs = scheduler.update_from_output(scheduler_output, output)
    return {o.request_id: o.new_token_ids for outs in engine_outputs.values() for o in outs.outputs}


def _num_output_tokens(scheduler_output, req_id: str) -> int:
    cached = scheduler_output.scheduled_cached_reqs
    return cached.num_output_tokens[cached.req_ids.index(req_id)]


def test_failed_load_with_next_step_in_flight(scheduler):
    """The in-flight step's token is dropped, no placeholder is left, and only the prompt is recomputed."""
    req = _request("r", fail_kv_load=True)
    scheduler.add_request(req)

    step_n = scheduler.schedule()
    assert step_n.num_scheduled_tokens["r"] == PROMPT - NUM_EXTERNAL
    assert isinstance(step_n.kv_connector_metadata, FailingLoadMetadata)
    failed_blocks = step_n.kv_connector_metadata.failed_block_ids
    assert len(failed_blocks) == NUM_EXTERNAL // BLOCK_SIZE

    step_n1 = scheduler.schedule()  # The first decode step, scheduled while step N is in flight.
    assert step_n1.num_scheduled_tokens["r"] == 1
    assert req.num_output_placeholders == 2

    assert "r" not in _update(scheduler, step_n, token_id=11, invalid_block_ids=failed_blocks)
    assert req.num_computed_tokens == 0
    assert req.num_output_placeholders == 0

    step_n2 = scheduler.schedule()
    assert step_n2.num_scheduled_tokens["r"] == PROMPT
    assert _num_output_tokens(step_n2, "r") == 0

    # Step N+1 decoded over the KV that failed to load.
    assert "r" not in _update(scheduler, step_n1, token_id=12)
    assert list(req.output_token_ids) == []
    assert req.num_output_placeholders == 1

    assert _update(scheduler, step_n2, token_id=13)["r"] == [13]
    assert list(req.output_token_ids) == [13]
    assert req.num_output_placeholders == 0
    assert req.num_tokens == PROMPT + 1


def test_shared_failed_block_with_next_step_in_flight(scheduler):
    """A request whose failed blocks another request recomputes resumes after its own last cached token."""
    failing = _request("failing", fail_kv_load=True)
    sharing = _request("sharing", fail_kv_load=False)
    scheduler.add_request(failing)
    scheduler.add_request(sharing)

    step_n = scheduler.schedule()
    # The loaded blocks are cached at allocation, so the second request hits them in the prefix cache.
    assert sharing.num_computed_tokens == PROMPT
    assert step_n.num_scheduled_tokens["sharing"] == PROMPT - NUM_EXTERNAL
    failed_blocks = step_n.kv_connector_metadata.failed_block_ids

    step_n1 = scheduler.schedule()
    assert step_n1.num_scheduled_tokens["sharing"] == 1

    _update(scheduler, step_n, token_id=11, invalid_block_ids=failed_blocks)
    assert failing.num_computed_tokens == 0
    # Back to what was computed before step N, not counting the step in flight.
    assert sharing.num_computed_tokens == NUM_EXTERNAL
    assert sharing.num_output_placeholders == 0

    step_n2 = scheduler.schedule()
    assert step_n2.num_scheduled_tokens["sharing"] == PROMPT - NUM_EXTERNAL
    assert _num_output_tokens(step_n2, "sharing") == 0
    assert "sharing" not in _update(scheduler, step_n1, token_id=12)
    assert _update(scheduler, step_n2, token_id=13)["sharing"] == [13]
    assert list(sharing.output_token_ids) == [13]
