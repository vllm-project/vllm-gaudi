# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from types import SimpleNamespace
from unittest.mock import call, create_autospec
import habana_frameworks.torch  # noqa: F401
from habana_frameworks.torch.utils.internal import is_lazy
from vllm.model_executor.model_loader import get_model

from vllm.model_executor.layers.attention import Attention
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig, set_current_vllm_config)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.core.kv_cache_planning import (estimate_max_model_len, get_global_kv_cache_groups, get_kv_cache_configs)
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData, SchedulerOutput)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheConfig, KVCacheGroup, KVCacheTensor)
from vllm_gaudi.v1._vendored.sample.metadata import SamplingMetadata
from vllm_gaudi.v1._vendored.sample.ops.penalties import apply_all_penalties
import vllm_gaudi.extension.environment as environment
from vllm_gaudi.v1.worker.hpu_model_runner import (HPUModelRunner, assert_output_rows, attn_pad_block_id,
                                                   bucket_ctx_to_kernel_blocks)
from vllm_gaudi.v1.worker.hpu_input_batch import InputBatch, splice_placeholders

BLOCK_SIZE = 128
NUM_BLOCKS = 10
DEVICE = current_platform.device_type


def initialize_kv_cache(runner: HPUModelRunner):
    """
    Only perform necessary steps in HPUModelRunner.initialize_kv_cache()
    """
    attn_spec = AttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_size,
                          layers=["layer.0"],
                          layer_stride=tensor_size,
                          block_stride=attn_spec.page_size_bytes),
        ],
        kv_cache_groups=[KVCacheGroup.from_layers({"layer.0": FullAttentionManager.layer_config(attn_spec)})],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_seqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[kv_cache_config.kv_cache_groups[0].block_size],
        kernel_block_sizes=[kv_cache_config.kv_cache_groups[0].block_size],
    )


#    runner.initialize_attn_backend(kv_cache_config)


def get_vllm_config():
    model_config = ModelConfig(
        model="facebook/opt-125m",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
    )
    # PR #51718: get_kv_cache_configs now calls get_resolved_kv_cache_layout(),
    # which raises "KV cache layout has not been resolved yet" unless the layout
    # is pre-set (normally done once by the engine core). Resolve it here so the
    # unit tests can call get_kv_cache_configs directly, mirroring upstream.
    cache_config.kv_cache_layout = "LBNHC"
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


@pytest.fixture
def model_runner():
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        # We need to update the environment before creating Attention
        environment.set_vllm_config(vllm_config)
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(num_heads, head_size, 0.1)
        runner = HPUModelRunner(vllm_config, DEVICE)
        initialize_kv_cache(runner)
        yield runner


def _schedule_new_request(*req_ids: str) -> SchedulerOutput:
    new_reqs = []
    num_scheduled_tokens = {}
    total_num_scheduled_tokens = 0
    for req_id in req_ids:
        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0], ),
                num_computed_tokens=0,
                lora_request=None,
                prefill_token_ids=[1, 2, 3],
            ))
        num_scheduled_tokens[req_id] = 3
        total_num_scheduled_tokens += num_scheduled_tokens[req_id]

    return SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _is_req_scheduled(model_runner, req_id: str) -> bool:
    return req_id in model_runner.input_batch.req_id_to_index


def _is_req_added(model_runner, req_id: str) -> bool:
    return req_id in model_runner.requests


def _is_sampling_metadata_changed(model_runner, sampling_metadata_before: SamplingMetadata):
    return model_runner.input_batch.sampling_metadata is not (sampling_metadata_before)


def _is_req_state_block_table_match(model_runner, req_id: str) -> bool:
    req_index = model_runner.input_batch.req_id_to_index[req_id]
    block_table = model_runner.input_batch.block_table[0]
    req_state = model_runner.requests[req_id]
    if block_table.num_blocks_per_row[req_index] != len(req_state.block_ids[0]):
        return False
    num_blocks = block_table.num_blocks_per_row[req_index]
    return (block_table.block_table.np[req_index, :num_blocks] == req_state.block_ids[0]).all()


def test_update_states_new_request(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_finished(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # finish req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids={req_id},
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)


def test_update_states_request_rescheduled(model_runner, dist_init):
    """A running request skipped for a step is added back from its cached state."""
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # unschedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    # The full prefill sampled a token: the input of the decode step below.
    model_runner.requests[req_id].output_token_ids.append(4)

    # reschedule req
    cached_req_data = CachedRequestData(
        req_ids=[req_id],
        new_token_ids=[[]],
        new_block_ids=[([1], )],
        num_computed_tokens=[3],
        num_output_tokens=[1],
    )

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)
    # New blocks are appended, never replace the cached ones.
    assert model_runner.requests[req_id].block_ids == ([0, 1], )


def _empty_step(**kwargs) -> SchedulerOutput:
    fields = dict(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    fields.update(kwargs)
    return SchedulerOutput(**fields)


def _new_requests_step(prefill_token_ids: dict[str, list[int]], sampling_params: SamplingParams) -> SchedulerOutput:
    """One step that schedules every request in prefill_token_ids as new (prompt [1, 2, 3])."""
    new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=[1, 2, 3],
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=([0], ),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=token_ids,
        ) for req_id, token_ids in prefill_token_ids.items()
    ]
    num_scheduled_tokens = {req_id: len(token_ids) for req_id, token_ids in prefill_token_ids.items()}
    return _empty_step(scheduled_new_reqs=new_reqs,
                       num_scheduled_tokens=num_scheduled_tokens,
                       total_num_scheduled_tokens=sum(num_scheduled_tokens.values()))


def _new_request_step(req_id: str, prefill_token_ids: list[int], sampling_params: SamplingParams) -> SchedulerOutput:
    return _new_requests_step({req_id: prefill_token_ids}, sampling_params)


def test_update_states_request_preempted_and_resumed(model_runner, dist_init):
    """A preempted request is dropped, then resumes as a new request whose prefill_token_ids hold its outputs."""
    req_id = "req_0"
    sampling_params = SamplingParams(temperature=0.8, seed=7)

    model_runner._update_states(_new_request_step(req_id, [1, 2, 3], sampling_params))
    generator = model_runner.requests[req_id].generator
    assert generator is not None

    model_runner._update_states(_empty_step(preempted_req_ids={req_id}))
    assert not _is_req_added(model_runner, req_id)
    assert not _is_req_scheduled(model_runner, req_id)

    model_runner._update_states(_new_request_step(req_id, [1, 2, 3, 7, 8], sampling_params))
    req_state = model_runner.requests[req_id]
    row = model_runner.input_batch.req_id_to_index[req_id]
    assert req_state.output_token_ids == [7, 8]
    assert req_state.num_prompt_tokens == 3
    # The batch re-prefills the earlier outputs together with the prompt.
    assert model_runner.input_batch.num_prompt_tokens[row] == 5
    assert model_runner.input_batch.token_ids_cpu[row, :5].tolist() == [1, 2, 3, 7, 8]
    assert req_state.generator is generator
    assert req_id not in model_runner._preempted_generators


def test_update_states_dropped_request_leaves_previous_step_rows(model_runner, dist_init):
    """A dropped request's previous-step row is removed, so if it resumes in the same step, _prepare_input_ids cannot
    scatter its discarded in-flight sample into it."""
    req_id = "req_0"
    model_runner._update_states(_new_request_step(req_id, [1, 2, 3], SamplingParams()))
    model_runner.input_batch.prev_req_id_to_index = {req_id: 0, "req_1": 1}
    model_runner._update_states(_empty_step(preempted_req_ids={req_id}))
    assert model_runner.input_batch.prev_req_id_to_index == {"req_1": 1}


def test_update_states_resumed_request_does_not_rescore_prompt(model_runner, dist_init):
    """Prompt logprobs go out once: a request resumed with outputs already sent them, one preempted before any
    output has not."""
    params = SamplingParams(prompt_logprobs=1)
    scored = model_runner.input_batch.num_prompt_logprobs
    # Both requests go in one step: a request left out of a step leaves the batch, and its scoring entry with it.
    model_runner._update_states(_new_requests_step({"req_0": [1, 2, 3], "req_1": [1, 2, 3]}, params))
    assert {"req_0", "req_1"} <= scored.keys()

    model_runner._update_states(_empty_step(preempted_req_ids={"req_0", "req_1"}))
    model_runner._update_states(_new_requests_step({"req_0": [1, 2, 3, 7, 8], "req_1": [1, 2, 3]}, params))
    assert {"req_0", "req_1"} <= model_runner.input_batch.req_id_to_index.keys()
    assert "req_0" not in scored
    assert "req_1" in scored


@pytest.mark.parametrize(
    "context_len, query_len, expected",
    [
        # Within the prompt.
        (0, 4, [[0, 1, 2, 3], [0, 1, 1, 3], [0, 2, 2, 3]]),
        # Prompt tail, then generated positions (index + delta).
        (2, 4, [[2, 3, 3, 4], [1, 3, 3, 4], [2, 3, 3, 4]]),
        # Three generated positions: a (3, 3) block that must not be transposed.
        (0, 7, [[0, 1, 2, 3, 3, 4, 5], [0, 1, 1, 3, 3, 4, 5], [0, 2, 2, 3, 3, 4, 5]]),
        # Entirely past the prompt, as for a resumed request whose prefix hit covers some outputs.
        (6, 2, [[5, 6], [5, 6], [5, 6]]),
    ])
def test_align_and_pad_mrope_positions(model_runner, context_len, query_len, expected):
    """Positions past the prompt follow the scheduled token index, also when the computed tokens already reach past
    the prompt."""
    model_runner.requests["req_0"] = SimpleNamespace(
        mrope_positions=torch.tensor([[0, 1, 2, 3], [0, 1, 1, 3], [0, 2, 2, 3]], dtype=torch.int32),
        mrope_position_delta=-1,
    )
    positions = model_runner._align_and_pad_mrope_positions(["req_0"], [context_len], [query_len], (1, query_len), -1)
    assert positions.tolist() == expected


class _ReadyEvent:
    """Stands in for the copy-ready event of a finished async device-to-host copy."""

    def synchronize(self):
        pass


def _skip_then_reschedule(model_runner, req_id: str, num_placeholders: int = 1):
    # Earlier steps sampled tokens for req_id; under async scheduling without penalties only -1 placeholders are
    # known, one per step.
    model_runner.use_async_scheduling = True
    model_runner._update_states(_new_request_step(req_id, [1, 2, 3], SamplingParams()))
    model_runner.requests[req_id].output_token_ids.extend([-1] * num_placeholders)
    # Step 2 leaves the running request out.
    model_runner._update_states(_empty_step())
    assert not _is_req_scheduled(model_runner, req_id)
    # Step 3 schedules it again.
    cached = CachedRequestData(req_ids=[req_id],
                               new_token_ids=[[]],
                               new_block_ids=[None],
                               num_computed_tokens=[2 + num_placeholders],
                               num_output_tokens=[num_placeholders])
    model_runner._update_states(
        _empty_step(scheduled_cached_reqs=cached, num_scheduled_tokens={req_id: 1}, total_num_scheduled_tokens=1))


def test_update_states_async_skipped_request_placeholder_repaired(model_runner, dist_init):
    req_id = "req_0"
    model_runner._last_async_copy = (torch.tensor([[42]], dtype=torch.int32), _ReadyEvent(), {req_id: 0})
    _skip_then_reschedule(model_runner, req_id)
    row = model_runner.input_batch.req_id_to_index[req_id]
    assert model_runner.requests[req_id].output_token_ids == [42]
    assert model_runner.input_batch.token_ids_cpu[row, :4].tolist() == [1, 2, 3, 42]
    assert model_runner.num_skipped_placeholder_repairs == 1
    assert req_id not in model_runner._pending_placeholder_repair


def test_update_states_async_skipped_request_keeps_older_placeholders(model_runner, dist_init):
    """Only the newest placeholder is this step's input; older ones are already in the KV cache and stay."""
    req_id = "req_0"
    model_runner._last_async_copy = (torch.tensor([[42]], dtype=torch.int32), _ReadyEvent(), {req_id: 0})
    _skip_then_reschedule(model_runner, req_id, num_placeholders=3)
    row = model_runner.input_batch.req_id_to_index[req_id]
    assert model_runner.requests[req_id].output_token_ids == [-1, -1, 42]
    assert model_runner.input_batch.num_tokens[row] == 6
    assert model_runner.input_batch.token_ids_cpu[row, 5] == 42


def test_update_states_async_skipped_request_keeps_only_its_row(model_runner, dist_init):
    """A skipped request keeps the step's CPU copy and its own row in it, not the step's whole request -> row map."""
    req_id = "req_0"
    model_runner.use_async_scheduling = True
    model_runner._last_async_copy = (torch.tensor([[41], [42]], dtype=torch.int32), _ReadyEvent(), {
        "req_1": 0,
        req_id: 1
    })
    model_runner._update_states(_new_request_step(req_id, [1, 2, 3], SamplingParams()))
    model_runner.requests[req_id].output_token_ids.append(-1)
    model_runner._update_states(_empty_step())
    assert model_runner._pending_placeholder_repair[req_id][2] == 1

    cached = CachedRequestData(req_ids=[req_id],
                               new_token_ids=[[]],
                               new_block_ids=[None],
                               num_computed_tokens=[3],
                               num_output_tokens=[1])
    model_runner._update_states(
        _empty_step(scheduled_cached_reqs=cached, num_scheduled_tokens={req_id: 1}, total_num_scheduled_tokens=1))
    assert model_runner.requests[req_id].output_token_ids == [42]


@pytest.mark.parametrize("output_ids, new_ids, expected", [
    ([5, -1], [42], [5, 42]),
    ([-1, -1, -1], [42], [-1, -1, 42]),
    ([5, -1, -1], [42, 43], [5, 42, 43]),
    ([5, -1], [42, -1], [5, 42]),
    ([5, -1], [-1], [5, -1]),
    ([5, -1, -1], [-1], [5, -1, -1]),
    ([5, 6], [42], [5, 6]),
])
def test_splice_placeholders(output_ids, new_ids, expected):
    splice_placeholders(output_ids, new_ids)
    assert output_ids == expected


def test_assert_output_rows_accepts_rows_in_batch_order():
    assert_output_rows(["a", "b"], 2)
    # Only the first len(req_ids) batch entries are compared.
    assert_output_rows(["a", "b"], 2, ["a", "b", "c"])


@pytest.mark.parametrize("req_ids, num_rows, batch_req_ids", [
    (["a", "b"], 3, None),
    (["a", "b"], 1, None),
    (["b", "a"], 2, ["a", "b"]),
    (["a", "c"], 2, ["a", "b", "c"]),
])
def test_assert_output_rows_rejects_mismatch(req_ids, num_rows, batch_req_ids):
    with pytest.raises(AssertionError):
        assert_output_rows(req_ids, num_rows, batch_req_ids)


def test_update_states_async_skipped_request_without_copy_fails(model_runner, dist_init):
    model_runner._last_async_copy = None
    with pytest.raises(RuntimeError, match="placeholder"):
        _skip_then_reschedule(model_runner, "req_0")


def _readd_after_skip(model_runner, req_id: str, output_ids: list[int], num_computed_tokens: int):
    model_runner._last_async_copy = (torch.tensor([[44]], dtype=torch.int32), _ReadyEvent(), {req_id: 0})
    model_runner.use_async_scheduling = True
    model_runner._update_states(_new_request_step(req_id, [1, 2, 3], SamplingParams()))
    model_runner.requests[req_id].output_token_ids.extend(output_ids)
    model_runner._update_states(_empty_step())
    cached = CachedRequestData(req_ids=[req_id],
                               new_token_ids=[[]],
                               new_block_ids=[None],
                               num_computed_tokens=[num_computed_tokens],
                               num_output_tokens=[len(output_ids)])
    num_new = 3 + len(output_ids) - num_computed_tokens
    model_runner._update_states(
        _empty_step(scheduled_cached_reqs=cached,
                    num_scheduled_tokens={req_id: max(num_new, 1)},
                    total_num_scheduled_tokens=max(num_new, 1)))


def test_update_states_async_penalty_splice_keeps_list_for_readd(model_runner, dist_init):
    """A penalty request joining the batch splices every request's newest placeholder. It must not drop the older
    ones of a request that decoded without a consumer, or that request's list would end short of its KV cache and a
    later skip and re-add would read past it."""
    req_id = "req_0"
    model_runner.use_async_scheduling = True
    model_runner._update_states(_new_request_step(req_id, [1, 2, 3], SamplingParams()))
    outputs = model_runner.requests[req_id].output_token_ids
    outputs.extend([-1, -1, -1])
    batch = model_runner.input_batch
    batch.prev_req_id_to_index = {req_id: 0}
    batch.sampled_token_ids_cpu = torch.tensor([[41]], dtype=torch.int32)
    batch.async_copy_ready_event = _ReadyEvent()
    batch.update_async_output_token_ids()
    assert outputs == [-1, -1, 41]
    # The next step's -1, then a skipped step and the re-add: the input position 6 is the repaired id.
    outputs.append(-1)
    model_runner._last_async_copy = (torch.tensor([[44]], dtype=torch.int32), _ReadyEvent(), {req_id: 0})
    model_runner._update_states(_empty_step())
    cached = CachedRequestData(req_ids=[req_id],
                               new_token_ids=[[]],
                               new_block_ids=[None],
                               num_computed_tokens=[6],
                               num_output_tokens=[4])
    model_runner._update_states(
        _empty_step(scheduled_cached_reqs=cached, num_scheduled_tokens={req_id: 1}, total_num_scheduled_tokens=1))
    row = model_runner.input_batch.req_id_to_index[req_id]
    assert model_runner.input_batch.token_ids_cpu[row, 6] == 44


def test_update_states_async_readded_request_shorter_than_kv_cache_fails(model_runner, dist_init):
    """Four outputs were sampled but the list holds two: the input position (6) lies past the re-added row, which
    would feed a stale token, so the re-add refuses instead."""
    with pytest.raises(RuntimeError, match="past its 5 known tokens"):
        _readd_after_skip(model_runner, "req_0", [41, -1], num_computed_tokens=6)


def test_update_states_async_readded_request_with_placeholder_to_compute_fails(model_runner, dist_init):
    """A KV-load failure rolled num_computed_tokens back to the prompt: the older placeholder must be recomputed."""
    with pytest.raises(RuntimeError, match="inside the tokens to compute"):
        _readd_after_skip(model_runner, "req_0", [5, -1, -1], num_computed_tokens=3)


def test_update_states_async_readded_request_with_placeholder_as_prompt_fails(model_runner, dist_init):
    """The tokens to compute (positions 5 and 6) are real ids, but with more than the input left to compute every
    known token goes into the penalty prompt tensor, which does not mask the older placeholders."""
    with pytest.raises(RuntimeError, match="re-adds as prompt"):
        _readd_after_skip(model_runner, "req_0", [-1, -1, 41, 42], num_computed_tokens=5)


def test_update_states_async_readded_request_with_older_placeholders_passes(model_runner, dist_init):
    """Only the input is left to compute: older placeholders stay outputs, which the penalty kernel masks."""
    _readd_after_skip(model_runner, "req_0", [-1, -1, 41, 42], num_computed_tokens=6)
    row = model_runner.input_batch.req_id_to_index["req_0"]
    assert model_runner.input_batch.num_prompt_tokens[row] == 3


def test_update_states_kv_load_failure_drops_discarded_output(model_runner, dist_init, monkeypatch):
    """A sync KV-load failure rolls the request back and the scheduler discards the token the failed step sampled.
    The runner drops it too, and the chunked recompute samples once, at the end of its last chunk."""
    req_id = "req_0"
    model_runner.use_async_scheduling = False
    # Resumed with outputs [4, 5] and a full external KV hit: only the last token is left to compute.
    new_req = NewRequestData(req_id=req_id,
                             prompt_token_ids=[1, 2, 3],
                             mm_features=[],
                             sampling_params=SamplingParams(),
                             pooling_params=None,
                             block_ids=([0], ),
                             num_computed_tokens=4,
                             lora_request=None,
                             prefill_token_ids=[1, 2, 3, 4, 5])
    model_runner._update_states(
        _empty_step(scheduled_new_reqs=[new_req], num_scheduled_tokens={req_id: 1}, total_num_scheduled_tokens=1))
    # That step sampled 6, recorded as sample_tokens does, but its KV load failed.
    batch = model_runner.input_batch
    row = batch.req_id_to_index[req_id]
    model_runner.requests[req_id].output_token_ids.append(6)
    batch.token_ids_cpu[row, 5] = 6
    batch.num_tokens[row] = batch.num_tokens_no_spec[row] = 6
    monkeypatch.setattr(model_runner, "_can_merge_prefill_contents", lambda lhs, rhs: False)

    def recompute_chunk(num_computed_tokens, num_scheduled_tokens):
        cached = CachedRequestData(req_ids=[req_id],
                                   new_token_ids=[[]],
                                   new_block_ids=[None],
                                   num_computed_tokens=[num_computed_tokens],
                                   num_output_tokens=[2])
        model_runner._update_states(
            _empty_step(scheduled_cached_reqs=cached,
                        num_scheduled_tokens={req_id: num_scheduled_tokens},
                        total_num_scheduled_tokens=num_scheduled_tokens))
        contents, _ = model_runner._extract_prefill_batch_contents(1, 0, [num_scheduled_tokens])
        return [c.logits_positions for c in contents if c.req_ids]

    # Rolled back to 0; the token budget splits the recompute of the 5 known tokens into 4 + 1.
    assert recompute_chunk(0, 4) == [[[]]]
    assert model_runner.requests[req_id].output_token_ids == [4, 5]
    assert batch.num_tokens_no_spec[row] == batch.num_tokens[row] == batch.num_prompt_tokens[row] == 5
    assert recompute_chunk(4, 1) == [[[0]]]


@pytest.mark.parametrize("use_async_scheduling", [False, True])
def test_partial_prefill_chunk_rows_with_structured_output(model_runner, dist_init, use_async_scheduling):
    """A chunk that does not finish its prompt samples nothing under sync scheduling, also right after a step with a
    grammar: nothing would discard its token. Async scheduling gives it one row, marked invalid."""
    req_id = "req_0"
    model_runner.use_async_scheduling = use_async_scheduling
    model_runner.use_structured_output = True  # The previous step had a grammar.
    model_runner.invalid_req_indices = []
    new_req = NewRequestData(req_id=req_id,
                             prompt_token_ids=[1, 2, 3, 4, 5],
                             mm_features=[],
                             sampling_params=SamplingParams(),
                             pooling_params=None,
                             block_ids=([0], ),
                             num_computed_tokens=0,
                             lora_request=None,
                             prefill_token_ids=[1, 2, 3, 4, 5])
    model_runner._update_states(
        _empty_step(scheduled_new_reqs=[new_req], num_scheduled_tokens={req_id: 4}, total_num_scheduled_tokens=4))
    contents, _ = model_runner._extract_prefill_batch_contents(1, 0, [4])
    positions = [c.logits_positions for c in contents if c.req_ids]
    if use_async_scheduling:
        assert positions == [[[3]]] and model_runner.invalid_req_indices == [0]
    else:
        assert positions == [[[]]] and model_runner.invalid_req_indices == []


def test_maybe_setup_kv_connector_handles_preemptions_before_load(monkeypatch):
    """The connector flushes saves from reused blocks before this step's metadata is bound and its loads start."""
    import vllm_gaudi.v1.worker.hpu_model_runner as runner_module
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1

    connector = create_autospec(KVConnectorBase_V1, instance=True)
    monkeypatch.setattr(runner_module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(runner_module, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(runner_module, "get_forward_context", lambda: None)
    metadata = object()

    runner_module.HpuKVConnectorModelRunnerMixin.maybe_setup_kv_connector(
        SimpleNamespace(kv_connector_metadata=metadata))

    assert connector.mock_calls == [
        call.handle_preemptions(metadata),
        call.bind_connector_metadata(metadata),
        call.start_load_kv(None),
    ]


def test_selective_sampling_metadata_passes_outputs_only_to_penalty_rows(model_runner, dist_init):
    """A row without penalties gets an empty output list, so its long run of placeholders does not widen the penalty
    tensor, and the penalized logits stay the same."""
    req_ids = ("req_0", "req_1", "req_2", "req_3")
    step = _new_requests_step({req_id: [1, 2, 3] for req_id in req_ids}, SamplingParams())
    step.scheduled_new_reqs[0].sampling_params = SamplingParams(repetition_penalty=1.5, frequency_penalty=0.5)
    step.scheduled_new_reqs[2].sampling_params = SamplingParams(presence_penalty=0.3)
    model_runner._update_states(step)
    batch = model_runner.input_batch
    # req_3 has no penalty but bad words, which read its outputs too.
    batch.bad_words_token_ids[batch.req_id_to_index["req_3"]] = [[9]]
    for req_id, outputs in zip(req_ids, ([5, 6, -1, 7], [-1] * 50 + [8], [9, 10], [11])):
        model_runner.requests[req_id].output_token_ids.extend(outputs)
    selected = [(req_id, model_runner.requests[req_id].output_token_ids) for req_id in req_ids]

    metadata = batch.make_selective_sampling_metadata(selected)
    assert metadata.output_token_ids == [[5, 6, -1, 7], [], [9, 10], [11]]

    logits = torch.randn(len(req_ids), model_runner.model_config.get_vocab_size(), device=DEVICE)
    penalties = (metadata.prompt_token_ids, metadata.presence_penalties, metadata.frequency_penalties,
                 metadata.repetition_penalties)
    full = apply_all_penalties(logits.clone(), *penalties, [list(ids) for _, ids in selected])
    filtered = apply_all_penalties(logits.clone(), *penalties, metadata.output_token_ids)
    assert torch.equal(full.cpu(), filtered.cpu())


def test_get_nans_in_logits(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    scheduler_output = _schedule_new_request(*req_ids)
    model_runner._update_states(scheduler_output)

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor([
        [1.0, float('nan'), 3.0],
        [4.0, float('nan'), float('nan')],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 1, "req_1": 2}

    logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, float('nan'), float('nan')],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {"req_0": 0, "req_1": 2}

    result = model_runner._get_nans_in_logits(logits=None)
    assert result == {"req_0": 0, "req_1": 0}

    logits = torch.tensor([
        [1.0, float('nan'), 3.0],
    ], device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {'req_0': 1, 'req_1': 0}

    logits = torch.tensor([
        [float('nan'), float('nan'), 2.0],
        [1.0, 2.0, 3.0],
        [float('nan'), 2.0, 3.0],
    ],
                          device=DEVICE)
    result = model_runner._get_nans_in_logits(logits)
    assert result == {'req_0': 2, 'req_1': 0}


def test_update_states_no_changes(model_runner, dist_init):
    req_id = "req_0"

    # new req
    scheduler_output = _schedule_new_request(req_id)

    model_runner._update_states(scheduler_output)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)

    # schedule req
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner.input_batch.sampling_metadata
    model_runner._update_states(scheduler_output)
    assert not _is_sampling_metadata_changed(model_runner, metadata_before)
    assert _is_req_added(model_runner, req_id)
    assert _is_req_scheduled(model_runner, req_id)
    assert _is_req_state_block_table_match(model_runner, req_id)


def test_update_states_request_unscheduled(model_runner, dist_init):
    req_ids = ("req_0", "req_1")

    # new reqs
    scheduler_output = _schedule_new_request(*req_ids)

    model_runner._update_states(scheduler_output)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert _is_req_scheduled(model_runner, req_ids[1])

    # unschedule req_1
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_ids[0]: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    metadata_before = model_runner._update_states(scheduler_output)
    assert _is_sampling_metadata_changed(model_runner, metadata_before)

    assert _is_req_added(model_runner, req_ids[0])
    assert _is_req_scheduled(model_runner, req_ids[0])

    assert _is_req_added(model_runner, req_ids[1])
    assert not _is_req_scheduled(model_runner, req_ids[1])


def test_update_config(model_runner):
    # Simple update
    model_runner.update_config({"load_config": {"load_format": "dummy"}})
    assert model_runner.load_config.load_format == "dummy"
    # Raise error on non-existing config
    with pytest.raises(AssertionError):
        model_runner.update_config({"do_not_exist_config": "dummy"})


def test_reload_weights_before_load_model(model_runner):
    with pytest.raises(AssertionError):
        model_runner.reload_weights()


def test_init_kv_cache_with_kv_sharing_invalid_target_layer_order(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} must come before the current layer"
    environment.set_vllm_config(get_vllm_config())
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
                kv_sharing_target_layer_name=layer_1,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_layer_not_exist(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    invalid_layer = "model.layers.0.cross_attn.attn"
    error_msg = f"{invalid_layer} is not a valid Attention layer in the model"
    environment.set_vllm_config(get_vllm_config())
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                # invalid layer: cross_attn.atn doesn't exist!
                kv_sharing_target_layer_name=invalid_layer,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_with_kv_sharing_target_same_as_current(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    error_msg = f"{layer_1} cannot be the same as the current layer"
    environment.set_vllm_config(get_vllm_config())
    with pytest.raises(ValueError, match=error_msg):
        fwd_context = {
            # initialization below will fail because target layer is invalid;
            # the target layer needs to come before layer 1
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name=layer_1,
            )
        }
        # suppress var not used error
        assert fwd_context is not None


def test_init_kv_cache_without_kv_sharing(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    environment.set_vllm_config(vllm_config)
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1: Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
            )
        }
        # suppress var not used error
        assert fwd_context is not None
    # Set high context length to test max context length estimation
    vllm_config.model_config.max_model_len = 3_000_000
    vllm_ctx = vllm_config.compilation_config.static_forward_context
    runner = HPUModelRunner(vllm_config, DEVICE)
    layer_configs = runner.get_kv_cache_layer_configs()
    assert len(layer_configs) == 2
    #assert len(runner.shared_kv_cache_layers) == 0

    available_memory = 20 * GiB_bytes
    # page size for layer 0's kv_cache_spec *should be* 256KB:
    # block_size*num_heads*head_size*dtype_size*coeff = 128*8*64*2*2 = 262144
    page_size = layer_configs[layer_0].spec.page_size_bytes
    assert page_size == 262144
    num_expected_blocks = 40960  # 20GB / 256KB (page_size) / 2 (num layers)
    assert num_expected_blocks == available_memory // page_size // 2
    global_kv_cache_groups = get_global_kv_cache_groups(vllm_config, [layer_configs])
    kv_cache_config = get_kv_cache_configs(vllm_config, [layer_configs], [available_memory], global_kv_cache_groups)[0]
    assert kv_cache_config.num_blocks == num_expected_blocks
    # PR #51718 coalesces same-spec layers into ONE backing KVCacheTensor
    # (layers=[layer_0, layer_1]) whose size spans the full allocation.
    assert len(kv_cache_config.kv_cache_tensors) == 1
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory

    max_context_len =\
        estimate_max_model_len(vllm_config, layer_configs, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate one block worth of memory per layer
    kv_cache_config.num_blocks = 1
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        page_size = layer_configs[kv_cache_tensor.layers[0]].spec.page_size_bytes
        kv_cache_tensor.size = page_size * len(kv_cache_tensor.layers)
        kv_cache_tensor.layer_stride = page_size
        kv_cache_tensor.block_stride = page_size

    runner.initialize_kv_cache(kv_cache_config)

    layer_0_kv = vllm_ctx[layer_0].kv_cache[0]
    layer_1_kv = vllm_ctx[layer_1].kv_cache[0]
    # check layer 1 kv cache does NOT share memory with layer 0
    assert id(layer_1_kv) != id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config.kv_cache_groups[0].layer_names[1] == layer_1


def test_init_kv_cache_with_kv_sharing_valid(default_vllm_config: None):
    torch.set_default_dtype(torch.bfloat16)
    layer_0 = "model.layers.0.self_attn.attn"
    layer_1 = "model.layers.1.self_attn.attn"
    vllm_config = get_vllm_config()
    environment.set_vllm_config(vllm_config)
    with set_current_vllm_config(vllm_config):
        fwd_context = {
            layer_0:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_0,
            ),
            layer_1:
            Attention(
                num_heads=8,
                head_size=64,
                scale=1.0,
                prefix=layer_1,
                kv_sharing_target_layer_name="model.layers.0.self_attn.attn",
            )
        }
        # suppress var not used error
        assert fwd_context is not None
    # Set high context length to test max context length estimation
    vllm_config.model_config.max_model_len = 3_000_000
    vllm_ctx = vllm_config.compilation_config.static_forward_context
    runner = HPUModelRunner(vllm_config, DEVICE)
    layer_configs = runner.get_kv_cache_layer_configs()
    assert len(layer_configs) == 1
    assert layer_0 in layer_configs
    assert runner.shared_kv_cache_layers[layer_1] == layer_0

    available_memory = 20 * GiB_bytes
    # page size for layer 0's kv_cache_spec is 256KB
    # with KV sharing, we can allocate (available_mem//page_size//1) blocks
    # which is twice as many as without KV sharing
    page_size = 128 * 8 * 64 * 2 * 2  # 128 for block_size, 2 for K+V, 2 for bfloat16
    num_expected_blocks = available_memory / page_size  # 20GB / 256KB
    global_kv_cache_groups = get_global_kv_cache_groups(vllm_config, [layer_configs])
    kv_cache_config = get_kv_cache_configs(vllm_config, [layer_configs], [available_memory], global_kv_cache_groups)[0]
    assert kv_cache_config.num_blocks == num_expected_blocks
    assert len(kv_cache_config.kv_cache_tensors) == 1
    # Each layer now has twice the available memory for KV cache
    # compared to no KV sharing
    assert kv_cache_config.kv_cache_tensors[0].size == available_memory

    max_context_len =\
        estimate_max_model_len(vllm_config, layer_configs, 5 * GiB_bytes)
    # max context len with KV sharing should be 2x as large as without
    assert max_context_len == 2 * 1310720

    # important: override tensor size to prevent large mem alloc during test
    # this will only allocate 1 block worth of memory (32kb)
    kv_cache_config.num_blocks = 1
    kv_cache_config.kv_cache_tensors[0].size =\
        layer_configs[layer_0].spec.page_size_bytes

    runner.initialize_kv_cache(kv_cache_config)

    layer_0_kv = vllm_ctx[layer_0].kv_cache[0]
    layer_1_kv = vllm_ctx[layer_1].kv_cache[0]
    # check layer 1 kv cache shares memory with layer 0
    assert id(layer_1_kv) == id(layer_0_kv)

    # check layer 1 added to kv cache group's layer names
    assert len(kv_cache_config.kv_cache_groups) == 1
    assert len(kv_cache_config.kv_cache_groups[0].layer_names) == 2
    assert kv_cache_config.kv_cache_groups[0].layer_names[0] == layer_0
    assert kv_cache_config.kv_cache_groups[0].layer_names[1] == layer_1


@pytest.mark.skipif(is_lazy(), reason="Test skipped because lazy mode is enabled.")
def test_model_torch_regional_compilation(default_vllm_config: None, dist_init, model_runner):
    from vllm_gaudi.utils import HPUCompileConfig
    from vllm.model_executor.models.opt import OPTDecoderLayer
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding  # noqa
    from vllm.model_executor.layers.layernorm import LayerNorm
    from torch._dynamo.eval_frame import OptimizedModule

    def assert_compilation(model, layer_name, module):
        submodule = model.get_submodule(layer_name)
        assert isinstance(submodule, OptimizedModule), (
            f"Layer: '{module.__name__}' was not wrapped with OptimizedModule"  # noqa
        )
        assert isinstance(submodule._orig_mod, module), (
            f"_orig_mod is different from the original module: '{module.__name__}'"  # noqa
        )

    vllm_config = get_vllm_config()
    model = get_model(vllm_config=vllm_config)
    model_runner.compile_config = HPUCompileConfig()
    model_runner.regional_compilation_layers_list = [LayerNorm, VocabParallelEmbedding]

    model_runner._regional_compilation(model)

    for i in range(len(model.get_submodule("model.decoder.layers"))):
        assert_compilation(model, f"model.decoder.layers.{i}", OPTDecoderLayer)
    assert_compilation(model, "lm_head", VocabParallelEmbedding)
    assert_compilation(model, "model.decoder.final_layer_norm", LayerNorm)
    assert_compilation(model, "model.decoder.embed_tokens", VocabParallelEmbedding)


def test_max_cudagraph_capture_size_defaults_to_max_num_batched_tokens(model_runner):
    """max_cudagraph_capture_size defaults to max_num_batched_tokens when not configured."""
    assert model_runner.max_cudagraph_capture_size == model_runner.max_num_batched_tokens


def test_max_cudagraph_capture_size_uses_explicit_value():
    """max_cudagraph_capture_size uses the configured value when explicitly set."""
    vllm_config = get_vllm_config()
    vllm_config.compilation_config.max_cudagraph_capture_size = 256
    with set_current_vllm_config(vllm_config):
        environment.set_vllm_config(vllm_config)
        num_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = vllm_config.model_config.get_head_size()
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(num_heads, head_size, 0.1)
        runner = HPUModelRunner(vllm_config, DEVICE)
        assert runner.max_cudagraph_capture_size == 256


@pytest.mark.parametrize(
    "is_prompt,batch_size,seq_len,num_blocks,block_size,max_capture,expected",
    [
        # Prefill within limits → use graphs
        (True, 1, 128, 0, 128, 512, True),
        # Prefill exceeding limits → skip graphs
        (True, 1, 256, 4, 128, 512, False),
        # Prefill at exact boundary → use graphs
        (True, 1, 256, 2, 128, 512, True),
        # Prefill just over boundary → skip graphs
        (True, 1, 256, 2, 128, 511, False),
        # Decode never skips graphs even with many tokens
        (False, 256, 1, 100, 128, 512, True),
        # Decode with many blocks → still use graphs
        (False, 64, 1, 1000, 128, 512, True),
    ])
def test_use_graphs(model_runner, is_prompt, batch_size, seq_len, num_blocks, block_size, max_capture, expected):
    model_runner.max_cudagraph_capture_size = max_capture
    attn_metadata = SimpleNamespace(is_prompt=is_prompt,
                                    block_size=block_size,
                                    seq_len=lambda: seq_len,
                                    num_blocks=lambda: num_blocks)
    result = model_runner._use_graphs(attn_metadata, batch_size)
    assert result == expected


def test_use_graphs_enforce_eager(model_runner):
    """When enforce_eager is set, never use graphs."""
    orig = model_runner.model_config.enforce_eager
    try:
        model_runner.model_config.enforce_eager = True
        attn_metadata = SimpleNamespace(is_prompt=False, block_size=128, seq_len=lambda: 1, num_blocks=lambda: 0)
        assert model_runner._use_graphs(attn_metadata, 1) is False
    finally:
        model_runner.model_config.enforce_eager = orig


def _attn_kv_layout(num_blocks, spec_block_sizes, kernel_block_size, hybrid_cache):
    """One attention layer per KV cache group, with the flat KV tensors that
    initialize_kv_cache allocates: the hybrid cache sizes them in kernel
    blocks, the other layouts in KV-manager blocks."""
    kv_cache_groups, kv_caches = [], {}
    for gid, spec_block_size in enumerate(spec_block_sizes):
        spec = AttentionSpec(block_size=spec_block_size, num_kv_heads=1, head_size=1, dtype=torch.bfloat16)
        kv_cache_groups.append(KVCacheGroup.from_layers({f"layer.{gid}": FullAttentionManager.layer_config(spec)}))
        if hybrid_cache:
            num_slots = (num_blocks * (spec_block_size // kernel_block_size) + 1) * kernel_block_size
        else:
            num_slots = (num_blocks + 1) * spec_block_size
        cache = torch.empty((num_slots, 1, 1), dtype=torch.bfloat16, device="cpu")
        kv_caches[f"layer.{gid}"] = (cache, cache, None, None)
    return kv_cache_groups, kv_caches


@pytest.mark.parametrize("hybrid_cache", [True, False])
# 128 is not split; the others are KV-manager block sizes of hybrid models,
# 4 to 17 kernel blocks each.
@pytest.mark.parametrize("spec_block_size", [128, 512, 640, 896, 1152, 1664, 2176])
def test_attn_pad_block_id_follows_real_blocks(hybrid_cache, spec_block_size):
    """Mamba-like models split their KV-manager blocks into 128-token kernel
    blocks in every layout; padded decode lanes must write past all real
    slots."""
    num_blocks, kernel_block_size = 10, 128
    kv_cache_groups, kv_caches = _attn_kv_layout(num_blocks, [spec_block_size], kernel_block_size, hybrid_cache)
    pad_block_id = attn_pad_block_id(num_blocks, kernel_block_size, kv_cache_groups, {0: kernel_block_size}, kv_caches)
    assert pad_block_id == num_blocks * (spec_block_size // kernel_block_size)
    # The slots _create_decode_input_data cycles padded decode lanes through.
    pad_slot_base = pad_block_id * kernel_block_size
    pad_slots = range(pad_slot_base, pad_slot_base + kernel_block_size)
    assert pad_slots.start >= num_blocks * spec_block_size
    assert pad_slots.stop <= kv_caches["layer.0"][0].shape[0]


def test_attn_pad_block_id_unsplit_block_tables():
    """Without kernel block sizes the block tables are not split, and the pad
    block is the extra KV-manager block."""
    num_blocks, block_size = 10, 128
    kv_cache_groups, kv_caches = _attn_kv_layout(num_blocks, [block_size], block_size, hybrid_cache=False)
    assert attn_pad_block_id(num_blocks, block_size, kv_cache_groups, {}, kv_caches) == num_blocks


def test_attn_pad_block_id_rejects_mixed_splits():
    kv_cache_groups, kv_caches = _attn_kv_layout(10, [256, 640], 128, hybrid_cache=True)
    with pytest.raises(ValueError, match="different numbers of kernel blocks"):
        attn_pad_block_id(10, 128, kv_cache_groups, {0: 128, 1: 128}, kv_caches)


def test_attn_pad_block_id_rejects_pad_block_in_real_slots():
    """A pad id counted in KV-manager blocks while slots are counted in kernel
    blocks lands in a real block."""
    kv_cache_groups, kv_caches = _attn_kv_layout(10, [512], 128, hybrid_cache=False)
    with pytest.raises(RuntimeError, match="must follow its 5120 real slots"):
        attn_pad_block_id(10, 128, kv_cache_groups, {}, kv_caches)


def test_attn_pad_block_id_rejects_missing_pad_block():
    kv_cache_groups, kv_caches = _attn_kv_layout(10, [512], 128, hybrid_cache=True)
    cache = kv_caches["layer.0"][0][:-128]
    kv_caches["layer.0"] = (cache, cache, None, None)
    with pytest.raises(RuntimeError, match="within its 5120 allocated ones"):
        attn_pad_block_id(10, 128, kv_cache_groups, {0: 128}, kv_caches)


@pytest.mark.parametrize(
    "num_blocks, block_size, kernel_block_size, expected",
    [
        (0, 128, 528, 0),
        (1, 128, 528, 1),  # one 128-token bucket block needs a whole 528-token kernel block
        (10, 128, 528, 3),  # 1280 tokens
        (63, 128, 528, 16),  # 8064 tokens; the integer ratio 128 // 528 gave 0 and dropped the context
        (4, 528, 128, 17),  # 2112 tokens; the integer ratio 528 // 128 = 4 gave 16
        (3, 1152, 128, 27),  # exact split
        (7, 128, 128, 7),
    ])
def test_bucket_ctx_to_kernel_blocks(num_blocks, block_size, kernel_block_size, expected):
    assert bucket_ctx_to_kernel_blocks(num_blocks, block_size, kernel_block_size) == expected


@pytest.mark.parametrize("block_size, kernel_block_size", [(128, 528), (128, 768), (528, 128), (2176, 128)])
@pytest.mark.parametrize("num_blocks", [1, 2, 10, 1986])
def test_bucket_ctx_to_kernel_blocks_covers_the_context(num_blocks, block_size, kernel_block_size):
    kernel_blocks = bucket_ctx_to_kernel_blocks(num_blocks, block_size, kernel_block_size)
    # Every token of the bucket's context fits, with less than one kernel block to spare.
    assert kernel_blocks * kernel_block_size >= num_blocks * block_size
    assert (kernel_blocks - 1) * kernel_block_size < num_blocks * block_size
