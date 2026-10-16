# SPDX-License-Identifier: Apache-2.0
"""Tests for the mamba chunk-grid precondition and the encoder-cap guard.

The encoder cap can leave a prefill resuming off the chunk grid. The runner
requires an aligned resume only where ``mamba_chunks_to_block_mapping`` is read
(MambaMixer2), and only if a state-block boundary falls inside the chunk.
CPU only.
"""
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler

CHUNK = 128
PROMPT_LEN = 1024
BUDGET = 16384

# Not the dotted path: vllm_gaudi.utils rebinds async_scheduler.AsyncScheduler
# to HPUAsyncScheduler, which would patch out the override under test.
_BASE = HPUAsyncScheduler.__bases__[0]


class _Position:

    def __init__(self, offset, length):
        self.offset = offset
        self.length = length

    def get_num_embeds(self):
        return self.length


def _request(items, prompt_len=PROMPT_LEN):
    """A stand-in Request exposing only what the override reads."""
    return SimpleNamespace(
        request_id="req-0",
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
        mm_features=[SimpleNamespace(mm_position=_Position(off, length)) for off, length in items],
    )


def _scheduler(*, grid_applies=True, chunk_size=CHUNK, disable_chunked_mm_input=False):
    """A bare scheduler with only the attributes the override reads.

    __init__ is bypassed on purpose: constructing a real Scheduler needs a
    model config, a KV cache config and a multimodal registry. Both properties
    below are cached_property (non-data descriptors), so an instance attribute
    shadows them.
    """
    scheduler = object.__new__(HPUAsyncScheduler)
    scheduler._mamba_chunk_grid_applies = grid_applies
    scheduler._mamba_align_chunk_size = chunk_size
    scheduler.scheduler_config = SimpleNamespace(disable_chunked_mm_input=disable_chunked_mm_input)
    return scheduler


def _call(scheduler, request, *, num_computed_tokens, num_new_tokens, base_result):
    with mock.patch.object(_BASE, "_try_schedule_encoder_inputs", return_value=base_result):
        return scheduler._try_schedule_encoder_inputs(request, num_computed_tokens, num_new_tokens, BUDGET)


# ---------------------------------------------------------------------------
# Pass-through: the override must not disturb a step the cap did not break.
# ---------------------------------------------------------------------------


def test_uncapped_step_is_untouched():
    scheduler = _scheduler()
    request = _request([(16, 64)])
    base = ([0], 512, BUDGET - 64, [])
    assert _call(scheduler, request, num_computed_tokens=0, num_new_tokens=512, base_result=base) == base


def test_already_deferring_step_is_untouched():
    """capped == 0 is the sibling branch of the same cap, already deferring."""
    scheduler = _scheduler()
    request = _request([(16, 64)])
    base = ([], 0, BUDGET, [])
    assert _call(scheduler, request, num_computed_tokens=200, num_new_tokens=512, base_result=base) == base


def test_capped_end_already_on_the_grid_is_untouched():
    scheduler = _scheduler()
    request = _request([(256, 64)])
    base = ([], 256, BUDGET, [])
    assert _call(scheduler, request, num_computed_tokens=0, num_new_tokens=512, base_result=base) == base


def test_non_mamba_model_is_untouched():
    """Without the runner's chunk-grid assert there is nothing to protect."""
    scheduler = _scheduler(grid_applies=False)
    request = _request([(80, 896)])
    base = ([], 80, BUDGET, [])
    assert _call(scheduler, request, num_computed_tokens=0, num_new_tokens=512, base_result=base) == base


def test_final_prefill_chunk_is_untouched():
    """A chunk reaching the end of the prompt has no resume point to align."""
    scheduler = _scheduler()
    request = _request([(16, 64)])
    base = ([], 24, BUDGET, [])
    assert _call(scheduler, request, num_computed_tokens=1000, num_new_tokens=30, base_result=base) == base


# ---------------------------------------------------------------------------
# Shrink: move the end back to the grid and unschedule what it no longer covers.
# ---------------------------------------------------------------------------


def test_cap_is_shrunk_to_the_previous_grid_boundary():
    scheduler = _scheduler()
    request = _request([(16, 64), (390, 10), (400, 200)])
    # super() admitted items 0 and 1, then clipped the chunk before item 2.
    base = ([0, 1], 400, BUDGET - 74, [])
    encoder_inputs, granted, budget, external = _call(scheduler,
                                                      request,
                                                      num_computed_tokens=0,
                                                      num_new_tokens=512,
                                                      base_result=base)

    assert granted == 384, "end should fall back to the previous 128 boundary"
    assert granted % CHUNK == 0
    # Item 1 starts at 390, past the new end, so it must not be encoded here
    # and its budget must come back.
    assert encoder_inputs == [0]
    assert budget == BUDGET - 74 + 10
    assert external == []


def test_shrink_keeps_items_the_shorter_chunk_still_covers():
    scheduler = _scheduler()
    request = _request([(16, 64), (300, 20), (600, 200)])
    base = ([0, 1], 600, BUDGET - 84, [])
    encoder_inputs, granted, budget, _ = _call(scheduler,
                                               request,
                                               num_computed_tokens=0,
                                               num_new_tokens=700,
                                               base_result=base)

    assert granted == 512
    assert encoder_inputs == [0, 1], "both items start below the new end"
    assert budget == BUDGET - 84, "nothing dropped, so nothing to refund"


def test_shrink_is_relative_to_num_computed_tokens():
    scheduler = _scheduler()
    request = _request([(600, 200)])
    base = ([], 344, BUDGET, [])
    _, granted, _, _ = _call(scheduler, request, num_computed_tokens=256, num_new_tokens=512, base_result=base)

    assert granted == 344 - 88, "granted is a length; 256 + 256 is the aligned end"
    assert (256 + granted) % CHUNK == 0


# ---------------------------------------------------------------------------
# Defer: no grid boundary left, so the step cannot be salvaged.
# ---------------------------------------------------------------------------


def test_defers_when_the_blocking_item_sits_inside_the_first_chunk():
    """The production failure: a chat-template prefix shorter than one chunk.

    The cap clips to the image's offset (16), which leaves no grid boundary
    above num_computed_tokens, so the step must be dropped rather than
    resumed off-grid.
    """
    scheduler = _scheduler()
    request = _request([(16, 16290)])
    base = ([], 16, BUDGET, [])
    assert _call(scheduler, request, num_computed_tokens=0, num_new_tokens=512, base_result=base) == ([], 0, BUDGET, [])


def test_defers_rather_than_splitting_an_unsplittable_mm_item():
    scheduler = _scheduler(disable_chunked_mm_input=True)
    request = _request([(16, 64), (300, 200)])
    base = ([0], 400, BUDGET - 64, [])
    assert _call(scheduler, request, num_computed_tokens=0, num_new_tokens=512, base_result=base) == ([], 0, BUDGET, [])


def test_unknown_chunk_size_is_untouched():
    scheduler = _scheduler(chunk_size=0)
    request = _request([(16, 16290)])
    base = ([], 16, BUDGET, [])
    assert _call(scheduler, request, num_computed_tokens=0, num_new_tokens=512, base_result=base) == base


# ---------------------------------------------------------------------------
# mamba_chunk_grid_violation
# ---------------------------------------------------------------------------

MBLOCK = 1152  # mamba_block_size on a Qwen3.5 hybrid


@pytest.mark.parametrize(
    "context_len, query_len, expected",
    [
        (0, 1136, None),
        (1024, 2048, None),
        (16, 1136, None),  # the encoder-cap failure: the boundary is the chunk's end
        (1136, 16, None),
        (100, 1052, None),
        (16, 1137, MBLOCK),  # boundary one token inside the chunk
        (16, 2288, MBLOCK),
        (2016, 1136, 2 * MBLOCK),
    ],
)
def test_precondition_fires_only_on_an_interior_boundary(context_len, query_len, expected):
    from vllm_gaudi.v1.worker.hpu_model_runner import mamba_chunk_grid_violation

    assert mamba_chunk_grid_violation(context_len, query_len, MBLOCK, CHUNK) == expected


def test_precondition_matches_the_block_index_arithmetic():
    """Wherever the precondition holds, the runner's mapping writes each block at
    its boundary, or the trailing partial block at the chunk's end."""
    import torch

    from vllm_gaudi.v1.worker.hpu_model_runner import (compute_prefix_caching_block_indices, mamba_chunk_grid_violation)

    stride = MBLOCK // CHUNK
    for context_len in range(0, 2 * MBLOCK, 16):
        for query_len in (16, 128, 1136, 1152, 2288):
            if mamba_chunk_grid_violation(context_len, query_len, MBLOCK, CHUNK) is not None:
                continue
            target_seq = -(-query_len // CHUNK) * CHUNK
            nphysical = target_seq // CHUNK
            _, first_sched, last_sched = compute_prefix_caching_block_indices(1, [context_len], [query_len], MBLOCK)
            chunk_offset = (first_sched[0].item() + 1) * stride - 1 - context_len // CHUNK
            n_blocks = last_sched[0].item() - first_sched[0].item() + 1
            indices = torch.clamp(torch.arange(n_blocks) * stride + chunk_offset, max=nphysical - 1)

            end = context_len + query_len
            for b_rel, chunk_idx in enumerate(indices.tolist()):
                boundary = (first_sched[0].item() + b_rel + 1) * MBLOCK
                written_at = context_len + min((chunk_idx + 1) * CHUNK, query_len)
                want = boundary if boundary <= end else end
                assert written_at == want, (f"ctx={context_len} qlen={query_len} block={first_sched[0].item() + b_rel} "
                                            f"written@{written_at} want@{want}")


# ---------------------------------------------------------------------------
# mamba_reads_chunk_grid
# ---------------------------------------------------------------------------


def _kv_cache_config(*mamba_types):
    import torch
    from vllm.v1.kv_cache_interface import MambaSpec

    attention = SimpleNamespace(kv_cache_spec=object())
    mamba = [
        SimpleNamespace(kv_cache_spec=MambaSpec(
            block_size=MBLOCK, shapes=((1, ), ), dtypes=(torch.float32, ), mamba_type=mamba_type))
        for mamba_type in mamba_types
    ]
    return SimpleNamespace(kv_cache_groups=[attention, *mamba])


def test_chunk_grid_readers_are_classified_by_mamba_type():
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Mamba

    from vllm_gaudi.platform import mamba_reads_chunk_grid

    assert not mamba_reads_chunk_grid(_kv_cache_config())
    assert not mamba_reads_chunk_grid(_kv_cache_config(Mamba.GDN_ATTN))
    assert not mamba_reads_chunk_grid(_kv_cache_config(Mamba.LINEAR))
    assert mamba_reads_chunk_grid(_kv_cache_config(Mamba.MAMBA2))
    assert mamba_reads_chunk_grid(_kv_cache_config(Mamba.GDN_ATTN, Mamba.MAMBA2))
    assert mamba_reads_chunk_grid(_kv_cache_config(Mamba.MAMBA1)), "unaudited types stay strict"


def test_layer_classes_report_the_types_the_classification_assumes():
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

    from vllm_gaudi.ops.hpu_mamba_mixer2 import HPUMambaMixer2
    from vllm_gaudi.platform import gdn_mamba_types

    assert GatedDeltaNetAttention.mamba_type.fget(None) in gdn_mamba_types()
    assert HPUMambaMixer2.mamba_type.fget(None) not in gdn_mamba_types()


def _source(module):
    import importlib.util

    with open(importlib.util.find_spec(module).origin) as handle:
        return handle.read()


def test_only_mamba_mixer2_reads_the_chunk_grid():
    """GDN is classified as a non-reader because it never touches these fields."""
    assert "ssm_state[mamba_chunks_to_block_mapping]" in _source("vllm_gaudi.ops.hpu_mamba_mixer2")
    # Source, not import: importing qwen3_5 patches upstream GDN classes.
    gdn = _source("vllm_gaudi.models.qwen3_5")
    for field in ("mamba_chunks_to_block_mapping", "seqlens_offsets_for_blocks", "blocks_caching_range"):
        assert field not in gdn, f"the GDN layer now reads {field}"
