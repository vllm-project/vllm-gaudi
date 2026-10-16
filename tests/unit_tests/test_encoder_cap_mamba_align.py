# SPDX-License-Identifier: Apache-2.0
"""Chunk-grid precondition and encoder-cap guard tests.

Since vllm-project/vllm#51603 the mamba-aligned split runs *before*
``_try_schedule_encoder_inputs``, so the encoder cap can stop a chunk just
before an mm item it cannot admit and leave the request resuming off the mamba
chunk grid. Nothing in the scheduler re-aligns it, and nothing can: the cap is
one of two upstream-owned sites that shorten ``num_new_tokens`` after the split
(``_reserve_prefill_lookahead`` is the other).

The requirement is therefore stated in the runner, where it belongs, and only
for the layers that actually have it:

* ``mamba_chunk_grid_violation``: the real precondition. Only
  ``mamba_chunks_to_block_mapping`` is addressed by
  ``context_lens // mamba_chunk_size``, so an unaligned resume is safe unless a
  mamba state-block boundary falls strictly inside the chunk.
* ``mamba_chunk_grid_consumed``: that mapping is read only by MambaMixer2. The
  GDN layer Qwen3-Next/3.5/3.6 run writes one state to ``store_indices_tensor``
  and never looks at the chunk grid, so the precondition is inert there.
* ``_try_schedule_encoder_inputs``: the decision table of the shrink-to-grid
  guard, which is consequently dormant for GDN and live for MambaMixer2 models.

These run on CPU; no Gaudi hardware required.
"""
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_gaudi.v1.core.sched.hpu_async_scheduler import HPUAsyncScheduler

CHUNK = 128
PROMPT_LEN = 1024
BUDGET = 16384

# The class super() resolves to, taken from the MRO rather than patched by its
# dotted path: importing vllm_gaudi.utils rebinds
# vllm.v1.core.sched.async_scheduler.AsyncScheduler to HPUAsyncScheduler, so
# patching through that name would replace the override under test.
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
# mamba_chunk_grid_violation: the precondition the runner actually has.
# ---------------------------------------------------------------------------

MBLOCK = 1152  # cache_config.mamba_block_size on a Qwen3.5 hybrid


@pytest.mark.parametrize(
    "context_len, query_len, expected, why",
    [
        # Aligned resumes are unconditionally fine.
        (0, 1136, None, "fresh prefill"),
        (1024, 2048, None, "aligned resume spanning two blocks"),
        # The production failure: resumes at the chat-template prefix. The only
        # boundary is at context_len + query_len, which the clamp to the last
        # physical chunk serves exactly, so this step was always safe.
        (16, 1136, None, "encoder cap at a 16-token prefix, boundary is the chunk's own end"),
        (1136, 16, None, "resume just below a boundary, chunk ends on it"),
        (100, 1052, None, "resume past a 100-token leading image"),
        # One token further and the boundary is strictly interior: the chunk
        # that mapping picks ends past it, so its state is never checkpointed.
        (16, 1137, MBLOCK, "boundary one token inside the chunk"),
        (16, 2288, MBLOCK, "chunk spans a whole block"),
        (2016, 1136, 2 * MBLOCK, "resume mid-block, chunk crosses the next boundary"),
    ],
)
def test_precondition_fires_only_on_an_interior_boundary(context_len, query_len, expected, why):
    from vllm_gaudi.v1.worker.hpu_model_runner import mamba_chunk_grid_violation

    assert mamba_chunk_grid_violation(context_len, query_len, MBLOCK, CHUNK) == expected, why


def test_precondition_matches_the_block_index_arithmetic():
    """Cross-check against the mapping the runner builds, not just the formula.

    For every (context_len, query_len) the precondition calls safe, each block
    in the caching range must be written at the chunk that ends exactly on its
    boundary -- or, for the trailing partial block, at the end of the chunk.
    """
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
# The consumer predicate, and a guard so it cannot silently rot.
# ---------------------------------------------------------------------------

_CHUNK_GRID_FIELDS = ("mamba_chunks_to_block_mapping", "seqlens_offsets_for_blocks", "blocks_caching_range")


def test_gdn_layer_reads_no_chunk_grid_field():
    """mamba_chunk_grid_consumed is False for GDN only because of this.

    If the GDN layer ever starts consuming the chunk->block mapping, the
    predicate in hpu_model_runner.py stops being safe and this must fail.
    Read the source rather than importing: importing patches
    QwenGatedDeltaNetAttention into three upstream modules.
    """
    import importlib.util

    origin = importlib.util.find_spec("vllm_gaudi.models.qwen3_5").origin
    with open(origin) as handle:
        source = handle.read()
    found = [field for field in _CHUNK_GRID_FIELDS if field in source]
    assert not found, (f"HPUGatedDeltaNetAttention now references {found}; "
                       "mamba_chunk_grid_consumed must account for it")


def test_mamba_mixer2_does_read_the_chunk_grid():
    """The other half: the predicate exists because this consumer is real."""
    import importlib.util

    origin = importlib.util.find_spec("vllm_gaudi.ops.hpu_mamba_mixer2").origin
    with open(origin) as handle:
        source = handle.read()
    assert "ssm_state[mamba_chunks_to_block_mapping]" in source


@pytest.mark.parametrize(
    "num_mamba_like, num_gdn, expected, why",
    [
        (0, 0, False, "dense model"),
        (24, 24, False, "GDN only: Qwen3-Next / Qwen3.5 / Qwen3.6"),
        (24, 0, True, "MambaMixer2 hybrid, and granitemoehybrid via num_gdn=0"),
        (24, 12, True, "mixed stack still has MambaMixer2 layers"),
    ],
)
def test_consumer_predicate(num_mamba_like, num_gdn, expected, why):
    assert (num_mamba_like > num_gdn) is expected, why
