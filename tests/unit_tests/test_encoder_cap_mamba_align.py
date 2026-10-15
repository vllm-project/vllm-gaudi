# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the HPUAsyncScheduler encoder-cap overrides.

Since vllm-project/vllm#51603 the mamba-aligned split runs *before*
``_try_schedule_encoder_inputs``, so the encoder cap can stop a chunk just
before an mm item it cannot admit and leave the request resuming off the
mamba chunk grid. The runner then trips
``assert context_lens[0] % mamba_chunk_size == 0`` and takes down every TP
worker.

Covers:

* ``_try_schedule_encoder_inputs``: the decision table of the guard that keeps
  the capped chunk on the grid -- pass through, shrink (with mm items dropped
  and their encoder budget refunded), and defer when no grid boundary is left.
These run on CPU; no Gaudi hardware required.
"""
from types import SimpleNamespace
from unittest import mock

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
