# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the compact-GDN prefix-cache checkpoint pool.

The correctness basis for compact-GDN prefix caching at TP>1 is the residency
subset invariant: the engine-core shadow pool's resident block set must stay a
subset of the real worker pool's. If it ever over-claims, the hit cap grants a
prefix hit the worker cannot back and a later request resumes from an evicted
(garbage) recurrent state. These tests assert that invariant directly -- a
weaker "shadow holds <= K blocks" check can pass while the real invariant is
already broken.
"""
from vllm_gaudi.v1.worker.gdn_checkpoint_pool import (
    GdnCheckpointMap,
    gdn_ckpt_num_slots,
    gdn_ckpt_shadow_num_slots,
    shadow_mirror_block_range,
)

# --- shadow_mirror_block_range: which blocks the tp>1 shadow may mirror -------


def test_mirror_range_spans_hit_region_from_fresh_resume():
    # before=0 (nothing cached yet), a full-prompt resume of 4 blocks: the
    # shadow mirrors the whole newly-cached prompt-block span.
    block_size = 16
    rng = shadow_mirror_block_range(num_cached_before=0,
                                    num_cached_after=4,
                                    num_prompt_tokens=4 * block_size,
                                    block_size=block_size)
    assert list(rng) == [0, 1, 2, 3]


def test_mirror_range_clamps_to_prompt_blocks():
    # The worker only checkpoints full *prompt* blocks; boundaries that fill
    # during decode are never checkpointed. A cache step that reports more
    # cached blocks than the prompt has must clamp, or the shadow would claim a
    # decode-region block the worker never wrote.
    block_size = 16
    rng = shadow_mirror_block_range(num_cached_before=2,
                                    num_cached_after=10,
                                    num_prompt_tokens=5 * block_size,
                                    block_size=block_size)
    assert list(rng) == [2, 3, 4]  # clamped at num_prompt_blocks == 5


def test_mirror_range_empty_when_after_not_greater_than_before():
    # A cache step that did not advance the cached-block count (after <= before)
    # mirrors nothing.
    block_size = 16
    assert list(
        shadow_mirror_block_range(num_cached_before=5,
                                  num_cached_after=3,
                                  num_prompt_tokens=8 * block_size,
                                  block_size=block_size)) == []
    assert list(
        shadow_mirror_block_range(num_cached_before=5,
                                  num_cached_after=5,
                                  num_prompt_tokens=8 * block_size,
                                  block_size=block_size)) == []


# --- GdnCheckpointMap: slot bookkeeping ---------------------------------------


def test_get_load_slot_miss_returns_null_slot():
    m = GdnCheckpointMap(num_slots=4)
    # Nothing stored: every block is a miss -> null slot 0, no residency.
    assert m.get_load_slot(7) == 0
    assert not m.is_resident(7)
    assert m.resident_ids() == set()


def test_resident_ids_tracks_stored_blocks():
    m = GdnCheckpointMap(num_slots=4)
    for bid in (10, 11, 12):
        m.alloc_store_slot(bid)
    assert m.resident_ids() == {10, 11, 12}
    assert all(m.is_resident(b) for b in (10, 11, 12))


def test_eviction_is_lru_and_load_touch_reorders():
    # K=4: fill it, then a 5th alloc evicts the least-recently-used block.
    m = GdnCheckpointMap(num_slots=4)
    for bid in (10, 11, 12, 13):
        m.alloc_store_slot(bid)
    assert m.resident_ids() == {10, 11, 12, 13}

    m.alloc_store_slot(14)  # evicts 10 (LRU)
    assert m.resident_ids() == {11, 12, 13, 14}
    assert m.get_load_slot(10) == 0  # evicted -> miss

    # A load hit on 11 moves it to MRU, so the next eviction victim is 12.
    assert m.get_load_slot(11) != 0
    m.alloc_store_slot(15)  # evicts 12, not 11
    assert m.resident_ids() == {11, 13, 14, 15}


def test_realloc_of_resident_block_is_idempotent():
    m = GdnCheckpointMap(num_slots=4)
    slot = m.alloc_store_slot(10)
    assert m.alloc_store_slot(10) == slot  # same slot, no new occupancy
    assert m.resident_ids() == {10}


# --- drop: invalidate a recycled/rehashed block key ---------------------------


def test_drop_invalidates_resident_block_and_frees_slot():
    # The map is keyed by block_id. When the scheduler recycles a freed block
    # into a new prefix (new hash) that the worker will not re-checkpoint, the
    # stale key must be dropped or is_resident would report the block's previous
    # prefix as restorable -- a wrong resume.
    m = GdnCheckpointMap(num_slots=4)
    m.alloc_store_slot(10)
    assert m.is_resident(10)
    m.drop(10)
    assert not m.is_resident(10)
    assert m.get_load_slot(10) == 0
    assert m.resident_ids() == set()


def test_drop_of_absent_block_is_noop():
    m = GdnCheckpointMap(num_slots=4)
    m.alloc_store_slot(10)
    m.drop(999)  # not resident
    assert m.resident_ids() == {10}


def test_drop_reclaims_capacity_without_eviction():
    # After a full pool drops a block, the freed slot is reused by the next
    # store instead of evicting a still-valid block.
    m = GdnCheckpointMap(num_slots=2)
    m.alloc_store_slot(10)
    m.alloc_store_slot(11)
    m.drop(10)
    m.alloc_store_slot(12)  # reuses 10's freed slot, keeps 11
    assert m.resident_ids() == {11, 12}


# --- reserved slots: a store must not evict a load target of the same step ----


def test_store_skips_reserved_slot_when_evicting():
    m = GdnCheckpointMap(num_slots=3)
    s10 = m.alloc_store_slot(10)  # oldest
    m.alloc_store_slot(11)
    m.alloc_store_slot(12)
    # Slot s10 is the LRU victim, but another batch loads from it this step.
    # The store must evict the next-oldest (11's slot) instead and keep s10.
    m.alloc_store_slot(20, reserved={s10})
    assert m.is_resident(10)  # reserved load target preserved
    assert not m.is_resident(11)  # next-oldest evicted instead
    assert m.resident_ids() == {10, 12, 20}


def test_store_skipped_when_all_slots_reserved():
    m = GdnCheckpointMap(num_slots=2)
    s10 = m.alloc_store_slot(10)
    s11 = m.alloc_store_slot(11)
    # Every slot is a load target this step: no evictable slot, so the store is
    # skipped (null slot 0) rather than clobbering a checkpoint about to be read.
    assert m.alloc_store_slot(20, reserved={s10, s11}) == 0
    assert not m.is_resident(20)
    assert m.resident_ids() == {10, 11}


def test_reserved_none_matches_plain_lru_eviction():
    # reserved=None (default) must behave exactly like the original LRU evict.
    m = GdnCheckpointMap(num_slots=2)
    m.alloc_store_slot(10)
    m.alloc_store_slot(11)
    m.alloc_store_slot(12)  # evicts 10 (LRU)
    assert m.resident_ids() == {11, 12}


# --- The residency subset invariant -------------------------------------------


def _worker_shadow_ks():
    # Pick a config where the worker floors K at its in-flight liveness
    # (gdn_max_reqs) above the memory-fraction base, so the shadow (which floors
    # at 1) gets a strictly smaller K -- the interesting case for the invariant.
    num_blocks, groups, frac, gdn_max_reqs = 100, 1, 0.05, 16
    worker_k = gdn_ckpt_num_slots(num_blocks, groups, frac, gdn_max_reqs, explicit_slots=0)
    shadow_k = gdn_ckpt_shadow_num_slots(num_blocks, groups, frac, explicit_slots=0)
    assert shadow_k < worker_k, (shadow_k, worker_k)
    return worker_k, shadow_k


def test_shadow_resident_stays_subset_under_shared_alloc_stream():
    # The scheduler drives both pools with the same boundary allocations in the
    # same order (shadow mirrors exactly what the worker checkpoints). With
    # shadow K <= worker K, LRU keeps the most-recent K in each, so the shadow's
    # resident set is always a subset of the worker's. Assert it at every step.
    worker_k, shadow_k = _worker_shadow_ks()
    worker = GdnCheckpointMap(num_slots=worker_k)
    shadow = GdnCheckpointMap(num_slots=shadow_k)

    # A stream of boundary block ids far exceeding either K, with revisits.
    stream = [b for rep in range(3) for b in range(1, 40)]
    for bid in stream:
        worker.alloc_store_slot(bid)
        shadow.alloc_store_slot(bid)
        assert shadow.resident_ids() <= worker.resident_ids()
    # Once past capacity the shadow holds strictly fewer blocks.
    assert len(shadow.resident_ids()) == shadow_k
    assert len(worker.resident_ids()) == worker_k


def test_shadow_subset_via_mirror_range_matches_worker_checkpoints():
    # Integration-style cross-check of the tp>1 path: the worker checkpoints
    # every full prompt block during prefill; the shadow mirrors only the
    # prompt-block slice shadow_mirror_block_range yields on each cache step.
    # The shadow must never hold a block the worker did not checkpoint, and its
    # residency must stay a subset of the worker's pool keys (== block_to_slot).
    worker_k, shadow_k = _worker_shadow_ks()
    worker = GdnCheckpointMap(num_slots=worker_k)
    shadow = GdnCheckpointMap(num_slots=shadow_k)
    block_size = 16

    next_block_id = 1
    worker_checkpointed: set[int] = set()
    # Several requests, each a multi-block prompt plus decode-region growth.
    for prompt_blocks, extra_decode_blocks in [(3, 2), (5, 1), (2, 4), (6, 0), (4, 3)]:
        prompt_tokens = prompt_blocks * block_size
        # Assign block ids for this request's cached blocks (prompt + decode).
        total_blocks = prompt_blocks + extra_decode_blocks
        block_ids = list(range(next_block_id, next_block_id + total_blocks))
        next_block_id += total_blocks

        # Worker: checkpoints full prompt blocks only (during prefill).
        for bid in block_ids[:prompt_blocks]:
            worker.alloc_store_slot(bid)
            worker_checkpointed.add(bid)

        # Shadow: cache_blocks fires once per newly cached block; mirror the
        # prompt-clamped slice. Model it as a single step before=0 -> after=all.
        for idx in shadow_mirror_block_range(0, total_blocks, prompt_tokens, block_size):
            shadow.alloc_store_slot(block_ids[idx])

        # Shadow never holds a decode-region block the worker skipped.
        assert shadow.resident_ids() <= worker_checkpointed
        # Shadow residency is a subset of the worker pool's live keys.
        assert shadow.resident_ids() <= worker.resident_ids()
