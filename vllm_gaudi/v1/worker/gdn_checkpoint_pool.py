# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded block_id -> checkpoint-slot map for compact-GDN prefix caching.

Compact GDN keeps only max_num_seqs live state slots, so it cannot address a
checkpoint by block_id (that would need num_blocks slots). This map lets a
small pool of K slots stand in: block_id is the key, the slot is a row in the
bounded checkpoint tensor. Slot 0 is the null/pad slot.
"""
from collections import OrderedDict


def gdn_ckpt_num_slots(num_blocks: int, num_gdn_groups: int, mem_fraction: float, gdn_max_reqs: int,
                       explicit_slots: int) -> int:
    """Per-group checkpoint-pool depth K.

    Env override wins; else a fraction of num_blocks split across groups,
    floored at in-flight liveness and capped below num_blocks (the compact win).
    """
    if explicit_slots:
        k = explicit_slots
    else:
        k = int(mem_fraction * (num_blocks + 1) / num_gdn_groups) - 1
        k = max(k, gdn_max_reqs)
    return min(k, num_blocks - 1)


def gdn_ckpt_shadow_num_slots(num_blocks: int, num_gdn_groups: int, mem_fraction: float, explicit_slots: int) -> int:
    """Engine-core shadow-pool depth: a lower bound on the worker's K.

    The worker floors K at its in-flight liveness (max_num_seqs), which
    engine-core cannot see. Dropping that floor to 1 keeps the shadow's K <=
    the worker's, so its resident set stays a subset of the worker's and the
    hit cap can only be conservative, never stale.
    """
    return gdn_ckpt_num_slots(num_blocks, num_gdn_groups, mem_fraction, 1, explicit_slots)


# Residency view of the checkpoint pools, keyed by kv_cache_group_id.
# find_longest_cache_hit reads it to cap a prefix hit at the last boundary the
# bounded pool still holds. TP=1: the runner registers the real worker pool
# (register_ckpt_map). TP>1: the real pools live in worker processes, so
# engine-core registers a shadow residency mirror (register_shadow_map) driven
# by the same blocks the scheduler caches. A group is real or shadow, never both.
_CKPT_MAPS_BY_KV_GROUP: "dict[int, GdnCheckpointMap]" = {}
# Group ids whose registered map is an engine-core shadow (TP>1), not a real
# worker pool. Kept separate so hit/load-touch bookkeeping only runs on shadows.
_SHADOW_GROUP_IDS: set[int] = set()


def shadow_mirror_block_range(num_cached_before: int, num_cached_after: int, num_prompt_tokens: int,
                              block_size: int) -> range:
    """Block indices the tp>1 shadow may mirror on a cache_blocks step.

    The worker checkpoints only full prompt blocks (during prefill); it never
    checkpoints boundaries that fill during decode. Clamping the shadow to the
    same prompt-block prefix keeps its residency a subset of the worker pool, so
    the hit cap can never grant a prefix hit the worker cannot back (a stale
    over-claim would resume a later request from garbage recurrent state).
    """
    num_prompt_blocks = num_prompt_tokens // block_size
    return range(num_cached_before, min(num_cached_after, num_prompt_blocks))


def register_ckpt_map(kv_cache_group_id: int, cmap: "GdnCheckpointMap") -> None:
    """Register the real worker pool for a group (TP=1); supersedes any shadow."""
    _CKPT_MAPS_BY_KV_GROUP[kv_cache_group_id] = cmap
    _SHADOW_GROUP_IDS.discard(kv_cache_group_id)


def register_shadow_map(kv_cache_group_id: int, cmap: "GdnCheckpointMap") -> None:
    """Register an engine-core residency mirror for a group (TP>1)."""
    _CKPT_MAPS_BY_KV_GROUP[kv_cache_group_id] = cmap
    _SHADOW_GROUP_IDS.add(kv_cache_group_id)


def is_shadow(kv_cache_group_id: int) -> bool:
    """Whether the registered map for a group is an engine-core shadow."""
    return kv_cache_group_id in _SHADOW_GROUP_IDS


def ckpt_map_for_kv_group(kv_cache_group_id: int) -> "GdnCheckpointMap | None":
    return _CKPT_MAPS_BY_KV_GROUP.get(kv_cache_group_id)


class GdnCheckpointMap:

    def __init__(self, num_slots: int):
        assert num_slots >= 1
        self._num_slots = num_slots
        self._block_to_slot: dict[int, int] = {}
        self._slot_to_block: dict[int, int] = {}
        # recency order over occupied slots; front = least recently used
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._free: list[int] = list(range(1, num_slots + 1))

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def is_resident(self, block_id: int) -> bool:
        """Whether ``block_id``'s checkpoint currently occupies a slot."""
        return block_id in self._block_to_slot

    def drop(self, block_id: int) -> None:
        """Invalidate any checkpoint keyed at ``block_id``, freeing its slot.

        The map is keyed by block_id, but the scheduler recycles a freed block
        into a new prefix with a new hash. If the worker will not re-checkpoint
        that recycled block (it is a decode-region boundary, never a full prompt
        block), a stale block_id->slot entry would otherwise make is_resident
        report the block's *previous* prefix as restorable -- a wrong resume.
        Dropping the key on the step the block is re-cached closes that window.
        No-op when the block holds no slot.
        """
        slot = self._block_to_slot.pop(block_id, None)
        if slot is None:
            return
        del self._slot_to_block[slot]
        self._lru.pop(slot, None)
        self._free.append(slot)

    def resident_ids(self) -> "set[int]":
        """Block ids currently holding a slot.

        The correctness invariant for tp>1 is that the engine-core shadow's
        resident set stays a subset of the real worker pool's, so the hit cap
        never grants a prefix hit the worker cannot back. Tests assert this.
        """
        return set(self._block_to_slot)

    def get_load_slot(self, block_id: int) -> int:
        """Look up ``block_id``'s slot (0 = miss). Pure read: does NOT reorder LRU.

        Recency is driven *solely* by the store stream (alloc_store_slot). The
        worker and the tp>1 shadow see the same stores (cache_blocks mirrors
        them), so store-only recency keeps their eviction order identical and
        the shadow's resident set a subset of the worker's at any K. A load
        touch would only ever fire on one side (the worker on a continuation, or
        the shadow on a still-unscheduled hit probe), diverging the two orders
        and breaking that invariant. A load target is instead protected from
        same-step store eviction by the ``reserved`` set in alloc_store_slot.
        """
        return self._block_to_slot.get(block_id, 0)

    def alloc_store_slot(self, block_id: int, reserved: "set[int] | None" = None) -> int:
        """Assign a store slot to ``block_id``, evicting the LRU if the pool is full.

        ``reserved`` is a set of slot indices another batch in the same worker
        step will load from; eviction skips them so a store never clobbers a
        checkpoint that is about to be read this step. If every slot is reserved
        (the pool cannot grow without evicting a load target), the store is
        skipped and slot 0 (null) is returned -- safe, since a missing
        checkpoint just falls back to recompute, never a wrong result.
        """
        slot = self._block_to_slot.get(block_id)
        if slot is not None:
            self._lru.move_to_end(slot)
            return slot
        if self._free:
            slot = self._free.pop()
        else:
            slot = self._evict_lru(reserved)
            if slot is None:
                return 0
            old_block = self._slot_to_block.pop(slot)
            del self._block_to_slot[old_block]
        self._block_to_slot[block_id] = slot
        self._slot_to_block[slot] = block_id
        self._lru[slot] = None
        return slot

    def _evict_lru(self, reserved: "set[int] | None") -> "int | None":
        """Pop the least-recently-used slot not in ``reserved``; None if none."""
        if not reserved:
            slot, _ = self._lru.popitem(last=False)
            return slot
        victim = next((slot for slot in self._lru if slot not in reserved), None)
        if victim is not None:
            del self._lru[victim]
        return victim
