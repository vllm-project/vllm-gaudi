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
    the worker's, so the shadow's resident set stays a subset of the worker's
    (LRU stack property): the hit cap can only be conservative, never stale.
    Equal to the worker's K in the common case (explicit slots, or the memory
    term dominating the floor).
    """
    return gdn_ckpt_num_slots(num_blocks, num_gdn_groups, mem_fraction, 1, explicit_slots)


# Residency view of the checkpoint pools, keyed by kv_cache_group_id
# (== worker group_idx). find_longest_cache_hit reads it to cap a prefix hit
# at the last boundary the bounded pool still holds.
#
# At TP=1 (UniProc) the scheduler shares the worker process, so the runner
# registers the *real* worker pool here (register_ckpt_map) and the cap reads
# the live pool directly -- never over-reporting a hit the pool has evicted.
#
# At TP>1 (MultiprocExecutor) the real pools live in the worker processes and
# this dict would otherwise stay empty in engine-core, making the cap a no-op
# so the scheduler could report a hit at a boundary the workers have evicted
# (stale-state resume). To close that gap, engine-core registers a *shadow*
# pool (register_shadow_map): a residency mirror driven by the same
# full-non-null block set the scheduler caches, so its is_resident matches the
# worker pools by construction. A group is either real (TP=1) or shadow (TP>1),
# never both.
_CKPT_MAPS_BY_KV_GROUP: "dict[int, GdnCheckpointMap]" = {}
# Group ids whose registered map is an engine-core shadow (TP>1), not a real
# worker pool. Kept separate so hit/load-touch bookkeeping only runs on shadows.
_SHADOW_GROUP_IDS: set[int] = set()


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

    def _dbg(self, *parts) -> None:
        import os
        if os.environ.get("GDN_EVICT_DEBUG"):
            import sys
            print("CKPTDBG", *parts, file=sys.stderr, flush=True)

    def is_resident(self, block_id: int) -> bool:
        """Whether ``block_id``'s checkpoint currently occupies a slot."""
        return block_id in self._block_to_slot

    def get_load_slot(self, block_id: int) -> int:
        slot = self._block_to_slot.get(block_id, 0)
        if slot:
            self._lru.move_to_end(slot)
        self._dbg("LOAD", f"bid={block_id}", f"slot={slot}", f"held_bid={self._slot_to_block.get(slot)}")
        return slot

    def alloc_store_slot(self, block_id: int) -> int:
        slot = self._block_to_slot.get(block_id)
        if slot is not None:
            self._lru.move_to_end(slot)
            self._dbg("STORE-REUSE", f"bid={block_id}", f"slot={slot}")
            return slot
        if self._free:
            slot = self._free.pop()
            self._dbg("STORE-NEW", f"bid={block_id}", f"slot={slot}")
        else:
            slot, _ = self._lru.popitem(last=False)
            old_block = self._slot_to_block.pop(slot)
            del self._block_to_slot[old_block]
            self._dbg("STORE-EVICT", f"bid={block_id}", f"slot={slot}", f"dropped={old_block}")
        self._block_to_slot[block_id] = slot
        self._slot_to_block[slot] = block_id
        self._lru[slot] = None
        return slot

    def free_block(self, block_id: int) -> None:
        slot = self._block_to_slot.pop(block_id, None)
        if slot is None:
            return
        del self._slot_to_block[slot]
        self._lru.pop(slot, None)
        self._free.append(slot)
