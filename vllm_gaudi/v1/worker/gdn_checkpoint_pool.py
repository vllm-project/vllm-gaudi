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


class GdnCheckpointMap:

    def __init__(self, num_slots: int):
        assert num_slots >= 1
        self._num_slots = num_slots
        self._block_to_slot: dict[int, int] = {}
        self._slot_to_block: dict[int, int] = {}
        # recency order over occupied slots; front = least recently used
        self._lru: "OrderedDict[int, None]" = OrderedDict()
        self._free: list[int] = list(range(1, num_slots + 1))

    @property
    def num_slots(self) -> int:
        return self._num_slots

    def _dbg(self, *parts) -> None:
        import os
        if os.environ.get("GDN_EVICT_DEBUG"):
            import sys
            print("CKPTDBG", *parts, file=sys.stderr, flush=True)

    def get_load_slot(self, block_id: int) -> int:
        slot = self._block_to_slot.get(block_id, 0)
        if slot:
            self._lru.move_to_end(slot)
        self._dbg("LOAD", f"bid={block_id}", f"slot={slot}",
                  f"held_bid={self._slot_to_block.get(slot)}")
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
