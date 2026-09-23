"""Bounded block_id -> checkpoint-slot map for compact-GDN prefix caching.

Compact GDN keeps only max_num_seqs live state slots, so it cannot address a
checkpoint by block_id (that would need num_blocks slots). This map lets a
small pool of K slots stand in: block_id is the key, the slot is a row in the
bounded checkpoint tensor. Slot 0 is the null/pad slot.
"""
from collections import OrderedDict


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

    def get_load_slot(self, block_id: int) -> int:
        slot = self._block_to_slot.get(block_id, 0)
        if slot:
            self._lru.move_to_end(slot)
        return slot

    def alloc_store_slot(self, block_id: int) -> int:
        slot = self._block_to_slot.get(block_id)
        if slot is not None:
            self._lru.move_to_end(slot)
            return slot
        if self._free:
            slot = self._free.pop()
        else:
            slot, _ = self._lru.popitem(last=False)
            old_block = self._slot_to_block.pop(slot)
            del self._block_to_slot[old_block]
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
