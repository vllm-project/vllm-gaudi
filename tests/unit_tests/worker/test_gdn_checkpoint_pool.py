import pytest
from vllm_gaudi.v1.worker.gdn_checkpoint_pool import GdnCheckpointMap, gdn_ckpt_num_slots


def test_num_slots_explicit_override_wins():
    # Env override is used verbatim (still capped below num_blocks).
    assert gdn_ckpt_num_slots(num_blocks=1000, num_gdn_groups=3, mem_fraction=0.1, gdn_max_reqs=8,
                              explicit_slots=64) == 64


def test_num_slots_auto_formula():
    # k = mem_fraction*(num_blocks+1)/groups - 1, floored/capped elsewhere.
    k = gdn_ckpt_num_slots(num_blocks=2244, num_gdn_groups=3, mem_fraction=0.1, gdn_max_reqs=8, explicit_slots=0)
    assert k == int(0.1 * 2245 / 3) - 1  # == 73


def test_num_slots_floored_at_liveness():
    # Tiny fraction -> floor at in-flight liveness (gdn_max_reqs).
    assert gdn_ckpt_num_slots(num_blocks=1000, num_gdn_groups=3, mem_fraction=0.001, gdn_max_reqs=16,
                              explicit_slots=0) == 16


def test_num_slots_capped_below_num_blocks():
    # Cap keeps the compact win even with a huge fraction or override.
    assert gdn_ckpt_num_slots(num_blocks=10, num_gdn_groups=1, mem_fraction=5.0, gdn_max_reqs=8,
                              explicit_slots=0) == 9
    assert gdn_ckpt_num_slots(num_blocks=10, num_gdn_groups=1, mem_fraction=0.1, gdn_max_reqs=4,
                              explicit_slots=100) == 9


def test_miss_returns_null_slot():
    m = GdnCheckpointMap(num_slots=4)
    assert m.get_load_slot(block_id=42) == 0


def test_store_then_load_roundtrip():
    m = GdnCheckpointMap(num_slots=4)
    slot = m.alloc_store_slot(block_id=42)
    assert slot != 0
    assert m.get_load_slot(block_id=42) == slot


def test_store_same_block_is_stable():
    m = GdnCheckpointMap(num_slots=4)
    s1 = m.alloc_store_slot(block_id=7)
    s2 = m.alloc_store_slot(block_id=7)
    assert s1 == s2


def test_distinct_blocks_get_distinct_slots():
    m = GdnCheckpointMap(num_slots=4)
    slots = {m.alloc_store_slot(b) for b in (1, 2, 3, 4)}
    assert slots == {1, 2, 3, 4}
    assert 0 not in slots


def test_eviction_is_lru():
    m = GdnCheckpointMap(num_slots=2)
    sa = m.alloc_store_slot(block_id=1)   # slots: {1->sa}
    sb = m.alloc_store_slot(block_id=2)   # slots full: {1->sa, 2->sb}
    m.get_load_slot(block_id=1)           # touch 1 -> 2 is now LRU
    sc = m.alloc_store_slot(block_id=3)   # evicts block 2
    assert sc == sb                       # reused block 2's slot
    assert m.get_load_slot(block_id=2) == 0
    assert m.get_load_slot(block_id=1) == sa


def test_free_block_returns_slot_to_pool():
    m = GdnCheckpointMap(num_slots=1)
    s1 = m.alloc_store_slot(block_id=1)
    m.free_block(block_id=1)
    assert m.get_load_slot(block_id=1) == 0
    s2 = m.alloc_store_slot(block_id=2)   # slot reusable, no eviction needed
    assert s2 == s1


def test_free_unmapped_block_is_noop():
    m = GdnCheckpointMap(num_slots=1)
    m.free_block(block_id=999)  # must not raise


def test_k_slot_index_never_exceeds_capacity():
    # A K-slot map never exposes a slot index outside [1, K].
    m = GdnCheckpointMap(num_slots=8)
    seen = {m.alloc_store_slot(b) for b in range(100)}
    assert max(seen) <= 8
    assert 0 not in seen


def test_translate_load_store_slots():
    """load slot must be a lookup (0 on miss); store slot must allocate."""
    m = GdnCheckpointMap(num_slots=4)
    # first request: prefix miss (load block 5 unseen), store block 6
    assert m.get_load_slot(5) == 0
    s_store = m.alloc_store_slot(6)
    assert s_store != 0
    # second request reusing prefix at block 6: load must now hit
    assert m.get_load_slot(6) == s_store
