# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the HPU DSA sparse-attention indexer forward (vllm_gaudi/ops/hpu_sparse_attn_indexer.py).

``forward_hpu`` is exercised on CPU tensors with a fake forward context: the fast path (``indexer_block_index``
present, per-request ranking) and the fallback path (broadcast mask) are both compared against a naive pure-fp32
reference written straight from the scoring contract (per owned block, per valid position, sum over heads of
relu(q.k) * weight with fp32 accumulation). The module is imported normally (it only needs
``vllm.forward_context``); the context is injected with ``monkeypatch.setattr``.
"""

from types import SimpleNamespace

import torch

from vllm_gaudi.ops import hpu_sparse_attn_indexer as indexer

def _build(batch_size, per_request, block_size, n_head, head_dim, topk, seed,
           interleaved=True, width_extra=0, neg_weights=False, block_ids=None,
           partial_frac=0.0, pad_blocks=0):
    """Build decode metadata + tensors for the indexer. Returns (state, metadata, q, k, weights, cache, ref)."""
    g = torch.Generator().manual_seed(seed)
    num_blocks = batch_size * per_request
    if interleaved:
        groups = torch.arange(batch_size).repeat_interleave(per_request)
        groups = groups[torch.randperm(num_blocks, generator=g)]
    else:
        groups = torch.arange(batch_size).repeat_interleave(per_request)

    max_block = max(int(x) for x in (block_ids or [num_blocks])) + 1
    blocks = torch.tensor(block_ids, dtype=torch.long) if block_ids is not None else \
        torch.randperm(max_block * 8, generator=g)[:num_blocks]
    if pad_blocks:
        blocks = torch.cat([blocks, torch.full((pad_blocks,), max_block + 999, dtype=torch.long)])
        groups = torch.cat([groups, torch.full((pad_blocks,), -1, dtype=torch.long)])
        num_blocks_total = blocks.shape[0]
    else:
        num_blocks_total = num_blocks

    usage = torch.full((num_blocks_total,), float(block_size))
    if partial_frac > 0:
        n_partial = max(1, int(num_blocks_total * partial_frac))
        usage[torch.randperm(num_blocks_total, generator=g)[:n_partial]] = \
            torch.randint(1, block_size, (n_partial,), generator=g, dtype=torch.float32)

    num_slots = (max_block + 1000) * block_size
    cache = torch.randn(num_slots, head_dim, generator=g, dtype=torch.bfloat16) * 0.1

    q = torch.randn(batch_size, n_head, head_dim, generator=g, dtype=torch.bfloat16) * 0.2
    k = torch.randn(batch_size, head_dim, generator=g, dtype=torch.bfloat16) * 0.2
    weights = torch.randn(batch_size, n_head, generator=g, dtype=torch.bfloat16) * 0.5
    if neg_weights:
        weights = torch.cat([weights[: batch_size // 2], -weights[batch_size // 2:]], dim=0)

    width = per_request + width_extra
    index = torch.full((batch_size, width), num_blocks_total, dtype=torch.int32)
    for b in range(batch_size):
        pos = (groups == b).nonzero().flatten()
        index[b, : pos.numel()] = pos.int()

    state = SimpleNamespace(topk_tokens=topk,
                            topk_indices_buffer=torch.empty(batch_size, topk, dtype=torch.int32),
                            k_cache=SimpleNamespace(kv_cache=cache),
                            skip_k_cache_insert=True)
    metadata = SimpleNamespace(is_prompt=False,
                               block_size=block_size,
                               slot_mapping=torch.zeros(batch_size, dtype=torch.long),
                               block_list=blocks,
                               block_groups=groups,
                               block_usage=usage,
                               indexer_block_index=index)
    ref = dict(blocks=blocks, groups=groups, usage=usage, cache=cache, q=q, weights=weights,
               block_size=block_size, n_head=n_head, head_dim=head_dim, batch_size=batch_size,
               num_blocks_total=num_blocks_total)
    return state, metadata, q, k, weights, cache, ref


def _ref_scores_for(ref, b):
    """Naive fp32 scores {slot: score} for request ``b`` over owned, valid slots (contract 5, 8, 9)."""
    blocks, groups, usage, cache, q, weights = (ref['blocks'], ref['groups'], ref['usage'],
                                                ref['cache'], ref['q'], ref['weights'])
    bs = ref['block_size']
    out = {}
    for i in range(blocks.shape[0]):
        if int(groups[i]) != b:
            continue
        n_valid = int(min(bs, float(usage[i])))
        for p in range(n_valid):
            slot = int(blocks[i]) * bs + p
            key = cache[slot].float()
            logits = (q[b].float() * key).sum(-1)          # (n_head,) == bmm row
            out[slot] = float((logits.relu() * weights[b].float()).sum())
    return out


def _run(indexer_mod, monkeypatch, state, metadata, q, k, weights):
    context = SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(indexer_mod, 'get_forward_context', lambda: context)
    return indexer_mod.forward_hpu(state, None, q, k, weights)


def test_fast_and_fallback_select_same_slots(monkeypatch):
    """The per-request fast path and the broadcast-mask fallback must pick identical slot sets."""
    for seed in (1, 2, 3):
        state, meta, q, k, w, cache, ref = _build(4, 6, 16, 8, 32, 24, seed,
                                                interleaved=True, width_extra=2, partial_frac=0.5)
        fast = _run(indexer, monkeypatch, state, meta, q, k, w)
        meta_fb = SimpleNamespace(**{**meta.__dict__, 'indexer_block_index': None})
        state_fb = SimpleNamespace(topk_tokens=24,
                                   topk_indices_buffer=torch.empty(4, 24, dtype=torch.int32),
                                   k_cache=SimpleNamespace(kv_cache=cache),
                                   skip_k_cache_insert=True)
        fb = _run(indexer, monkeypatch, state_fb, meta_fb, q, k, w)
        assert torch.equal(fast.sort(dim=-1).values, fb.sort(dim=-1).values)
        # per-request non-(-1) count == min(topk, #valid owned slots)
        for b in range(4):
            n_valid = sum(int(min(16, float(ref['usage'][i]))) for i in range(24) if int(ref['groups'][i]) == b)
            expected_nonneg = min(24, n_valid)
            assert int((fast[b] >= 0).sum()) == expected_nonneg, f"seed {seed} req {b}"


def test_prefill_returns_minus_one(monkeypatch):
    state, meta, q, k, w, cache, _ = _build(2, 3, 8, 4, 16, 6, 11)
    meta = SimpleNamespace(**{**meta.__dict__, 'is_prompt': True})
    out = _run(indexer, monkeypatch, state, meta, q, k, w)
    assert (out[:2] == -1).all()


def test_empty_kv_cache_returns_minus_one(monkeypatch):
    state, meta, q, k, w, cache, _ = _build(2, 3, 8, 4, 16, 6, 12)
    state.k_cache = SimpleNamespace(kv_cache=torch.empty(0, 16, dtype=torch.bfloat16))
    out = _run(indexer, monkeypatch, state, meta, q, k, w)
    assert (out[:2] == -1).all()


def test_negative_weights_rank_by_score_not_magnitude(monkeypatch):
    """With negative weights the score can be negative; top-k still means the largest (least-negative) scores."""
    state, meta, q, k, w, cache, ref = _build(3, 4, 8, 4, 16, 5, 21, neg_weights=True, width_extra=1)
    fast = _run(indexer, monkeypatch, state, meta, q, k, w)
    meta_fb = SimpleNamespace(**{**meta.__dict__, 'indexer_block_index': None})
    state_fb = SimpleNamespace(topk_tokens=5,
                               topk_indices_buffer=torch.empty(3, 5, dtype=torch.int32),
                               k_cache=SimpleNamespace(kv_cache=cache),
                               skip_k_cache_insert=True)
    fb = _run(indexer, monkeypatch, state_fb, meta_fb, q, k, w)
    assert torch.equal(fast.sort(dim=-1).values, fb.sort(dim=-1).values)
    for b in range(3):
        sel = sorted(int(s) for s in fast[b].tolist() if s >= 0)
        sc = _ref_scores_for(ref, b)
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        expected = sorted(s for s, _ in ranked[: min(5, len(ranked))])
        assert sel == expected, f"req {b}: {sel} != {expected}"

def test_deterministic_topk_exact(monkeypatch):
    """With crafted strictly-ordered scores the selected slots are exactly the largest owned valid slots."""
    batch_size, per_request, block_size, n_head, head_dim, topk = 2, 2, 4, 4, 8, 4
    block_ids = [10, 20, 30, 40]          # block0->req0, block1->req1, block2->req0, block3->req1
    groups = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    blocks = torch.tensor(block_ids, dtype=torch.long)
    usage = torch.full((4,), float(block_size))
    num_blocks = 4
    width = per_request
    index = torch.full((batch_size, width), num_blocks, dtype=torch.int32)
    for b in range(batch_size):
        pos = (groups == b).nonzero().flatten()
        index[b, : pos.numel()] = pos.int()
    max_slot = max(block_ids) * block_size + block_size
    cache = torch.empty(max_slot, head_dim, dtype=torch.bfloat16)
    for s in range(max_slot):
        cache[s] = torch.full((head_dim,), float(s + 1), dtype=torch.bfloat16)   # score = 32 * (s + 1)
    q = torch.ones(batch_size, n_head, head_dim, dtype=torch.bfloat16)
    weights = torch.ones(batch_size, n_head, dtype=torch.bfloat16)
    k = torch.zeros(batch_size, head_dim, dtype=torch.bfloat16)
    state = SimpleNamespace(topk_tokens=topk,
                            topk_indices_buffer=torch.empty(batch_size, topk, dtype=torch.int32),
                            k_cache=SimpleNamespace(kv_cache=cache),
                            skip_k_cache_insert=True)
    meta = SimpleNamespace(is_prompt=False, block_size=block_size,
                           slot_mapping=torch.zeros(batch_size, dtype=torch.long),
                           block_list=blocks, block_groups=groups, block_usage=usage,
                           indexer_block_index=index)
    out = _run(indexer, monkeypatch, state, meta, q, k, weights)
    # req0 owns physical blocks 10,30 -> slots 40..43,120..123 ; topk=4 largest -> 120..123
    # req1 owns physical blocks 20,40 -> slots 80..83,160..163 ; topk=4 largest -> 160..163
    assert out[0].sort().values.tolist() == [120, 121, 122, 123]
    assert out[1].sort().values.tolist() == [160, 161, 162, 163]


def test_k_cache_insert_and_skip(monkeypatch):
    """skip_k_cache_insert=False writes k into the cache at slot_mapping; True leaves the cache untouched."""
    state, meta, q, k, w, cache, _ = _build(2, 2, 8, 4, 16, 3, 31)
    meta.slot_mapping = torch.tensor([5, 7], dtype=torch.long)
    k = torch.randn(2, 16, dtype=torch.bfloat16) * 0.3
    state_f = SimpleNamespace(topk_tokens=3,
                              topk_indices_buffer=torch.empty(2, 3, dtype=torch.int32),
                              k_cache=SimpleNamespace(kv_cache=cache),
                              skip_k_cache_insert=False)
    _run(indexer, monkeypatch, state_f, meta, q, k, w)
    assert torch.equal(cache[5], k[0]) and torch.equal(cache[7], k[1])
    cache2 = torch.randn(8000, 16, generator=torch.Generator().manual_seed(7), dtype=torch.bfloat16)
    before = cache2[[5, 7]].clone()
    state_t = SimpleNamespace(topk_tokens=3,
                              topk_indices_buffer=torch.empty(2, 3, dtype=torch.int32),
                              k_cache=SimpleNamespace(kv_cache=cache2),
                              skip_k_cache_insert=True)
    _run(indexer, monkeypatch, state_t, meta, q, k, w)
    assert torch.equal(cache2[[5, 7]], before)


def test_pad_blocks_group_minus_one_never_selected(monkeypatch):
    """Trailing blocks with group=-1 belong to no request and must never be selected."""
    state, meta, q, k, w, cache, ref = _build(2, 3, 8, 4, 16, 4, 41, pad_blocks=3)
    fast = _run(indexer, monkeypatch, state, meta, q, k, w)
    meta_fb = SimpleNamespace(**{**meta.__dict__, 'indexer_block_index': None})
    state_fb = SimpleNamespace(topk_tokens=4,
                               topk_indices_buffer=torch.empty(2, 4, dtype=torch.int32),
                               k_cache=SimpleNamespace(kv_cache=cache),
                               skip_k_cache_insert=True)
    fb = _run(indexer, monkeypatch, state_fb, meta_fb, q, k, w)
    assert torch.equal(fast.sort(dim=-1).values, fb.sort(dim=-1).values)
    pad_ids = set(int(x) for x in ref['blocks'][-3:].tolist())
    for b in range(2):
        for s in [int(x) for x in fast[b].tolist()]:
            if s >= 0:
                assert (s // 8) not in pad_ids, f"slot {s} lies in a group=-1 padded block"


def test_topk_larger_than_candidates_pads_minus_one(monkeypatch):
    """When a request has fewer valid owned slots than topk, the remainder is filled with -1."""
    batch_size, per_request, block_size, topk = 2, 1, 16, 10
    state, meta, q, k, w, cache, ref = _build(batch_size, per_request, block_size, 4, 16, topk, 51,
                                              width_extra=0, partial_frac=1.0)
    for b in range(batch_size):
        i = int((ref['groups'] == b).nonzero().flatten()[0])
        ref['usage'][i] = 3.0
    fast = _run(indexer, monkeypatch, state, meta, q, k, w)
    meta_fb = SimpleNamespace(**{**meta.__dict__, 'indexer_block_index': None})
    state_fb = SimpleNamespace(topk_tokens=topk,
                               topk_indices_buffer=torch.empty(batch_size, topk, dtype=torch.int32),
                               k_cache=SimpleNamespace(kv_cache=cache),
                               skip_k_cache_insert=True)
    fb = _run(indexer, monkeypatch, state_fb, meta_fb, q, k, w)
    assert torch.equal(fast.sort(dim=-1).values, fb.sort(dim=-1).values)
    for b in range(batch_size):
        assert int((fast[b] >= 0).sum()) == 3
        assert int((fast[b] == -1).sum()) == topk - 3


def test_shape_constant_cache(monkeypatch):
    """_BLK_CACHE: stable device-keyed entries, FIFO cap, reset hook, correct constant contents."""
    indexer._reset_block_cache()
    monkeypatch.setattr(indexer, "_BLK_CACHE_MAX", 3)
    try:
        dev0 = torch.device("cpu", 0)
        dev1 = torch.device("cpu", 1)
        t0 = indexer._blk_constants(2, 64, 128, dev0)
        t1 = indexer._blk_constants(4, 64, 128, dev1)
        # a fresh but equal device object (same type and index) must hit the same entry
        assert indexer._blk_constants(2, 64, 128, torch.device("cpu", 0)) is t0
        assert len(indexer._BLK_CACHE) == 2
        # at the cap, FIFO eviction drops the oldest entry before the new one is stored
        indexer._blk_constants(8, 64, 128, dev1)
        indexer._blk_constants(16, 64, 128, dev1)
        assert len(indexer._BLK_CACHE) <= 3
        assert (4, 64, 128, dev1) in indexer._BLK_CACHE
        assert (2, 64, 128, dev0) not in indexer._BLK_CACHE
        rows_int, pad_block_row, pad_row = t1
        assert rows_int.tolist() == [[i * 64] for i in range(4)]  # shape (B, 1)
        assert rows_int.dtype == torch.int32
        assert pad_block_row.tolist() == [0] and pad_block_row.dtype == torch.int32
        assert bool((pad_row == float("-inf")).all()) and pad_row.dtype == torch.float32
    finally:
        indexer._reset_block_cache()
