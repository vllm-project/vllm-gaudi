# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``HPUMLAImpl.forward_mqa_sparse`` (vllm_gaudi/attention/backends/hpu_attn.py).

The OOT-MLA decode sparse path is exercised on CPU tensors with a minimal object bound to the
real method (the module itself pulls in the HPU stack at import time, so it is imported when
available and the tests skip otherwise). Outputs are compared against a pure-fp32 reference
written straight from the scoring contract (scale, -1 padding excluded, softmax over the
remaining slots); the bf16-scoring chain is non-bit-exact by design, so tolerances are bf16-sized.
All-padding rows must still yield exactly zero output (the -inf mask is zeroed back to zero
weights, so no NaN can reach the caller).
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

try:
    from vllm_gaudi.attention.backends.hpu_attn import HPUMLAImpl
except Exception as e:  # no HPU stack in this environment (imports raise before device setup)
    HPUMLAImpl = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

pytestmark = pytest.mark.skipif(HPUMLAImpl is None,
                                reason=f"hpu_attn import unavailable on this platform: {_IMPORT_ERROR}")

# GLM-5.3-Flash production shapes
KV_RANK = 512
ROPE = 64
HEADS = 32
TOPK = 64
L = KV_RANK + ROPE
SCALE = 1.0 / L**0.5


def _impl():
    """Bind the real forward_mqa_sparse to a namespace carrying the attributes it reads."""
    ns = SimpleNamespace(scale=SCALE, kv_lora_rank=KV_RANK, qk_rope_head_dim=ROPE, num_heads=HEADS)
    ns.forward_mqa_sparse = MethodType(HPUMLAImpl.forward_mqa_sparse, ns)
    return ns


def _ref_fp32(q, selected, pad_mask, scale):
    """Scoring contract in fp32: scale, -1 padding excluded, softmax, value gather over KV_RANK."""
    logits = torch.bmm(q.float(), selected.transpose(1, 2).float()) * scale
    logits.masked_fill_(pad_mask.unsqueeze(1), float("-inf"))
    attn = logits.softmax(dim=-1)
    attn.masked_fill_(pad_mask.unsqueeze(1), 0)
    return torch.bmm(attn, selected[..., :KV_RANK].float()).reshape(q.shape[0], HEADS * KV_RANK)


def _build(batch, topk, pad, seed, poison_slot0=False):
    """Random q / latent cache / per-request top-k selection with a tail of -1 padding slots."""
    g = torch.Generator().manual_seed(seed)
    slots = (batch * topk) + 64
    cache = torch.randn(slots, L, generator=g, dtype=torch.bfloat16) * 0.25
    if poison_slot0:
        cache[0] = torch.full_like(cache[0], 1e30)  # finite poison: 0-weight x 1e30 = 0 stays finite
    q = torch.randn(batch, HEADS, L, generator=g, dtype=torch.bfloat16) * 0.25
    topk_indices = torch.empty(batch, topk, dtype=torch.int32)
    pool = slots - 1 if poison_slot0 else slots
    offset = 1 if poison_slot0 else 0  # reserve slot 0 so only the -1 clamp can reach it
    for b in range(batch):
        perm = (torch.randperm(pool, generator=g, dtype=torch.int64) + offset)[:topk].to(torch.int32)
        if pad > 0:
            perm[-pad:] = -1
        topk_indices[b] = perm
    return q, cache, topk_indices


def _selected(cache, topk_indices, batch, topk):
    return cache[topk_indices.clamp(min=0).reshape(-1)].view(batch, topk, L)


def test_forward_matches_fp32_reference():
    """bf16-scoring chain must match the fp32 reference within bf16 tolerances (B=1, B>1, partial pad)."""
    impl = _impl()
    for batch, topk, pad in ((1, TOPK, 0), (1, TOPK, 12), (5, TOPK, 17), (3, 16, 3)):
        q, cache, topk_indices = _build(batch, topk, pad, seed=batch * 100 + pad)
        out = impl.forward_mqa_sparse(q, cache, None, topk_indices)
        selected = _selected(cache, topk_indices, batch, topk)
        ref = _ref_fp32(q, selected, topk_indices == -1, SCALE)
        assert out.dtype == torch.bfloat16
        assert out.shape == (batch, HEADS * KV_RANK)
        diff = (out.float() - ref).abs().max().item()
        assert diff < 8e-3 and torch.allclose(out.float(), ref, rtol=1.6e-2, atol=1e-2), \
            f"B={batch} topk={topk} pad={pad}: max abs diff {diff:.3e}"


def test_all_padding_row_is_exactly_zero():
    """A row whose top-k slots are all -1 must return exactly zero (no NaN, no slot leakage)."""
    impl = _impl()
    q, cache, topk_indices = _build(2, TOPK, TOPK, seed=7)
    assert (topk_indices == -1).all()
    out = impl.forward_mqa_sparse(q, cache, None, topk_indices)
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


def test_padding_slots_never_selected():
    """Slot -1 is clamped to slot 0 for the gather; a poisoned slot 0 must not leak into the output."""
    impl = _impl()
    q, cache, topk_indices = _build(4, TOPK, 8, seed=11, poison_slot0=True)
    assert (topk_indices == -1).any()
    out = impl.forward_mqa_sparse(q, cache, None, topk_indices)
    assert torch.isfinite(out).all()
    selected = _selected(cache, topk_indices, 4, TOPK)
    ref = _ref_fp32(q, selected, topk_indices == -1, SCALE)
    assert torch.allclose(out.float(), ref, rtol=1.6e-2, atol=1e-2)


def test_k_cache_tuple():
    """The physical latent cache is handed over as a (kv, ) tuple; only its first entry may be read."""
    impl = _impl()
    q, cache, topk_indices = _build(2, TOPK, 4, seed=23)
    poison = torch.ones_like(cache)
    via_tuple = impl.forward_mqa_sparse(q, (cache, poison), None, topk_indices)
    via_tensor = impl.forward_mqa_sparse(q, cache, None, topk_indices)
    assert torch.equal(via_tuple, via_tensor)
