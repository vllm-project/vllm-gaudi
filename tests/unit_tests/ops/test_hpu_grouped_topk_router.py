# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: the HPU grouped top-k router selects the same experts as the plain torch.topk version.

The router module is loaded by file path with stubbed vllm imports.
"""

import importlib.util
import pathlib
import sys
import types

import pytest
import torch
import torch._dynamo as dynamo
from torch._dynamo.testing import CompileCounter

_ROUTER_PATH = (pathlib.Path(__file__).resolve().parents[3] / "vllm_gaudi" / "ops" / "hpu_grouped_topk_router.py")


@pytest.fixture
def router(monkeypatch):

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, mod)
        return mod

    class GroupedTopk:

        @classmethod
        def register_oot(cls, op):
            return op

        def __init__(self,
                     topk,
                     renormalize,
                     num_expert_group=0,
                     topk_group=0,
                     scoring_func="softmax",
                     routed_scaling_factor=1.0,
                     num_fused_shared_experts=0):
            self.topk = topk
            self.renormalize = renormalize
            self.num_expert_group = num_expert_group
            self.topk_group = topk_group
            self.scoring_func = scoring_func
            self.routed_scaling_factor = routed_scaling_factor

    module("vllm", envs=None)
    module("vllm.envs", VLLM_BATCH_INVARIANT=False, VLLM_USE_FUSED_MOE_GROUPED_TOPK=False)
    module("vllm.platforms",
           current_platform=types.SimpleNamespace(simple_compile_backend="eager", is_cuda=lambda: False))
    for name in ("vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe",
                 "vllm.model_executor.layers.fused_moe.router"):
        module(name)
    module("vllm.model_executor.utils", maybe_disable_graph_partition=lambda backend: {})
    module("vllm.model_executor.layers.fused_moe.router.grouped_topk_router",
           GroupedTopk=GroupedTopk,
           fused_grouped_topk=None)
    spec = importlib.util.spec_from_file_location("hpu_grouped_topk_router_under_test", _ROUTER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_grouped_topk(gating_output, topk, renormalize, num_expert_group, topk_group, scoring_func,
                           routed_scaling_factor, e_score_correction_bias, use_sorted):
    """The previous HPU grouped_topk: group mask for every group count, one torch.topk over all experts, gather."""
    gating_output = gating_output.float()
    scores = torch.softmax(gating_output, dim=-1) if scoring_func == "softmax" else gating_output.sigmoid()
    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.float().unsqueeze(0)
        scores_tmp = scores.clone().reshape(num_token, num_expert_group, -1)
        top1_val, top1_idx = torch.max(scores_tmp, dim=-1)
        scores_tmp.scatter_(-1, top1_idx.unsqueeze(-1), torch.finfo(scores.dtype).min)
        group_scores, _ = torch.max(scores_tmp, dim=-1)
        group_scores.add_(top1_val)
    else:
        group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    if num_token > 1024:
        group_mask = torch.zeros_like(group_scores)
        for i in range(topk_group):
            _, group_idx = torch.max(group_scores, dim=-1)
            group_mask.scatter_(1, group_idx.unsqueeze(-1), 1)
            if i < topk_group - 1:
                group_scores.scatter_(1, group_idx.unsqueeze(-1), torch.finfo(scores.dtype).min)
    else:
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
    tmp_scores = scores.reshape(num_token, num_expert_group, -1) + (
        (1 - group_mask) * torch.finfo(scores.dtype).min).unsqueeze(-1)
    tmp_scores = tmp_scores.reshape(num_token, -1)
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def _kimi_like_inputs(seed, tokens, experts):
    """bf16 GEMM logits upcast to fp32 and an fp32 correction bias, as in the Kimi-K3 gate."""
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(tokens, experts, generator=g) * 2).to(torch.bfloat16).float()
    bias = torch.randn(experts, generator=g) * 0.05
    return logits, bias


def _impl(router,
          logits,
          topk,
          renormalize,
          num_expert_group,
          topk_group,
          scoring_func,
          routed_scaling_factor,
          bias,
          out_dtype,
          fn="kimi_k3_grouped_topk"):
    hidden = torch.zeros(logits.shape[0], 1, dtype=out_dtype)
    return getattr(router, fn)(hidden, logits, topk, renormalize, num_expert_group, topk_group, scoring_func,
                               routed_scaling_factor, bias)


def _by_expert(weights, ids):
    order = ids.argsort(dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


def test_flat_take_is_gather(router):
    g = torch.Generator().manual_seed(1)
    src = torch.randn(37, 896, generator=g)
    index = torch.randint(0, 896, (37, 16), generator=g)
    assert torch.equal(router._flat_take(src, index), src.gather(1, index))


@pytest.mark.parametrize("experts", [512, 896, 1024])
@pytest.mark.parametrize("tokens", [1, 5, 300])
def test_two_stage_topk_selects_the_same_entries_as_torch_topk(router, experts, tokens):
    g = torch.Generator().manual_seed(experts + tokens)
    scores = torch.randn(tokens, experts, generator=g)
    values, ids = router._topk(scores, 16, True)
    ref_values, ref_ids = torch.topk(scores, 16, dim=-1, sorted=True)
    assert torch.equal(values, ref_values) and torch.equal(ids, ref_ids)
    values, ids = router._topk(scores, 16, False)
    assert torch.equal(ids.sort(dim=-1).values, ref_ids.sort(dim=-1).values)
    assert torch.equal(scores.gather(1, ids), values)


def test_two_stage_topk_with_ties_selects_the_same_values(router):
    """Many equal scores, also across the k-th place: the selected values are those of torch.topk."""
    g = torch.Generator().manual_seed(7)
    scores = torch.randint(0, 6, (64, 896), generator=g).float()
    values, ids = router._topk(scores, 16, True)
    assert torch.equal(values, torch.topk(scores, 16, dim=-1, sorted=True).values)
    assert torch.equal(scores.gather(1, ids), values)
    assert all(len(set(row.tolist())) == 16 for row in ids)


@pytest.mark.parametrize("experts", [256, 384, 500])
def test_narrow_rows_keep_one_topk(router, experts):
    scores = torch.randn(9, experts)
    values, ids = router._topk(scores, 8, False)
    ref_values, ref_ids = torch.topk(scores, 8, dim=-1, sorted=False)
    assert torch.equal(values, ref_values) and torch.equal(ids, ref_ids)


@pytest.mark.parametrize("tokens", [3, 257, 1500])
@pytest.mark.parametrize("use_sorted", [True, False])
def test_kimi_k3_routing_selects_the_same_experts(router, monkeypatch, tokens, use_sorted):
    """896 experts, one group, sigmoid + correction bias: same expert set, same weights per expert."""
    monkeypatch.setattr(router.envs, "VLLM_BATCH_INVARIANT", use_sorted)
    logits, bias = _kimi_like_inputs(tokens, tokens, 896)
    args = (logits, 16, True, 1, 1, "sigmoid", 2.5, bias)
    weights, ids = _impl(router, *args, torch.float32)
    ref_weights, ref_ids = reference_grouped_topk(*args, use_sorted)
    assert ids.dtype == torch.int64
    if use_sorted:
        assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)
    else:
        weights, ids = _by_expert(weights, ids)
        ref_weights, ref_ids = _by_expert(ref_weights, ref_ids)
        assert torch.equal(ids, ref_ids)
        # renormalisation sums the same 16 weights in another order
        torch.testing.assert_close(weights, ref_weights, rtol=1e-6, atol=0)


def test_kimi_k3_routing_without_two_stage_topk_is_bit_identical(router, monkeypatch):
    monkeypatch.setattr(router, "_TWO_STAGE_TOPK", False)
    logits, bias = _kimi_like_inputs(11, 200, 896)
    args = (logits, 16, True, 1, 1, "sigmoid", 2.5, bias)
    weights, ids = _impl(router, *args, torch.float32)
    ref_weights, ref_ids = reference_grouped_topk(*args, False)
    assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)


@pytest.mark.parametrize("tokens", [7, 1100])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("scoring_func", ["sigmoid", "softmax"])
def test_multi_group_routing_is_bit_identical(router, tokens, with_bias, scoring_func):
    """DeepSeek-V3 like: 256 experts in 8 groups, 4 groups kept, top-8."""
    logits, bias = _kimi_like_inputs(tokens + 3, tokens, 256)
    args = (logits, 8, True, 8, 4, scoring_func, 2.5, bias if with_bias else None)
    weights, ids = _impl(router, *args, torch.bfloat16)
    ref_weights, ref_ids = reference_grouped_topk(*args, False)
    assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights.to(torch.bfloat16))


def test_single_group_shortcut_is_bit_identical_without_bias(router):
    logits, _ = _kimi_like_inputs(5, 40, 384)
    args = (logits, 8, True, 1, 1, "softmax", 1.0, None)
    weights, ids = _impl(router, *args, torch.float32)
    ref_weights, ref_ids = reference_grouped_topk(*args, False)
    assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)


def test_hpu_grouped_topk_op_selects_the_same_experts(router):
    op = router.HPUGroupedTopk(topk=16,
                               renormalize=True,
                               num_expert_group=1,
                               topk_group=1,
                               scoring_func="sigmoid",
                               routed_scaling_factor=2.5)
    logits, bias = _kimi_like_inputs(3, 20, 896)
    hidden = torch.zeros(20, 8, dtype=torch.bfloat16)
    weights, ids = _by_expert(*op.forward_oot(hidden, logits, bias))
    ref_weights, ref_ids = _by_expert(*_impl(router, logits, 16, True, 1, 1, "sigmoid", 2.5, bias, torch.bfloat16))
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights, rtol=1e-2, atol=0)


def test_kimi_k3_routing_compiles_to_one_graph_for_all_token_counts(router):
    dynamo.reset()
    counter = CompileCounter()
    targets: set[str] = set()

    def backend(gm, example_inputs):
        targets.update(str(node.target) for node in gm.graph.nodes if node.op in ("call_function", "call_method"))
        return counter(gm, example_inputs)

    _, bias = _kimi_like_inputs(0, 1, 896)
    compiled = torch.compile(lambda logits: _impl(router, logits, 16, True, 1, 1, "sigmoid", 2.5, bias, torch.bfloat16),
                             backend=backend,
                             fullgraph=True,
                             dynamic=True)
    for tokens in (5, 33, 1500):
        logits, _ = _kimi_like_inputs(tokens, tokens, 896)
        weights, ids = compiled(logits)
        ref_weights, ref_ids = _impl(router, logits, 16, True, 1, 1, "sigmoid", 2.5, bias, torch.bfloat16)
        assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)
    assert counter.frame_count == 1
    assert not any("gather" in name for name in targets)
    dynamo.reset()


@pytest.mark.parametrize("tokens", [3, 1100])
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("shape", [(896, 16, 1, 1), (256, 8, 8, 4)])
def test_grouped_topk_without_kimi_flag_is_the_upstream_implementation(router, monkeypatch, tokens, with_bias, shape):
    """The shared grouped_topk keeps the plain group mask + one torch.topk for every group count."""
    experts, topk, groups, topk_group = shape
    monkeypatch.setattr(router, "_kimi_k3_routing", False)
    logits, bias = _kimi_like_inputs(tokens, tokens, experts)
    args = (logits, topk, True, groups, topk_group, "sigmoid", 2.5, bias if with_bias else None)
    weights, ids = _impl(router, *args, torch.float32, fn="grouped_topk_entry")
    ref_weights, ref_ids = reference_grouped_topk(*args, False)
    assert ids.dtype == torch.int64
    assert torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)


def test_grouped_topk_entry_dispatches_on_the_kimi_flag(router, monkeypatch):
    monkeypatch.setattr(router, "_kimi_k3_routing", False)
    calls = []
    monkeypatch.setattr(router, "grouped_topk", lambda *a, **k: calls.append("shared"))
    monkeypatch.setattr(router, "kimi_k3_grouped_topk", lambda *a, **k: calls.append("kimi"))
    router.grouped_topk_entry()
    router.enable_kimi_k3_grouped_topk()
    router.grouped_topk_entry()
    assert calls == ["shared", "kimi"]


@pytest.mark.parametrize("fn", ["grouped_topk", "kimi_k3_grouped_topk"])
def test_grouped_topk_accepts_is_padding_like_upstream(router, fn):
    """Upstream GroupedTopKRouter calls grouped_topk(..., is_padding=mask): padded rows get weight 0 and id -1."""
    logits, bias = _kimi_like_inputs(6, 6, 256)
    hidden = torch.zeros(6, 4, dtype=torch.bfloat16)
    is_padding = torch.tensor([False, True, False, False, True, False])
    kwargs = dict(topk=8,
                  renormalize=True,
                  num_expert_group=8,
                  topk_group=4,
                  scoring_func="sigmoid",
                  routed_scaling_factor=2.5,
                  e_score_correction_bias=bias)
    ref_weights, ref_ids = getattr(router, fn)(hidden, logits, **kwargs)
    weights, ids = getattr(router, fn)(hidden, logits, is_padding=is_padding, **kwargs)
    keep = ~is_padding
    assert torch.equal(ids[keep], ref_ids[keep]) and torch.equal(weights[keep], ref_weights[keep])
    assert (ids[is_padding] == -1).all() and (weights[is_padding] == 0).all()
