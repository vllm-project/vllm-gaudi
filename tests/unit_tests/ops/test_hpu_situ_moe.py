# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the exact SiTU routed-expert path (vllm_gaudi/ops/hpu_situ_moe.py, pure torch)."""

import importlib.util
import pathlib

import pytest
import torch

_PATH = pathlib.Path(__file__).resolve().parents[3] / "vllm_gaudi" / "ops" / "hpu_situ_moe.py"
_spec = importlib.util.spec_from_file_location("hpu_situ_moe_under_test", _PATH)
situ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(situ)

BETA, LINEAR_BETA = 4.0, 25.0  # Kimi-K3 checkpoint config


def upstream_situ(x, beta, linear_beta):
    """vllm.model_executor.layers.activation.SituAndMul.forward_native (ac7f3e11e), verbatim."""
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


@pytest.mark.parametrize("linear_beta", [LINEAR_BETA, None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_situ_activation_matches_the_upstream_formula(linear_beta, dtype):
    x = (torch.randn(7, 2, 32, generator=torch.Generator().manual_seed(0)) * 12).to(dtype)  # large -> clipping active
    torch.testing.assert_close(situ.situ_activation(x, BETA, linear_beta), upstream_situ(x, BETA, linear_beta))


def test_situ_is_not_silu_for_large_inputs():
    gate = torch.tensor([[8.0, 1.0]])
    up = torch.tensor([[30.0, 1.0]])
    x = torch.cat([gate, up], dim=-1)
    silu = torch.nn.functional.silu(gate) * up
    exact = situ.situ_activation(x, BETA, LINEAR_BETA)
    assert not torch.allclose(exact[0, 0], silu[0, 0], rtol=0.1)  # clipped gate and up
    assert torch.allclose(exact[0, 1], silu[0, 1], rtol=0.05)  # small inputs: SiTU ~ silu


def _encode_mxfp4(values, scales_exp):
    """values [rows, cols] holding E2M1 magnitudes -> (packed uint8 [rows, cols/2], scale uint8 [rows, cols/32])."""
    lut = list(situ._FP4_VALUES)
    codes = torch.tensor([[lut.index(float(v)) for v in row] for row in values.tolist()], dtype=torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, (scales_exp + 127).to(torch.uint8)


def test_dequant_mxfp4_low_nibble_first_and_e8m0_scale():
    rows, cols = 3, 64
    g = torch.Generator().manual_seed(1)
    magnitudes = torch.tensor(situ._FP4_VALUES)[torch.randint(0, 16, (rows, cols), generator=g)]
    exps = torch.randint(-3, 4, (rows, cols // 32), generator=g)
    packed, scale = _encode_mxfp4(magnitudes, exps)
    got = situ.dequant_mxfp4(packed, scale, torch.float32)
    expected = magnitudes * (2.0**exps.repeat_interleave(32, dim=1).float())
    assert torch.equal(got, expected)
    with pytest.raises(ValueError):
        situ.dequant_mxfp4(packed, scale[:, :1])


def _toy_experts(seed, experts=6, hidden=16, inter=8, pad=0):
    g = torch.Generator().manual_seed(seed)
    w13 = torch.randn(experts, 2 * (inter + pad), hidden, generator=g) * 0.6
    w2 = torch.randn(experts, hidden, inter + pad, generator=g) * 0.6
    if pad:  # zero padded intermediate rows/cols (Kimi pads the MoE intermediate size)
        w13[:, inter:inter + pad] = 0
        w13[:, 2 * inter + pad:] = 0
        w2[:, :, inter:] = 0
    return w13, w2


def naive_reference(x, topk_ids, topk_weights, w13, w2, experts_min, beta, linear_beta):
    """Per token, per routed expert, straight from the definition (fp32)."""
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32)
    d = w13.shape[1] // 2
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j]) - experts_min
            if not 0 <= e < w13.shape[0]:
                continue
            h = w13[e].float() @ x[t].float()
            gate, up = h[:d], h[d:]
            gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            if linear_beta is not None:
                up = linear_beta * torch.tanh(up / linear_beta)
            out[t] += topk_weights[t, j] * (w2[e].float() @ (gate * up))
    return out


def _route(tokens, experts_total, k, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(experts_total, generator=g)[:k] for _ in range(tokens)])
    weights = torch.softmax(torch.randn(tokens, k, generator=g), dim=-1)
    return ids, weights


@pytest.mark.parametrize("tokens,chunk,tile", [(1, 8, 512), (3, 2, 512), (40, 4, 16), (40, 8, 0)])
@pytest.mark.parametrize("experts_min", [0, 6])
@pytest.mark.parametrize("pad", [0, 2])
def test_unfused_routed_moe_matches_naive_reference(tokens, chunk, tile, experts_min, pad):
    experts, hidden, inter, k = 6, 16, 8, 3
    w13, w2 = _toy_experts(3, experts, hidden, inter, pad)
    total = 12  # global experts; this rank owns experts_min..experts_min+5 (EP), the other six are remote
    ids, weights = _route(tokens, total, k, 5)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(6)) * 3
    out = situ.situ_moe_experts(x,
                                ids,
                                weights,
                                lambda e: w13.index_select(0, e),
                                lambda e: w2.index_select(0, e),
                                experts,
                                experts_min,
                                BETA,
                                LINEAR_BETA,
                                expert_chunk=chunk,
                                token_tile=tile,
                                gather_max_tokens=8)
    ref = naive_reference(x, ids, weights, w13, w2, experts_min, BETA, LINEAR_BETA)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_gather_path_equals_dense_path():
    experts, hidden, inter, k = 12, 16, 8, 2
    w13, w2 = _toy_experts(7, experts, hidden, inter)
    ids, weights = _route(2, experts, k, 8)  # 2 tokens * 2 < 12 experts -> gather path
    x = torch.randn(2, hidden, generator=torch.Generator().manual_seed(9))
    args = (lambda e: w13.index_select(0, e), lambda e: w2.index_select(0, e), experts, 0, BETA, None)
    gathered = situ.situ_moe_experts(x, ids, weights, *args, gather_max_tokens=8)
    dense = situ.situ_moe_experts(x, ids, weights, *args, gather_max_tokens=0)
    torch.testing.assert_close(gathered, dense, rtol=1e-5, atol=1e-5)


def test_mxfp4_path_equals_reference_on_dequantized_weights():
    experts, hidden, inter, k = 5, 64, 32, 2
    g = torch.Generator().manual_seed(11)
    mags13 = torch.tensor(situ._FP4_VALUES)[torch.randint(0, 16, (experts, 2 * inter, hidden), generator=g)]
    mags2 = torch.tensor(situ._FP4_VALUES)[torch.randint(0, 16, (experts, hidden, inter), generator=g)]
    exp13 = torch.randint(-5, -3, (experts, 2 * inter, hidden // 32), generator=g)
    exp2 = torch.randint(-5, -3, (experts, hidden, inter // 32), generator=g)
    p13, s13 = zip(*[_encode_mxfp4(mags13[e], exp13[e]) for e in range(experts)])
    p2, s2 = zip(*[_encode_mxfp4(mags2[e], exp2[e]) for e in range(experts)])
    p13, s13, p2, s2 = (torch.stack(t) for t in (p13, s13, p2, s2))
    w13 = situ.dequant_mxfp4(p13, s13, torch.float32)
    w2 = situ.dequant_mxfp4(p2, s2, torch.float32)
    ids, weights = _route(4, experts + 3, k, 12)
    x = torch.randn(4, hidden, generator=torch.Generator().manual_seed(13))
    out = situ.situ_moe_mxfp4(x, ids, weights, p13, s13, p2, s2, 3, BETA, LINEAR_BETA, expert_chunk=2)
    ref = naive_reference(x, ids, weights, w13, w2, 3, BETA, LINEAR_BETA)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_exact_path_differs_from_the_silu_approximation():
    experts, hidden, inter = 4, 16, 8
    w13, w2 = _toy_experts(21, experts, hidden, inter)
    ids, weights = _route(3, experts, 2, 22)
    x = torch.randn(3, hidden, generator=torch.Generator().manual_seed(23)) * 6  # large activations
    exact = situ.situ_moe_experts(x, ids, weights, lambda e: w13.index_select(0, e), lambda e: w2.index_select(0, e),
                                  experts, 0, BETA, LINEAR_BETA)
    # beta -> infinity and no linear clip reduces SiTU to silu
    approx = situ.situ_moe_experts(x, ids, weights, lambda e: w13.index_select(0, e), lambda e: w2.index_select(0, e),
                                   experts, 0, 1e9, None)
    assert not torch.allclose(exact, approx, rtol=0.05, atol=0.05)


def test_flag_resolution():
    assert situ.resolve_situ_impl(None) == "exact"
    assert situ.resolve_situ_impl("") == "exact"
    assert situ.resolve_situ_impl(" SILU ") == "silu"
    assert situ.resolve_situ_impl("exact") == "exact"
    with pytest.raises(ValueError, match="HPU_KIMI_SITU_IMPL"):
        situ.resolve_situ_impl("fused")


def test_env_registry_default_and_override(monkeypatch):
    spec = importlib.util.spec_from_file_location("vllm_gaudi_envs_situ_test", _PATH.parents[1] / "envs.py")
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    monkeypatch.delenv("HPU_KIMI_SITU_IMPL", raising=False)
    assert envs.HPU_KIMI_SITU_IMPL == "exact"
    monkeypatch.setenv("HPU_KIMI_SITU_IMPL", "SiLU")
    assert envs.HPU_KIMI_SITU_IMPL == "silu"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_lut_dequant_equals_the_reference_dequant_for_every_byte_and_scale(dtype):
    g = torch.Generator().manual_seed(31)
    rows, groups = 8, 6
    packed = torch.randint(0, 256, (3, rows, groups * 16), generator=g, dtype=torch.uint8)
    scales = torch.randint(1, 255, (3, rows, groups), generator=g, dtype=torch.uint8)  # 0 / 255 are not real scales
    assert torch.equal(situ.dequant_mxfp4(packed, scales, dtype), situ.dequant_mxfp4_reference(packed, scales, dtype))
    # every one of the 256 byte values and realistic scale exponents (2^-20 .. 2^8)
    all_bytes = torch.arange(256, dtype=torch.uint8).reshape(1, 16, 16).repeat(1, 1, 2).reshape(1, 16, 32)
    sc = torch.arange(107, 135, dtype=torch.uint8)[:2].reshape(1, 1, 2).repeat(1, 16, 1)
    assert torch.equal(situ.dequant_mxfp4(all_bytes, sc, dtype), situ.dequant_mxfp4_reference(all_bytes, sc, dtype))


class _FakeLayer:

    def __init__(self, activation, beta=BETA, linear_beta=LINEAR_BETA, ep_rank=1, experts=4, hidden=64, inter=32):
        g = torch.Generator().manual_seed(41)

        def packed(rows, cols):
            return (torch.randint(0, 256, (experts, rows, cols // 2), generator=g, dtype=torch.uint8),
                    torch.randint(120, 126, (experts, rows, cols // 32), generator=g, dtype=torch.uint8))

        w13, w13s = packed(2 * inter, hidden)
        w2, w2s = packed(hidden, inter)
        self.activation = activation
        self.local_num_experts = experts
        self.moe_config = type(
            "Cfg", (), dict(ep_rank=ep_rank, activation_situ_beta=beta, activation_situ_linear_beta=linear_beta))()
        self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(w13s, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(w2s, requires_grad=False)


class _Enum:  # stands in for MoEActivation.SITU

    def __init__(self, value):
        self.value = value


@pytest.mark.parametrize("activation", ["situ", _Enum("situ")])
def test_maybe_situ_moe_dispatch_for_the_mxfp4_moe_methods(activation):
    layer = _FakeLayer(activation)
    ids, weights = _route(3, 4 * 3, 2, 42)  # rank 1 of 3 owns experts 4..7
    x = torch.randn(3, 64, generator=torch.Generator().manual_seed(43))

    out = situ.maybe_situ_moe(layer, x, ids, weights, "exact")

    w13 = situ.dequant_mxfp4(layer.w13_weight.data, layer.w13_weight_scale.data, torch.float32)
    w2 = situ.dequant_mxfp4(layer.w2_weight.data, layer.w2_weight_scale.data, torch.float32)
    ref = naive_reference(x, ids, weights, w13, w2, 4, BETA, LINEAR_BETA)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


def test_maybe_situ_moe_leaves_other_cases_to_the_fused_kernel_and_validates_inputs():
    x = torch.randn(2, 64)
    ids, weights = _route(2, 12, 2, 44)
    assert situ.maybe_situ_moe(_FakeLayer("silu"), x, ids, weights, "exact") is None
    assert situ.maybe_situ_moe(_FakeLayer("situ"), x, ids, weights, "silu") is None
    with pytest.raises(ValueError, match="groups of 32"):
        situ.maybe_situ_moe(_FakeLayer("situ"), x, ids, weights, "exact", group_size=16)
    with pytest.raises(ValueError, match="activation_situ_beta"):
        situ.maybe_situ_moe(_FakeLayer("situ", beta=None), x, ids, weights, "exact")


def test_invalid_impl_falls_back_to_exact_and_reports():
    errors = []
    assert situ.situ_impl_or_exact("fused", errors.append) == "exact"
    assert len(errors) == 1 and "HPU_KIMI_SITU_IMPL" in errors[0]
    assert situ.situ_impl_or_exact("silu", errors.append) == "silu"
    assert len(errors) == 1


def test_gather_is_used_for_any_token_count_below_the_hit_bound():
    # 8 tokens * 2 experts = 16 < 64 local experts: only 16 experts may be dequantized, however many tokens
    experts, hidden, inter, k = 64, 16, 8, 2
    w13, w2 = _toy_experts(51, experts, hidden, inter)
    ids, weights = _route(8, experts, k, 52)
    x = torch.randn(8, hidden, generator=torch.Generator().manual_seed(53))
    requested = []

    def get_w13(e):
        requested.append(int(e.shape[0]))
        return w13.index_select(0, e)

    out = situ.situ_moe_experts(x, ids, weights, get_w13, lambda e: w2.index_select(0, e), experts, 0, BETA,
                                LINEAR_BETA)
    assert sum(requested) == 16
    torch.testing.assert_close(out,
                               naive_reference(x, ids, weights, w13, w2, 0, BETA, LINEAR_BETA),
                               rtol=1e-4,
                               atol=1e-4)


def test_fused_alias_follows_the_env_at_call_time(monkeypatch):
    monkeypatch.delenv("HPU_KIMI_SITU_IMPL", raising=False)
    assert situ.situ_fused_alias() is None  # exact: a "situ" reaching the fused op must fail loudly
    monkeypatch.setenv("HPU_KIMI_SITU_IMPL", "silu")
    assert situ.situ_fused_alias() == "silu"
    monkeypatch.setenv("HPU_KIMI_SITU_IMPL", "bogus")
    assert situ.situ_fused_alias() is None


def test_normalizer_source_uses_the_lazy_alias():
    # The activation normalizers must not rely on import-time mutation of the alias dicts (circular plugin load).
    import inspect
    import pathlib
    root = pathlib.Path(inspect.getfile(situ)).parents[1]
    for rel in ("extension/ops.py", "ops/hpu_fused_moe.py"):
        text = (root / rel).read_text()
        assert "situ_fused_alias" in text and '"situ": "silu"' not in text
    patches = (root / "patches.py").read_text()
    assert "importlib.import_module(module_name)._MOE_ACTIVATION_ALIASES" not in patches


def _dense_case(tokens):
    experts, hidden, inter, k = 12, 16, 8, 2
    w13, w2 = _toy_experts(11, experts, hidden, inter)
    ids, weights = _route(tokens, experts, k, 13)
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(14))
    return x, ids, weights, w13, w2, experts


def test_dequant_runs_once_per_expert_chunk_not_once_per_token_tile():
    x, ids, weights, w13, w2, experts = _dense_case(40)
    calls = []

    def get(w):
        return lambda e: calls.append(1) or w.index_select(0, e)

    out = situ.situ_moe_experts(x,
                                ids,
                                weights,
                                get(w13),
                                get(w2),
                                experts,
                                0,
                                BETA,
                                LINEAR_BETA,
                                expert_chunk=4,
                                token_tile=8,
                                gather_max_tokens=0)
    assert len(calls) == 2 * (experts // 4)  # 5 token tiles, 3 chunks: 6 dequants, not 30
    ref = naive_reference(x, ids, weights, w13, w2, 0, BETA, LINEAR_BETA)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tokens", [2, 20])  # gather path (T * K < experts) and dense path with token tiles
@pytest.mark.parametrize("cold_luts", [True, False])
def test_traces_as_one_graph_with_parsable_stack_traces(tokens, cold_luts):
    """torch.compile mode: no graph break (also while the LUT cache is filled inside the trace) and every FX node
    has a stack_trace the HPU compile backend can parse."""
    import torch._dynamo as dynamo
    if not hasattr(torch.fx.graph, "_parse_stack_trace"):
        pytest.skip("torch.fx.graph._parse_stack_trace is not available in this torch version")
    experts, inter, hidden, topk = 16, 64, 128, 2
    gen = torch.Generator().manual_seed(0)
    w13 = torch.randint(0, 256, (experts, 2 * inter, hidden // 2), dtype=torch.uint8, generator=gen)
    w13_s = torch.randint(120, 130, (experts, 2 * inter, hidden // 32), dtype=torch.uint8, generator=gen)
    w2 = torch.randint(0, 256, (experts, hidden, inter // 2), dtype=torch.uint8, generator=gen)
    w2_s = torch.randint(120, 130, (experts, hidden, inter // 32), dtype=torch.uint8, generator=gen)
    x = torch.randn(tokens, hidden, generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, experts, (tokens, topk), generator=gen)
    weights = torch.rand(tokens, topk, generator=gen)

    def fn(x, ids, weights):
        return situ.situ_moe_mxfp4(x, ids, weights, w13, w13_s, w2, w2_s, 0, BETA, LINEAR_BETA, token_tile=8)

    graphs, unparsable = [], []

    def backend(gm, example_inputs):
        graphs.append(gm)
        for node in gm.graph.nodes:
            if node.op in ("call_function",
                           "call_method") and (node.stack_trace is None
                                               or torch.fx.graph._parse_stack_trace(node.stack_trace) is None):
                unparsable.append(node.name)
        return gm.forward

    if cold_luts:
        situ._LUT_CACHE.clear()
    dynamo.reset()
    try:
        out = torch.compile(fn, backend=backend)(x, ids, weights)
    finally:
        dynamo.reset()
    assert len(graphs) == 1
    assert not unparsable
    torch.testing.assert_close(out, fn(x, ids, weights))


def test_maybe_situ_moe_on_a_layer_traces_as_one_graph_in_compile_mode():
    """The layer-level entry point (``.data`` reads, config attributes) must not break the graph."""
    import types

    import torch._dynamo as dynamo

    experts, inter, hidden = 16, 64, 128
    gen = torch.Generator().manual_seed(1)
    param = lambda *shape, lo, hi: torch.nn.Parameter(  # noqa: E731
        torch.randint(lo, hi, shape, dtype=torch.uint8, generator=gen), requires_grad=False)
    layer = types.SimpleNamespace(activation="situ",
                                  local_num_experts=experts,
                                  moe_config=types.SimpleNamespace(ep_rank=0,
                                                                   activation_situ_beta=BETA,
                                                                   activation_situ_linear_beta=LINEAR_BETA),
                                  w13_weight=param(experts, 2 * inter, hidden // 2, lo=0, hi=256),
                                  w13_weight_scale=param(experts, 2 * inter, hidden // 32, lo=120, hi=130),
                                  w2_weight=param(experts, hidden, inter // 2, lo=0, hi=256),
                                  w2_weight_scale=param(experts, hidden, inter // 32, lo=120, hi=130))
    x = torch.randn(20, hidden, generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, experts, (20, 2), generator=gen)
    weights = torch.rand(20, 2, generator=gen)

    def fn(x, ids, weights):
        return situ.maybe_situ_moe(layer, x, ids, weights, "exact", token_tile=8)

    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    dynamo.reset()
    try:
        out = torch.compile(fn, backend=backend)(x, ids, weights)
        expected = fn(x, ids, weights)
    finally:
        dynamo.reset()
    assert len(graphs) == 1
    torch.testing.assert_close(out, expected)
