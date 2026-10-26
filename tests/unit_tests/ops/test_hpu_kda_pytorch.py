# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the pure PyTorch Kimi-K3 KDA (vllm_gaudi/ops/hpu_kda_pytorch.py).

The module is loaded by file path (it has no vLLM / Habana dependencies) and compared with a naive reference
written straight from the KDA math: per head, per token loops with explicit outer products.
"""

import importlib.util
import math
import pathlib
import subprocess
import sys
import types

import pytest
import torch

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[3] / "vllm_gaudi" / "ops" / "hpu_kda_pytorch.py"
_spec = importlib.util.spec_from_file_location("hpu_kda_pytorch_under_test", _MODULE_PATH)
kda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kda)

H, D, W = 3, 8, 4  # heads, head_dim, conv kernel
P = H * D
C = 3 * P
LOWER_BOUND = -5.0


def _make_params(seed: int, lower_bound=LOWER_BOUND):
    g = torch.Generator().manual_seed(seed)
    return dict(
        conv_weight=torch.randn(C, W, generator=g) * 0.5,
        conv_bias=None,
        A_log=torch.randn(H, generator=g) * 0.3,
        dt_bias=torch.randn(P, generator=g) * 0.3,
        lower_bound=lower_bound,
    )


def _make_inputs(seed: int, length: int):
    g = torch.Generator().manual_seed(seed)
    return dict(
        mixed_qkv=torch.randn(length, C, generator=g),
        raw_g=torch.randn(length, P, generator=g),
        raw_beta=torch.randn(length, H, generator=g),
    )


def reference_kda(x, params, conv_state=None, ssm_state=None):
    """Naive KDA for ONE sequence. x: dict(mixed_qkv [T,C], raw_g [T,P], raw_beta [T,H]).

    conv_state [C, W-1] (oldest first) / ssm_state [H, D(v), D(k)] default to zeros.
    Returns (out [T,H,D], conv_state, ssm_state), all fp32.
    """
    T = x["mixed_qkv"].shape[0]
    conv_state = torch.zeros(C, W - 1) if conv_state is None else conv_state.clone().float()
    ssm_state = torch.zeros(H, D, D) if ssm_state is None else ssm_state.clone().float()
    xin = x["mixed_qkv"].float()
    outs = []
    for t in range(T):
        window = torch.cat([conv_state, xin[t].unsqueeze(1)], dim=1)  # [C, W]
        conv_out = torch.nn.functional.silu((window * params["conv_weight"]).sum(dim=1))
        conv_state = window[:, 1:]
        q, k, v = conv_out[:P], conv_out[P:2 * P], conv_out[2 * P:]
        out_t = torch.zeros(H, D)
        for h in range(H):
            qh, kh, vh = q[h * D:(h + 1) * D], k[h * D:(h + 1) * D], v[h * D:(h + 1) * D]
            qh = qh / math.sqrt(float((qh * qh).sum()) + 1e-6) * D**-0.5
            kh = kh / math.sqrt(float((kh * kh).sum()) + 1e-6)
            raw = x["raw_g"][t, h * D:(h + 1) * D].float() + params["dt_bias"][h * D:(h + 1) * D]
            a = math.exp(float(params["A_log"][h]))
            if params["lower_bound"] is not None:
                gate = params["lower_bound"] * torch.sigmoid(a * raw)
            else:
                gate = -a * torch.nn.functional.softplus(raw)
            beta = torch.sigmoid(x["raw_beta"][t, h].float())
            S = ssm_state[h] * torch.exp(gate).unsqueeze(0)  # decay along k
            vnew = beta * (vh - S @ kh)
            S = S + torch.outer(vnew, kh)
            ssm_state[h] = S
            out_t[h] = S @ qh
        outs.append(out_t)
    return torch.stack(outs), conv_state, ssm_state


def _new_caches(slots: int, dim_first: bool, dtype=torch.float32):
    conv = torch.zeros(slots, C, W - 1, dtype=dtype) if dim_first else torch.zeros(slots, W - 1, C, dtype=dtype)
    ssm = torch.zeros(slots, H, D, D, dtype=dtype)
    return conv, ssm


def _conv_of(conv_cache, slot, dim_first):
    s = conv_cache[slot].float()
    return s if dim_first else s.t()


def _pad_batch(seqs, length):
    """seqs: list of input dicts of different lengths -> uniform padded [B, L, *] + token mask [B, L]."""
    bsz = len(seqs)
    out = {k: torch.zeros(bsz, length, v.shape[-1]) for k, v in seqs[0].items()}
    mask = torch.zeros(bsz, length)
    for i, s in enumerate(seqs):
        n = s["mixed_qkv"].shape[0]
        for k in out:
            out[k][i, :n] = s[k]
        mask[i, :n] = 1.0
    return out, mask


def _run_prefill(params, batch, mask, caches, state_idx, has_init):
    conv, ssm = caches
    return kda.kda_core_prefill(batch["mixed_qkv"],
                                batch["raw_g"],
                                batch["raw_beta"],
                                params["conv_weight"],
                                params["conv_bias"],
                                params["A_log"],
                                params["dt_bias"],
                                params["lower_bound"],
                                conv,
                                ssm,
                                state_idx,
                                has_init,
                                mask,
                                H,
                                D,
                                block=4)


def _run_decode(params, tok, caches, state_idx):
    conv, ssm = caches
    return kda.kda_core_decode(tok["mixed_qkv"], tok["raw_g"], tok["raw_beta"], params["conv_weight"],
                               params["conv_bias"], params["A_log"], params["dt_bias"], params["lower_bound"], conv,
                               ssm, state_idx, H, D)


@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("lower_bound", [LOWER_BOUND, None])
def test_single_sequence_prefill_matches_reference(dim_first, lower_bound):
    params = _make_params(0, lower_bound)
    x = _make_inputs(1, 11)
    batch, mask = _pad_batch([x], 11)
    caches = _new_caches(5, dim_first)
    idx = torch.tensor([2])

    out = _run_prefill(params, batch, mask, caches, idx, torch.tensor([0]))

    ref_out, ref_conv, ref_ssm = reference_kda(x, params)
    torch.testing.assert_close(out[0], ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_conv_of(caches[0], 2, dim_first), ref_conv, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(caches[1][2], ref_ssm, rtol=1e-5, atol=1e-5)
    # untouched slots stay zero
    assert float(caches[1][0].abs().sum()) == 0.0


def test_two_sequences_different_lengths_and_initial_states_do_not_mix():
    params = _make_params(2)
    dim_first = True
    xa, xb = _make_inputs(3, 9), _make_inputs(4, 5)
    # sequence B continues from a previous chunk (has_initial_state = 1), sequence A starts fresh
    prev_b = _make_inputs(5, 6)
    _, conv_b0, ssm_b0 = reference_kda(prev_b, params)
    caches = _new_caches(6, dim_first)
    caches[0][3] = conv_b0
    caches[1][3] = ssm_b0
    # garbage in A's slot must be ignored because has_initial_state = 0 for A
    caches[0][1].fill_(7.0)
    caches[1][1].fill_(7.0)

    batch, mask = _pad_batch([xa, xb], 9)
    out = _run_prefill(params, batch, mask, caches, torch.tensor([1, 3]), torch.tensor([0, 1]))

    ref_a, conv_a, ssm_a = reference_kda(xa, params)
    ref_b, conv_b, ssm_b = reference_kda(xb, params, conv_b0, ssm_b0)
    torch.testing.assert_close(out[0, :9], ref_a, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[1, :5], ref_b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_conv_of(caches[0], 1, dim_first), conv_a, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_conv_of(caches[0], 3, dim_first), conv_b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(caches[1][1], ssm_a, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(caches[1][3], ssm_b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dim_first", [True, False])
def test_prefill_then_decode_keeps_state_continuity(dim_first):
    params = _make_params(6)
    full = _make_inputs(7, 12)
    prefill_len = 8
    caches = _new_caches(4, dim_first)
    idx = torch.tensor([1])

    head = {k: v[:prefill_len] for k, v in full.items()}
    batch, mask = _pad_batch([head], 10)  # padded bucket longer than the real prompt
    _run_prefill(params, batch, mask, caches, idx, torch.tensor([0]))

    decoded = []
    for t in range(prefill_len, 12):
        tok = {k: v[t:t + 1] for k, v in full.items()}
        decoded.append(_run_decode(params, tok, caches, idx)[0])

    ref_out, ref_conv, ref_ssm = reference_kda(full, params)
    torch.testing.assert_close(torch.stack(decoded), ref_out[prefill_len:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_conv_of(caches[0], 1, dim_first), ref_conv, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(caches[1][1], ref_ssm, rtol=1e-5, atol=1e-5)


def test_chunked_prefill_continuation_matches_one_shot():
    params = _make_params(8)
    full = _make_inputs(9, 14)
    caches = _new_caches(3, True)
    idx = torch.tensor([2])
    first = {k: v[:6] for k, v in full.items()}
    second = {k: v[6:] for k, v in full.items()}
    b1, m1 = _pad_batch([first], 8)
    b2, m2 = _pad_batch([second], 8)
    _run_prefill(params, b1, m1, caches, idx, torch.tensor([0]))
    out2 = _run_prefill(params, b2, m2, caches, idx, torch.tensor([1]))

    ref_out, _, ref_ssm = reference_kda(full, params)
    torch.testing.assert_close(out2[0, :8], ref_out[6:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(caches[1][2], ref_ssm, rtol=1e-5, atol=1e-5)


def test_padding_rows_and_tokens_do_not_touch_real_state():
    params = _make_params(10)
    dim_first = True
    x = _make_inputs(11, 6)
    caches = _new_caches(5, dim_first)
    # batch of 3 rows: one real sequence, two padded rows (state index -1 -> garbage slot = last slot)
    batch, mask = _pad_batch([x, _make_inputs(12, 1), _make_inputs(13, 1)], 8)
    mask[1:] = 0.0  # fully padded rows
    idx = torch.tensor([0, -1, -1])
    out = _run_prefill(params, batch, mask, caches, idx, torch.tensor([0, 0, 0]))

    ref_out, ref_conv, ref_ssm = reference_kda(x, params)
    torch.testing.assert_close(out[0, :6], ref_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(caches[1][0], ref_ssm, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(_conv_of(caches[0], 0, dim_first), ref_conv, rtol=1e-5, atol=1e-5)
    # slots 1..3 untouched; padded rows only touch the garbage slot (last)
    assert float(caches[1][1:4].abs().sum()) == 0.0
    assert float(caches[0][1:4].abs().sum()) == 0.0
    assert torch.isfinite(out).all()


def test_decode_batch_with_padded_row():
    params = _make_params(14)
    x1, x2 = _make_inputs(15, 5), _make_inputs(16, 7)
    caches = _new_caches(6, True)
    for slot, x in ((1, x1), (4, x2)):
        b, m = _pad_batch([x], 7)
        _run_prefill(params, b, m, caches, torch.tensor([slot]), torch.tensor([0]))
    nxt1, nxt2 = _make_inputs(17, 1), _make_inputs(18, 1)
    tok = {k: torch.cat([nxt1[k], nxt2[k], torch.zeros_like(nxt1[k])], dim=0) for k in nxt1}
    out = _run_decode(params, tok, caches, torch.tensor([1, 4, -1]))

    cat1 = {k: torch.cat([x1[k], nxt1[k]]) for k in x1}
    cat2 = {k: torch.cat([x2[k], nxt2[k]]) for k in x2}
    torch.testing.assert_close(out[0], reference_kda(cat1, params)[0][-1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[1], reference_kda(cat2, params)[0][-1], rtol=1e-5, atol=1e-5)
    assert torch.isfinite(out).all()


def test_bf16_caches_and_inputs_stay_close_to_fp32_reference():
    params = _make_params(19)
    x = _make_inputs(20, 10)
    x_bf16 = {k: v.to(torch.bfloat16) for k, v in x.items()}
    caches = _new_caches(3, True, dtype=torch.bfloat16)
    batch, mask = _pad_batch([x_bf16], 10)
    batch = {k: v.to(torch.bfloat16) for k, v in batch.items()}
    out = _run_prefill(params, batch, mask, caches, torch.tensor([1]), torch.tensor([0]))

    ref_out, _, ref_ssm = reference_kda({k: v.float() for k, v in x_bf16.items()}, params)
    torch.testing.assert_close(out[0], ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(caches[1][1].float(), ref_ssm, rtol=3e-2, atol=3e-2)


def test_gated_rmsnorm_sigmoid_matches_formula():
    g = torch.Generator().manual_seed(21)
    x = torch.randn(4, H, D, generator=g)
    gate = torch.randn(4, H, D, generator=g)
    weight = torch.randn(D, generator=g)
    eps = 1e-5
    ref = x / torch.sqrt((x**2).mean(-1, keepdim=True) + eps) * weight * torch.sigmoid(gate)
    torch.testing.assert_close(kda.gated_rmsnorm_sigmoid(x, gate, weight, eps), ref, rtol=1e-5, atol=1e-5)


def test_step_hook_is_called_between_blocks_only():
    calls = []
    params = _make_params(22)
    x = _make_inputs(23, 10)
    batch, mask = _pad_batch([x], 10)
    caches = _new_caches(3, True)
    conv, ssm = caches
    kda.kda_core_prefill(batch["mixed_qkv"], batch["raw_g"], batch["raw_beta"], params["conv_weight"], None,
                         params["A_log"], params["dt_bias"], params["lower_bound"], conv, ssm, torch.tensor([1]),
                         torch.tensor([0]), mask, H, D, block=4, step_hook=lambda: calls.append(1),
                         prefill_impl="loop")
    assert len(calls) == 2  # after tokens 4 and 8 (not after the last token)


# ---------------------------------------------------------------------------------------------------------------
# Layer-level wiring (models/kimi_k3_kda.py) with stubbed vLLM / Habana modules
# ---------------------------------------------------------------------------------------------------------------
def _load_envs_module():
    spec = importlib.util.spec_from_file_location("vllm_gaudi_envs_under_test", _MODULE_PATH.parents[1] / "envs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_layer_module(monkeypatch, metadata):
    import sys

    habana = types.ModuleType("habana_frameworks")
    habana_torch = types.ModuleType("habana_frameworks.torch")
    habana_torch.core = types.SimpleNamespace(mark_step=lambda: None)
    habana_torch.utils = types.SimpleNamespace(internal=types.SimpleNamespace(is_lazy=lambda: False))
    habana.torch = habana_torch
    forward_context = types.ModuleType("vllm.forward_context")
    forward_context.get_forward_context = lambda: types.SimpleNamespace(attn_metadata=metadata["value"])
    for name, mod in (("habana_frameworks", habana), ("habana_frameworks.torch", habana_torch),
                      ("vllm", types.ModuleType("vllm")), ("vllm.forward_context", forward_context),
                      ("vllm_gaudi", types.ModuleType("vllm_gaudi")),
                      ("vllm_gaudi.ops", types.ModuleType("vllm_gaudi.ops")), ("vllm_gaudi.ops.hpu_kda_pytorch", kda),
                      ("vllm_gaudi.envs", _load_envs_module())):
        monkeypatch.setitem(sys.modules, name, mod)
    path = _MODULE_PATH.parents[1] / "models" / "kimi_k3_kda.py"
    spec = importlib.util.spec_from_file_location("kimi_k3_kda_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLayer:

    def __init__(self, params, hidden, seed, caches):
        g = torch.Generator().manual_seed(seed)
        self.local_num_heads, self.head_dim, self.local_projection_size = H, D, P
        self.in_proj_padding = 2
        self.gate_lower_bound = params["lower_bound"]
        self.w_in = torch.randn(3 * P + P + D + H + self.in_proj_padding, hidden, generator=g) * 0.3
        self.w_fb = torch.randn(P, D, generator=g) * 0.3
        self.w_o = torch.randn(hidden, P, generator=g) * 0.3
        self.norm_w = torch.randn(D, generator=g)
        self.conv1d = types.SimpleNamespace(weight=params["conv_weight"].unsqueeze(1), bias=None)
        self.A_log, self.dt_bias = params["A_log"], params["dt_bias"]
        self.o_norm = types.SimpleNamespace(weight=self.norm_w, eps=1e-5)
        self.kv_cache = caches
        self.cache_group_idx = torch.tensor(1)

    def in_proj_qkvgfab(self, x):
        return (x @ self.w_in.t(), None)

    def f_b_proj(self, x):
        return (x @ self.w_fb.t(), None)

    def o_proj(self, x):
        return (x @ self.w_o.t(), None)



def _reference_layer(layer, hidden_states, params, conv_state=None, ssm_state=None):
    proj = hidden_states @ layer.w_in.t()
    qkv, gate = proj[:, :3 * P], proj[:, 3 * P:4 * P]
    f_a, beta = proj[:, 4 * P:4 * P + D], proj[:, 4 * P + D:4 * P + D + H]
    x = dict(mixed_qkv=qkv, raw_g=f_a @ layer.w_fb.t(), raw_beta=beta)
    out, conv_state, ssm_state = reference_kda(x, params, conv_state, ssm_state)
    normed = out / torch.sqrt((out**2).mean(-1, keepdim=True) + 1e-5) * layer.norm_w
    normed = normed * torch.sigmoid(gate.view(-1, H, D))
    return normed.reshape(-1, P) @ layer.w_o.t(), conv_state, ssm_state


def test_layer_forward_prefill_then_decode_matches_reference(monkeypatch):
    params = _make_params(30)
    hidden = 16
    caches = _new_caches(4, False)  # SD layout like the HPU Qwen path
    metadata = {"value": None}
    module = _load_layer_module(monkeypatch, metadata)
    layer = _FakeLayer(params, hidden, 31, caches)

    g = torch.Generator().manual_seed(32)
    hs = torch.randn(10, hidden, generator=g)
    # two cache groups -> load_indices is [groups, batch]; the layer picks row 1. Prompt has 7 real tokens in an
    # 8-token bucket, plus one padded row.
    prompt = torch.zeros(2, 8, hidden)
    prompt[0, :7] = hs[:7]
    mask = torch.zeros(2, 8)
    mask[0, :7] = 1.0
    metadata["value"] = types.SimpleNamespace(is_prompt=True,
                                               load_indices_tensor=torch.tensor([[0, 0], [2, -1]]),
                                               padding_mask_flat=mask.reshape(-1, 1),
                                               has_initial_states_p=torch.tensor([0, 0], dtype=torch.int32))
    out = module.hpu_kda_forward(layer, prompt.reshape(-1, hidden))
    ref, conv_s, ssm_s = _reference_layer(layer, hs[:7], params)
    torch.testing.assert_close(out.reshape(2, 8, hidden)[0, :7], ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(caches[1][2], ssm_s, rtol=1e-5, atol=1e-5)

    for t in range(7, 10):
        metadata["value"] = types.SimpleNamespace(is_prompt=False, load_indices_tensor=torch.tensor([[0], [2]]))
        out = module.hpu_kda_forward(layer, hs[t:t + 1])
        ref_all, _, _ = _reference_layer(layer, hs[:t + 1], params)
        torch.testing.assert_close(out[0], ref_all[-1], rtol=1e-4, atol=1e-4)


def test_layer_forward_without_metadata_returns_zero_core(monkeypatch):
    params = _make_params(33)
    metadata = {"value": None}
    module = _load_layer_module(monkeypatch, metadata)
    layer = _FakeLayer(params, 16, 34, ())
    out = module.hpu_kda_forward(layer, torch.randn(5, 16))
    assert out.shape == (5, 16)
    assert float(out.abs().sum()) == 0.0


# ---------------------------------------------------------------------------------------------------------------
# KV cache group tagging
# ---------------------------------------------------------------------------------------------------------------
def _fake_group(layer_names, mamba):
    return types.SimpleNamespace(layer_names=layer_names, kind="mamba" if mamba else "attn")


def _fake_kda(prefix):
    return types.SimpleNamespace(prefix=prefix, in_proj_qkvgfab=object())


def test_tag_kda_cache_groups_uses_layer_prefix_not_module_path():
    groups = [
        _fake_group(["lm.layers.0.self_attn", "lm.layers.4.self_attn"], mamba=False),  # MLA
        _fake_group(["lm.layers.1.self_attn", "lm.layers.2.self_attn"], mamba=True),
        _fake_group(["lm.layers.3.self_attn", "lm.layers.5.self_attn"], mamba=True),
    ]
    mla = types.SimpleNamespace(prefix="lm.layers.0.self_attn")
    k1, k2, k3, k5 = (_fake_kda(f"lm.layers.{i}.self_attn") for i in (1, 2, 3, 5))
    # module paths deliberately differ from the registered prefixes (wrapper name, different nesting)
    modules = [("model.language_model.layers.0.self_attn", mla), ("model.language_model.layers.1.self_attn", k1),
               ("model.language_model.layers.2.self_attn", k2), ("x.3", k3), ("x.5", k5)]

    tagged, untagged, unmatched = kda.tag_kda_cache_groups(modules, groups, lambda g: g.kind == "mamba",
                                                           lambda gid: torch.tensor(gid))

    assert tagged == {1: 2, 2: 2}
    assert untagged == [] and unmatched == []
    assert [int(m.cache_group_idx) for m in (k1, k2, k3, k5)] == [1, 1, 2, 2]
    assert not hasattr(mla, "cache_group_idx")


def test_tag_kda_cache_groups_reports_untagged_and_unmatched():
    groups = [_fake_group(["a.self_attn", "ghost.self_attn"], mamba=True)]
    known, stray = _fake_kda("a.self_attn"), _fake_kda("b.self_attn")

    tagged, untagged, unmatched = kda.tag_kda_cache_groups([("m.a", known), ("m.b", stray)], groups,
                                                           lambda g: g.kind == "mamba", lambda gid: gid)

    assert tagged == {0: 1}
    assert untagged == ["b.self_attn"]
    assert unmatched == ["ghost.self_attn"]
    assert known.cache_group_idx == 0 and not hasattr(stray, "cache_group_idx")


def test_tag_kda_cache_groups_falls_back_to_module_name_without_prefix():
    groups = [_fake_group(["layers.1.self_attn"], mamba=True)]
    module = types.SimpleNamespace(in_proj_qkvgfab=object())  # no prefix attribute
    tagged, untagged, _ = kda.tag_kda_cache_groups([("layers.1.self_attn", module)], groups,
                                                   lambda g: g.kind == "mamba", lambda gid: gid)
    assert tagged == {0: 1} and untagged == [] and module.cache_group_idx == 0


def _unparsable_stack_trace_nodes(layer_forward, layer, hidden_states):
    """Compile with a backend that checks every FX node the way the HPU lowering reads stack traces."""
    from torch.fx.graph import _parse_stack_trace

    bad = []

    def backend(gm, example_inputs):
        for node in gm.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.stack_trace is None or _parse_stack_trace(node.stack_trace) is None:
                bad.append((node.op, str(node.target), node.stack_trace))
        return gm.forward

    compiled = torch.compile(layer_forward, backend=backend, dynamic=False)
    compiled(layer, hidden_states)
    return bad


def _compile_check(mode):
    """Runs in a fresh interpreter (see the test): dynamo state from an earlier compile in the same process makes
    the stubbed forward context of a later compile resolve to the earlier metadata."""
    params = _make_params(40)
    hidden = 16
    caches = _new_caches(4, False)
    metadata = {"value": None}

    class _MonkeyPatch:

        def setitem(self, mapping, key, value):
            mapping[key] = value

    module = _load_layer_module(_MonkeyPatch(), metadata)
    layer = _FakeLayer(params, hidden, 41, caches)
    g = torch.Generator().manual_seed(42)
    if mode == "prefill":
        metadata["value"] = types.SimpleNamespace(is_prompt=True,
                                                   load_indices_tensor=torch.tensor([[0, 0], [1, 2]]),
                                                   padding_mask_flat=torch.ones(16, 1),
                                                   has_initial_states_p=torch.tensor([0, 0], dtype=torch.int32))
        hidden_states = torch.randn(16, hidden, generator=g)
    else:
        metadata["value"] = types.SimpleNamespace(is_prompt=False, load_indices_tensor=torch.tensor([[0], [1]]))
        hidden_states = torch.randn(1, hidden, generator=g)
    return _unparsable_stack_trace_nodes(module.hpu_kda_forward, layer, hidden_states)


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_compiled_layer_forward_has_parsable_fx_stack_traces(mode):
    """Regression: ops chained on the last line of the call to the torch.compiler.disable'd KDA core ended up in
    the dynamo resume frame without a source line; the HPU compile backend then crashed on the FX stack_trace
    ("'NoneType' object has no attribute 'file'")."""
    pytest.importorskip("torch._dynamo")
    if not hasattr(torch.fx.graph, "_parse_stack_trace"):
        pytest.skip("torch.fx.graph._parse_stack_trace is not available in this torch version")
    result = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), mode],
                            capture_output=True,
                            text=True,
                            timeout=600)
    assert result.returncode == 0, (
        f"FX nodes without a parsable stack_trace or compile error:\n{result.stdout}\n{result.stderr[-2000:]}")


if __name__ == "__main__":
    unparsable = _compile_check(sys.argv[1])
    for item in unparsable:
        print(item)
    sys.exit(1 if unparsable else 0)


# ---------------------------------------------------------------------------------------------------------------
# Chunked (WY form) prefill vs token loop vs naive reference
# ---------------------------------------------------------------------------------------------------------------
def _random_delta_inputs(seed, batch, length, heads=H, dim=D, gate_scale=5.0):
    g = torch.Generator().manual_seed(seed)
    q = kda.l2norm(torch.randn(batch, length, heads, dim, generator=g)) * dim**-0.5
    k = kda.l2norm(torch.randn(batch, length, heads, dim, generator=g))
    v = torch.randn(batch, length, heads, dim, generator=g)
    log_gate = -gate_scale * torch.sigmoid(torch.randn(batch, length, heads, dim, generator=g))  # (-5, 0)
    beta = torch.sigmoid(torch.randn(batch, length, heads, generator=g))
    state = torch.randn(batch, heads, dim, dim, generator=g) * 0.1
    return q, k, v, log_gate, beta, state


@pytest.mark.parametrize("length,chunk", [(1, 8), (7, 8), (8, 8), (13, 8), (64, 16), (100, 32), (1030, 64)])
def test_chunked_matches_token_loop(length, chunk):
    q, k, v, log_gate, beta, state = _random_delta_inputs(50 + length, 2, length)
    out_loop, state_loop = kda.kda_recurrent_steps(state, q, k, v, torch.exp(log_gate), beta)
    out_chunk, state_chunk = kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=chunk)
    assert out_chunk.shape == out_loop.shape
    torch.testing.assert_close(out_chunk, out_loop, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_chunk, state_loop, rtol=1e-4, atol=1e-4)


def test_chunked_is_finite_and_matches_for_extreme_gates():
    # gate -5 on every step: exp(-sum g) over a 64 chunk would be exp(320) (overflow), the relative form must not
    q, k, v, _, beta, state = _random_delta_inputs(60, 1, 256)
    log_gate = torch.full((1, 256, H, D), -5.0)
    out_loop, state_loop = kda.kda_recurrent_steps(state, q, k, v, torch.exp(log_gate), beta)
    out_chunk, state_chunk = kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=64)
    assert torch.isfinite(out_chunk).all() and torch.isfinite(state_chunk).all()
    torch.testing.assert_close(out_chunk, out_loop, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_chunk, state_loop, rtol=1e-4, atol=1e-4)


def test_chunked_matches_for_nearly_equal_keys():
    # beta close to 1 and repeated key directions stress the triangular inverse
    q, k, v, log_gate, beta, state = _random_delta_inputs(61, 1, 128, gate_scale=0.05)
    k = kda.l2norm(k[:, :1].expand_as(k) + 0.01 * k)
    beta = torch.full_like(beta, 0.99)
    out_loop, state_loop = kda.kda_recurrent_steps(state, q, k, v, torch.exp(log_gate), beta)
    out_chunk, state_chunk = kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=64)
    torch.testing.assert_close(out_chunk, out_loop, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(state_chunk, state_loop, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("impl", ["chunked", "loop"])
def test_core_prefill_impls_match_reference_long_multi_sequence(impl):
    params = _make_params(70)
    xa, xb = _make_inputs(71, 150), _make_inputs(72, 97)  # lengths not divisible by the chunk size
    prev_b = _make_inputs(73, 20)
    _, conv_b0, ssm_b0 = reference_kda(prev_b, params)
    caches = _new_caches(6, True)
    caches[0][3] = conv_b0
    caches[1][3] = ssm_b0
    caches[1][1].fill_(7.0)  # ignored: has_initial_state = 0
    batch, mask = _pad_batch([xa, xb], 160)
    conv, ssm = caches

    out = kda.kda_core_prefill(batch["mixed_qkv"], batch["raw_g"], batch["raw_beta"], params["conv_weight"], None,
                               params["A_log"], params["dt_bias"], params["lower_bound"], conv, ssm,
                               torch.tensor([1, 3]), torch.tensor([0, 1]), mask, H, D, block=32, prefill_impl=impl,
                               chunk_size=32)

    ref_a, conv_a, ssm_a = reference_kda(xa, params)
    ref_b, conv_b, ssm_b = reference_kda(xb, params, conv_b0, ssm_b0)
    torch.testing.assert_close(out[0, :150], ref_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[1, :97], ref_b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(caches[1][1], ssm_a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(caches[1][3], ssm_b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(_conv_of(caches[0], 3, True), conv_b, rtol=1e-4, atol=1e-4)


def test_chunked_prefill_then_decode_continues_the_state():
    params = _make_params(80)
    full = _make_inputs(81, 140)
    caches = _new_caches(3, True)
    head = {key: val[:130] for key, val in full.items()}
    batch, mask = _pad_batch([head], 192)  # 192 = 6 chunks of 32, 62 padded tokens
    conv, ssm = caches
    kda.kda_core_prefill(batch["mixed_qkv"], batch["raw_g"], batch["raw_beta"], params["conv_weight"], None,
                         params["A_log"], params["dt_bias"], params["lower_bound"], conv, ssm, torch.tensor([2]),
                         torch.tensor([0]), mask, H, D, prefill_impl="chunked", chunk_size=32)
    decoded = [_run_decode(params, {key: val[t:t + 1] for key, val in full.items()}, caches, torch.tensor([2]))[0]
               for t in range(130, 140)]
    ref_out, _, ref_ssm = reference_kda(full, params)
    torch.testing.assert_close(torch.stack(decoded), ref_out[130:], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(caches[1][2], ref_ssm, rtol=1e-4, atol=1e-4)


def test_chunked_bf16_inputs_close_to_fp32():
    q, k, v, log_gate, beta, state = _random_delta_inputs(90, 1, 200)
    out32, state32 = kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=64)
    bf = lambda x: x.to(torch.bfloat16).float()  # noqa: E731
    out16, state16 = kda.kda_chunked_steps(bf(state), bf(q), bf(k), bf(v), log_gate, bf(beta), chunk_size=64)
    torch.testing.assert_close(out16, out32, rtol=0.1, atol=0.1)
    torch.testing.assert_close(state16, state32, rtol=0.1, atol=0.1)


def test_chunked_hook_called_between_chunk_blocks_only():
    q, k, v, log_gate, beta, state = _random_delta_inputs(91, 1, 40)
    calls = []
    kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=4, block_chunks=3,
                          step_hook=lambda: calls.append(1))
    assert len(calls) == 3  # 10 chunks: after chunks 3, 6 and 9


def test_chunked_runs_far_fewer_ops_than_the_token_loop():
    """Sanity check of the speed-up: the Python-level op count (what a lazy/HPU graph has to carry) is much lower."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Counter(TorchDispatchMode):

        def __init__(self):
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.count += 1
            return func(*args, **(kwargs or {}))

    q, k, v, log_gate, beta, state = _random_delta_inputs(92, 1, 512)
    with Counter() as loop_ops:
        kda.kda_recurrent_steps(state, q, k, v, torch.exp(log_gate), beta)
    with Counter() as chunk_ops:
        kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=64)
    assert chunk_ops.count * 4 < loop_ops.count


@pytest.mark.parametrize("size", [8, 16, 48, 64])
def test_unit_lower_inverse_matches_exact_inverse_even_when_ill_conditioned(size):
    g = torch.Generator().manual_seed(100 + size)
    for scale in (0.3, 0.99):
        a = torch.tril(torch.rand(size, size, generator=g, dtype=torch.float64) * scale, -1)
        ref = torch.linalg.inv(torch.eye(size, dtype=torch.float64) + a)
        got = kda._unit_lower_inverse(a.float().expand(2, 3, size, size)).double()
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got[1, 2], ref, rtol=1e-3, atol=1e-3 * float(ref.abs().max()))


def test_chunked_handles_identical_keys_with_beta_one():
    q, k, v, _, beta, state = _random_delta_inputs(101, 1, 130)
    k = k[:, :1].expand_as(k).contiguous()
    beta = torch.ones_like(beta)
    log_gate = torch.zeros(1, 130, H, D)
    out_loop, state_loop = kda.kda_recurrent_steps(state, q, k, v, torch.exp(log_gate), beta)
    out_chunk, state_chunk = kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=64)
    torch.testing.assert_close(out_chunk, out_loop, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_chunk, state_loop, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunk", [0, -16, 24, 100, 17])
def test_chunked_rejects_invalid_chunk_sizes(chunk):
    q, k, v, log_gate, beta, state = _random_delta_inputs(110, 1, 40)
    with pytest.raises(ValueError, match="chunk_size"):
        kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=chunk)


@pytest.mark.parametrize("chunk", [5, 16, 32, 48, 128])
def test_chunked_valid_non_default_chunk_sizes_match_the_loop(chunk):
    q, k, v, log_gate, beta, state = _random_delta_inputs(111, 1, 300)
    out_loop, state_loop = kda.kda_recurrent_steps(state, q, k, v, torch.exp(log_gate), beta)
    out_chunk, state_chunk = kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=chunk)
    torch.testing.assert_close(out_chunk, out_loop, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_chunk, state_loop, rtol=1e-4, atol=1e-4)


def test_prefill_settings_come_from_envs_once(monkeypatch):
    module = _load_layer_module(monkeypatch, {"value": None})
    module._prefill_settings.cache_clear()
    monkeypatch.setenv("HPU_KIMI_KDA_PREFILL", "Loop")
    monkeypatch.setenv("HPU_KIMI_KDA_CHUNK_SIZE", "32")
    assert module._prefill_settings() == ("loop", 32)
    monkeypatch.setenv("HPU_KIMI_KDA_CHUNK_SIZE", "64")  # cached: read once
    assert module._prefill_settings() == ("loop", 32)
    module._prefill_settings.cache_clear()
    monkeypatch.delenv("HPU_KIMI_KDA_PREFILL")
    assert module._prefill_settings() == ("chunked", 64)
