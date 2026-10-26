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


def _in_place_decode_f552b8a5(mixed_qkv, raw_g, raw_beta, conv_weight, conv_bias, A_log, dt_bias, lower_bound,
                              conv_cache, ssm_cache, state_idx, num_heads, head_dim):
    """kda_core_decode as it was before the functional split (vllm-gaudi f552b8a5), verbatim."""
    bsz, channels = mixed_qkv.shape
    width = conv_weight.shape[-1]
    slots = kda._slots(state_idx, conv_cache)
    state_c = kda.read_conv_state(conv_cache, slots, channels, width - 1)
    seq = torch.cat([state_c, mixed_qkv.float().unsqueeze(1)], dim=1)
    w = conv_weight.reshape(channels, width).float()
    conv_out = (seq * w.t().unsqueeze(0)).sum(dim=1)
    if conv_bias is not None:
        conv_out = conv_out + conv_bias.float()
    conv_out = torch.nn.functional.silu(conv_out)
    kda.write_conv_state(conv_cache, slots, seq[:, 1:, :])
    q, k, v, g, beta = kda._prepare_qkv_gates(conv_out.unsqueeze(1), raw_g.unsqueeze(1), raw_beta.unsqueeze(1), A_log,
                                              dt_bias, lower_bound, num_heads, head_dim, None)
    state = kda.read_ssm_state(ssm_cache, slots)
    out, state = kda.kda_recurrent_step(state, q[:, 0], k[:, 0], v[:, 0], torch.exp(g[:, 0]), beta[:, 0])
    kda.write_ssm_state(ssm_cache, slots, state)
    return out


@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("conv_bias", [False, True])
def test_functional_decode_step_equals_the_in_place_decode_bit_for_bit(dim_first, dtype, conv_bias):
    params = _make_params(60)
    if conv_bias:
        params["conv_bias"] = torch.randn(C, generator=torch.Generator().manual_seed(61))
    g = torch.Generator().manual_seed(62)
    old = _new_caches(6, dim_first, dtype)
    for cache in old:
        cache.copy_(torch.randn(cache.shape, generator=g).to(dtype))
    new = tuple(cache.clone() for cache in old)
    state_idx = torch.tensor([4, 1, -1, 2])  # -1: padded row -> garbage slot
    for step in range(5):
        tok = {k: v.to(dtype) for k, v in _make_inputs(70 + step, 4).items()}
        args = (tok["mixed_qkv"], tok["raw_g"], tok["raw_beta"], params["conv_weight"], params["conv_bias"],
                params["A_log"], params["dt_bias"], params["lower_bound"])
        expected = _in_place_decode_f552b8a5(*args, *old, state_idx, H, D)
        before = tuple(cache.clone() for cache in new)
        out, slots, conv_rows, ssm_rows = kda.kda_decode_step(*args, *new, state_idx, H, D)
        assert all(torch.equal(a, b) for a, b in zip(new, before))  # the step only reads the caches
        kda.write_kda_decode_state(*new, slots, conv_rows, ssm_rows)
        assert torch.equal(out, expected)
        assert all(torch.equal(a, b) for a, b in zip(new, old))
        composed = tuple(cache.clone() for cache in before)
        assert torch.equal(kda.kda_core_decode(*args, *composed, state_idx, H, D), expected)
        assert all(torch.equal(a, b) for a, b in zip(composed, old))


def _aten_ops(fn, *args):
    """aten ops executed by ``fn(*args)`` (each one is a separate eager launch outside of a compiled graph)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    ops = []

    class _Count(TorchDispatchMode):

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            ops.append(str(func.overloadpacket.__name__))
            return func(*args, **(kwargs or {}))

    with _Count():
        fn(*args)
    return ops


def test_eager_part_of_a_decode_step_is_two_cache_writes():
    params = _make_params(63)
    tok = _make_inputs(64, 2)
    caches = _new_caches(4, False)
    args = (tok["mixed_qkv"], tok["raw_g"], tok["raw_beta"], params["conv_weight"], params["conv_bias"],
            params["A_log"], params["dt_bias"], params["lower_bound"], *caches, torch.tensor([1, 2]), H, D)
    old_ops = _aten_ops(_in_place_decode_f552b8a5, *args)
    _, slots, conv_rows, ssm_rows = kda.kda_decode_step(*args)
    new_ops = _aten_ops(kda.write_kda_decode_state, *caches, slots, conv_rows, ssm_rows)
    assert new_ops == ["index_copy_", "index_copy_"]
    assert len(old_ops) > 30, old_ops  # the previously eager (compiler.disable'd) decode core


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


def _load_graph_state_module():
    spec = importlib.util.spec_from_file_location("vllm_gaudi_graph_state_under_test",
                                                  _MODULE_PATH.parents[1] / "extension" / "graph_state.py")
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

    class _ForwardContext:  # an existing object like vLLM's (dynamo cannot trace building a SimpleNamespace)

        @property
        def attn_metadata(self):
            return metadata["value"]

    context = _ForwardContext()
    forward_context.get_forward_context = lambda: context
    for name, mod in (("habana_frameworks", habana), ("habana_frameworks.torch", habana_torch),
                      ("vllm", types.ModuleType("vllm")), ("vllm.forward_context", forward_context),
                      ("vllm_gaudi", types.ModuleType("vllm_gaudi")),
                      ("vllm_gaudi.ops", types.ModuleType("vllm_gaudi.ops")), ("vllm_gaudi.ops.hpu_kda_pytorch", kda),
                      ("vllm_gaudi.extension", types.ModuleType("vllm_gaudi.extension")),
                      ("vllm_gaudi.extension.graph_state", _load_graph_state_module()),
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


def _decode_graph_check():
    """Fresh interpreter like ``_compile_check``: compiles the layer decode through aot_autograd with
    keep_inference_input_mutations=True (input mutations stay visible as copy_ on the placeholders) and returns
    the forward graphs' sizes, the in-place targets on graph inputs and whether result and caches match eager."""
    from torch._dynamo.backends.common import aot_autograd

    params = _make_params(43)
    hidden = 16
    g = torch.Generator().manual_seed(44)
    caches = tuple(cache.copy_(torch.randn(cache.shape, generator=g)) for cache in _new_caches(4, False))
    metadata = {"value": None}

    class _MonkeyPatch:

        def setitem(self, mapping, key, value):
            mapping[key] = value

    module = _load_layer_module(_MonkeyPatch(), metadata)
    metadata["value"] = types.SimpleNamespace(is_prompt=False, load_indices_tensor=torch.tensor([[0, 0], [1, 3]]))
    hidden_states = torch.randn(2, hidden, generator=g)
    eager_layer = _FakeLayer(params, hidden, 45, tuple(cache.clone() for cache in caches))
    expected = module.hpu_kda_forward(eager_layer, hidden_states)

    graphs, mutated = [], []

    def fw_compiler(gm, example_inputs):
        placeholders = {n for n in gm.graph.nodes if n.op == "placeholder"}
        graphs.append(sum(n.op == "call_function" for n in gm.graph.nodes))
        for node in gm.graph.nodes:
            schema = getattr(node.target, "_schema", None)
            if node.op == "call_function" and schema is not None and schema.is_mutable and node.args \
                    and node.args[0] in placeholders:
                mutated.append(str(node.target))
        return gm.forward

    layer = _FakeLayer(params, hidden, 45, caches)
    backend = aot_autograd(fw_compiler=fw_compiler, keep_inference_input_mutations=True)
    with torch.no_grad():
        out = torch.compile(module.hpu_kda_forward, backend=backend, dynamic=False)(layer, hidden_states)
    return dict(graph_ops=graphs,
                mutated=mutated,
                out_equal=bool(torch.allclose(out, expected, rtol=1e-6, atol=1e-6)),
                caches_equal=all(bool(torch.equal(a, b)) for a, b in zip(caches, eager_layer.kv_cache)))


def test_compiled_layer_decode_traces_the_step_and_mutates_no_graph_input():
    """The decode step is one compiled graph with the projections, the cache writes are the only eager part (two
    index_copy_ in ``_write_decode_state``), followed by the resume graph (norm, o_proj). No graph writes into its
    inputs, so HPU compile cannot drop a state update, and the caches end up as in eager mode."""
    import json

    pytest.importorskip("torch._dynamo")
    result = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "decode_graphs"],
                            capture_output=True,
                            text=True,
                            timeout=600)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr[-2000:]}"
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    assert stats["mutated"] == []
    assert stats["out_equal"] and stats["caches_equal"]
    assert len(stats["graph_ops"]) == 2, stats  # step graph | 2 x index_copy_ (eager) | resume graph
    assert stats["graph_ops"][0] > 30, stats  # the decode math is inside the first graph


if __name__ == "__main__":
    if sys.argv[1] == "decode_graphs":
        import json
        print(json.dumps(_decode_graph_check()))
        sys.exit(0)
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
    assert len(calls) == 4  # after the whole-sequence preparation, then after chunks 3, 6 and 9 of 10


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
    assert module._prefill_settings() == ("loop", 32, 1)
    monkeypatch.setenv("HPU_KIMI_KDA_CHUNK_SIZE", "64")  # cached: read once
    assert module._prefill_settings() == ("loop", 32, 1)
    module._prefill_settings.cache_clear()
    monkeypatch.delenv("HPU_KIMI_KDA_PREFILL")
    monkeypatch.setenv("HPU_KIMI_KDA_MARK_STEP_CHUNKS", "4")
    assert module._prefill_settings() == ("chunked", 64, 4)


# ---------------------------------------------------------------------------------------------------------------
# attn_res: same results AND side effects as the upstream Triton kernel
# ---------------------------------------------------------------------------------------------------------------
def reference_attn_res_kernel(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, num_blocks,
                              block_write_idx, eps, output_norm_eps):
    """Row-by-row transcription of vllm/models/kimi_k3/amd/ops/attn_res.py::_attn_res_kernel (side effects included)."""
    tokens, hidden = prefix.shape
    output = prefix.new_empty(prefix.shape)
    input_qk_weight = norm_weight.float() * qk_weight.float()
    for row in range(tokens):
        updated = prefix[row].float()
        if delta is not None:
            updated = updated + delta[row].float()
            updated = updated.to(prefix.dtype).float()  # "match the BF16 prefix-add result"
            prefix[row] = updated.to(prefix.dtype)  # tl.store(prefix_ptr, ...)
        if block_write_idx >= 0:
            blocks[row, block_write_idx] = updated.to(blocks.dtype)  # WRITE_BLOCK
        if num_blocks == 0:
            mixed = updated
        else:
            sources = [blocks[row, r].float() for r in range(num_blocks)] + [updated]
            logits = []
            for value in sources:
                rstd = torch.rsqrt((value * value).sum() / hidden + eps)
                logits.append((value * input_qk_weight).sum() * rstd)
            weights = torch.softmax(torch.stack(logits), dim=0)
            mixed = sum(w * v for w, v in zip(weights, sources))
        out = mixed
        if output_norm_weight is not None:
            out = mixed * torch.rsqrt((mixed * mixed).sum() / hidden + output_norm_eps) * output_norm_weight.float()
        output[row] = out.to(prefix.dtype)
    return output


def _attn_res_case(seed, dtype, tokens=5, hidden=16, bank=4):
    g = torch.Generator().manual_seed(seed)
    prefix = torch.randn(tokens, hidden, generator=g).to(dtype)
    delta = torch.randn(tokens, hidden, generator=g).to(dtype)
    blocks = torch.randn(tokens, bank, hidden, generator=g).to(dtype)
    norm_weight = torch.rand(hidden, generator=g) + 0.5
    qk_weight = torch.randn(hidden, generator=g)
    output_norm_weight = torch.rand(hidden, generator=g) + 0.5
    return prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_delta", [True, False])
@pytest.mark.parametrize("num_blocks,block_write_idx", [(0, -1), (0, 0), (2, -1), (2, 2), (3, 1)])
@pytest.mark.parametrize("output_norm", [True, False])
def test_attn_res_torch_matches_kernel_results_and_side_effects(dtype, has_delta, num_blocks, block_write_idx,
                                                                output_norm):
    prefix, delta, blocks, norm_w, qk_w, out_w = _attn_res_case(120 + num_blocks, dtype)
    delta = delta if has_delta else None
    out_w = out_w if output_norm else None
    ref_prefix, ref_blocks = prefix.clone(), blocks.clone()
    ref_out = reference_attn_res_kernel(ref_prefix, delta, ref_blocks, norm_w, qk_w, out_w, num_blocks,
                                        block_write_idx, 1e-6, 1e-6)

    out = kda.attn_res_torch(prefix, delta, blocks, norm_w, qk_w, out_w, num_blocks, block_write_idx, 1e-6, 1e-6)

    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=3e-2, atol=3e-2)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref_out, **tol)
    # the side effects must be identical, not merely close: they feed every later layer
    assert torch.equal(prefix, ref_prefix)
    assert torch.equal(blocks, ref_blocks)
    if has_delta:
        assert not torch.equal(prefix, _attn_res_case(120 + num_blocks, dtype)[0])  # really updated in place


def test_attn_res_torch_updates_a_3d_prefix_in_place_and_rejects_mismatched_banks():
    prefix, delta, blocks, norm_w, qk_w, out_w = _attn_res_case(130, torch.float32, tokens=6)
    expected = (prefix + delta).clone()
    kda.attn_res_torch(prefix.view(2, 3, -1), delta.view(2, 3, -1), blocks, norm_w, qk_w, out_w, 2, -1, 1e-6, 1e-6)
    torch.testing.assert_close(prefix, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="block bank"):
        kda.attn_res_torch(prefix, None, blocks[:3], norm_w, qk_w, None, 1, -1, 1e-6, 1e-6)


def _toy_params(num_layers, hidden, seed):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *shape: torch.rand(*shape, generator=g) + 0.5  # noqa: E731
    layers = []
    for _ in range(num_layers):
        layers.append(
            dict(attn_norm=mk(hidden),
                 attn_qk=torch.randn(hidden, generator=g),
                 mlp_norm=mk(hidden),
                 mlp_qk=torch.randn(hidden, generator=g),
                 ln1=mk(hidden),
                 ln2=mk(hidden),
                 w_attn=torch.randn(hidden, hidden, generator=g) * 0.3,
                 w_mlp=torch.randn(hidden, hidden, generator=g) * 0.3))
    return g, mk, layers


def _toy_residual_stream(attn_res_fn, num_layers=14, tokens=5, hidden=16, block_size=12, seed=140):
    """The Kimi-K3 forward_attn_residual / KimiLinearModel.forward control flow with toy sublayers."""
    g, mk, layers = _toy_params(num_layers, hidden, seed)
    final_norm, final_qk = mk(hidden), torch.randn(hidden, generator=g)
    embeddings = torch.randn(tokens, hidden, generator=g)
    num_bank = -(-num_layers // block_size)
    blocks = embeddings.new_empty(tokens, num_bank, hidden)
    hidden_states, prefix_delta = embeddings.clone(), None
    for idx, p in enumerate(layers):
        is_write = idx % block_size == 0
        prev_valid = -(-idx // block_size)
        prefix_sum = hidden_states
        h = attn_res_fn(prefix_sum, prefix_delta, blocks, p["attn_norm"], p["attn_qk"], p["ln1"], prev_valid,
                        idx // block_size if is_write else -1, 1e-6, 1e-6)
        if is_write:
            prefix_sum = None
        h = torch.tanh(h @ p["w_attn"])
        if prefix_sum is None:
            prefix_sum, prefix_delta = h, None
        else:
            prefix_delta = h
        h = attn_res_fn(prefix_sum, prefix_delta, blocks, p["mlp_norm"], p["mlp_qk"], p["ln2"],
                        prev_valid + (1 if is_write else 0), -1, 1e-6, 1e-6)
        h = torch.tanh(h @ p["w_mlp"])
        hidden_states, prefix_delta = prefix_sum, h
    final = attn_res_fn(hidden_states, prefix_delta, blocks, final_norm, final_qk, None, num_bank, -1, 1e-6, 0.0)
    return final, hidden_states, blocks


def test_residual_stream_accumulates_across_layers_like_the_kernel():
    final, last_prefix, bank = _toy_residual_stream(kda.attn_res_torch)
    ref_final, ref_prefix, ref_bank = _toy_residual_stream(reference_attn_res_kernel)
    torch.testing.assert_close(final, ref_final, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(last_prefix, ref_prefix, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(bank, ref_bank, rtol=1e-4, atol=1e-4)


def test_residual_stream_test_detects_the_old_out_of_place_behaviour():
    def out_of_place(prefix, delta, *args):
        # what the previous fallback did: add the delta to a local copy only
        if delta is not None:
            prefix = prefix + delta
        return kda.attn_res_torch(prefix, None, *args)

    broken_final, _, _ = _toy_residual_stream(out_of_place)
    ref_final, _, _ = _toy_residual_stream(reference_attn_res_kernel)
    assert not torch.allclose(broken_final, ref_final, rtol=1e-2, atol=1e-2)


def test_default_is_one_lazy_graph_per_chunk():
    q, k, v, log_gate, beta, state = _random_delta_inputs(150, 1, 64 * 5)
    calls = []
    kda.kda_chunked_steps(state, q, k, v, log_gate, beta, chunk_size=64, block_chunks=1,
                          step_hook=lambda: calls.append(1))
    assert len(calls) == 5  # preparation + after each of the first four chunks (not after the last one)


def test_inverse_has_no_in_place_writes():
    """In-place writes into views are what the lazy HPU mode handles with strided-insert ops and aliasing."""
    from torch.utils._python_dispatch import TorchDispatchMode

    names = []

    class Recorder(TorchDispatchMode):

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            names.append(func.__name__)
            return func(*args, **(kwargs or {}))

    a = torch.tril(torch.rand(1, 3, 64, 64) * 0.5, -1)
    with Recorder():
        inv = kda._unit_lower_inverse(a)
    assert not [n for n in names if n.endswith("_") or n in ("copy_", "slice_scatter", "select_scatter")]
    torch.testing.assert_close(inv, torch.linalg.inv(torch.eye(64) + a), rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------------------------------------------
# Out-of-place attn_res for torch.compile mode (regional compilation: one graph per decoder layer)
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("has_delta", [True, False])
@pytest.mark.parametrize("num_blocks,block_write_idx", [(0, -1), (0, 0), (2, -1), (2, 2), (3, 1)])
def test_attn_res_functional_matches_in_place_version_and_modifies_nothing(has_delta, num_blocks, block_write_idx):
    prefix, delta, blocks, norm_w, qk_w, out_w = _attn_res_case(160 + num_blocks, torch.bfloat16)
    delta = delta if has_delta else None
    inputs = (prefix.clone(), blocks.clone())
    new_prefix, new_blocks, out = kda.attn_res_functional(prefix, delta, blocks, norm_w, qk_w, out_w, num_blocks,
                                                          block_write_idx, 1e-6, 1e-6)
    assert torch.equal(prefix, inputs[0]) and torch.equal(blocks, inputs[1])
    in_place_out = kda.attn_res_torch(prefix, delta, blocks, norm_w, qk_w, out_w, num_blocks, block_write_idx, 1e-6,
                                      1e-6)
    assert torch.equal(out, in_place_out)
    assert torch.equal(new_prefix, prefix) and torch.equal(new_blocks, blocks)


class _ToyLayer(torch.nn.Module):
    """A KimiLinearDecoderLayer stand-in around the real ``forward_attn_residual_functional``."""

    def __init__(self, idx, block_size, p):
        super().__init__()
        ns = lambda w, qk=None: types.SimpleNamespace(weight=w, variance_epsilon=1e-6)  # noqa: E731
        self.self_attention_res_proj = types.SimpleNamespace(weight=p["attn_qk"].unsqueeze(0))
        self.self_attention_res_norm = ns(p["attn_norm"])
        self.mlp_res_proj = types.SimpleNamespace(weight=p["mlp_qk"].unsqueeze(0))
        self.mlp_res_norm = ns(p["mlp_norm"])
        self.input_layernorm = ns(p["ln1"])
        self.post_attention_layernorm = ns(p["ln2"])
        self.is_block_write_layer = idx % block_size == 0
        self.block_write_idx = idx // block_size
        self.prev_valid_blocks = -(-idx // block_size)
        self.w_attn, self.w_mlp = p["w_attn"], p["w_mlp"]

    def _run_self_attn(self, positions, h):
        return _opaque_tanh_matmul(h, self.w_attn)  # a torch.compiler.disable'd call: forces a graph break

    def mlp(self, h):
        return torch.tanh(h @ self.w_mlp)

    def forward(self, positions, hidden_states, block_residual, prefix_delta):
        return kda.forward_attn_residual_functional(self, positions, hidden_states, block_residual, prefix_delta)


@torch.compiler.disable
def _opaque_tanh_matmul(h, w):
    return torch.tanh(h @ w)


def _run_toy_layers(wrap, num_layers=14, block_size=4, tokens=5, hidden=16, seed=140):
    g, mk, params = _toy_params(num_layers, hidden, seed)
    mk(hidden), torch.randn(hidden, generator=g)  # the final norm / qk of _toy_residual_stream: same generator state
    hidden_states = torch.randn(tokens, hidden, generator=g)
    num_bank = -(-num_layers // block_size)
    blocks = hidden_states.new_empty(tokens, num_bank, hidden)
    prefix_delta = None
    layers = [wrap(_ToyLayer(i, block_size, p)) for i, p in enumerate(params)]
    for layer in layers:
        hidden_states, blocks, prefix_delta = layer(None, hidden_states, blocks, prefix_delta)
    return hidden_states, blocks, prefix_delta


def test_functional_layers_match_the_in_place_upstream_flow():
    prefix, bank, delta = _run_toy_layers(lambda layer: layer)
    ref_final, ref_prefix, ref_bank = _toy_residual_stream(reference_attn_res_kernel, num_layers=14, block_size=4,
                                                           seed=140)
    # the toy stream's final attn_res adds the last mlp output into the prefix in place
    torch.testing.assert_close(prefix + delta, ref_prefix, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(bank, ref_bank, rtol=1e-4, atol=1e-4)


def test_per_layer_compiled_functional_attn_res_equals_eager_across_graph_breaks():
    """Regional compilation: each layer is its own graph, the disabled attention call splits it again."""
    import torch._dynamo as dynamo
    dynamo.reset()
    try:
        expected = _run_toy_layers(lambda layer: layer)
        got = _run_toy_layers(lambda layer: torch.compile(layer, backend="aot_eager"))
    finally:
        dynamo.reset()
    for a, b in zip(got, expected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def _mutated_placeholders(wrap_layer, inputs):
    """Names of graph inputs that the forward graphs of ``wrap_layer(inputs)`` write to in place."""
    from torch._dynamo.backends.common import aot_autograd
    import torch._dynamo as dynamo

    mutated = []

    def fw_compiler(gm, example_inputs):
        placeholders = {n for n in gm.graph.nodes if n.op == "placeholder"}
        for node in gm.graph.nodes:
            if node.op == "call_function" and getattr(node.target, "_schema", None) is not None \
                    and node.target._schema.is_mutable and node.args and node.args[0] in placeholders:
                mutated.append(str(node.target))
        return gm.forward

    backend = aot_autograd(fw_compiler=fw_compiler, keep_inference_input_mutations=True)
    dynamo.reset()
    try:
        with torch.no_grad():
            wrap_layer(backend)(*inputs)
    finally:
        dynamo.reset()
    return mutated


def test_compiled_functional_layer_has_no_input_mutation():
    _, _, params = _toy_params(2, 16, 140)
    g = torch.Generator().manual_seed(1)
    hidden_states, blocks = torch.randn(5, 16, generator=g), torch.randn(5, 3, 16, generator=g)
    for idx in (0, 1):  # a block-write layer and a plain one
        layer = _ToyLayer(idx, 4, params[idx])
        delta = None if idx == 0 else torch.randn(5, 16, generator=g)
        mutated = _mutated_placeholders(lambda backend, layer=layer: torch.compile(layer, backend=backend),
                                        (None, hidden_states, blocks, delta))
        assert mutated == []


def test_mutation_probe_detects_the_in_place_attn_res():
    """Positive control: the in-place variant's graph writes into its inputs (copy_ on the placeholders)."""
    prefix, delta, blocks, norm_w, qk_w, out_w = _attn_res_case(170, torch.float32)

    def wrap(backend):
        return torch.compile(lambda *a: kda.attn_res_torch(*a, 2, 1, 1e-6, 1e-6), backend=backend)

    mutated = _mutated_placeholders(wrap, (prefix, delta, blocks, norm_w, qk_w, out_w))
    assert mutated


def _upstream_linear_module(class_name="KimiDecoderLayer"):
    """A ``vllm.models.kimi_k3.amd.linear`` stand-in: forward -> self.forward_attn_residual (in place, upstream)."""

    def _apply_attn_res(prefix_sum,
                        block_residual,
                        proj,
                        norm,
                        num_valid_blocks,
                        delta=None,
                        output_norm=None,
                        block_write_idx=-1):  # what patches.py binds upstream's _apply_attn_res to
        return kda.attn_res_torch(prefix_sum, delta, block_residual, norm.weight, proj.weight.squeeze(0),
                                  None if output_norm is None else output_norm.weight, num_valid_blocks,
                                  block_write_idx, norm.variance_epsilon,
                                  0.0 if output_norm is None else output_norm.variance_epsilon)

    class Layer(_ToyLayer):
        use_attn_residuals = True

        def forward(self, positions, hidden_states, residual, prefix_delta=None):
            return self.forward_attn_residual(positions, hidden_states, residual, prefix_delta)

        def forward_attn_residual(self, positions, hidden_states, block_residual, prefix_delta):
            prefix_sum = hidden_states
            hidden_states = _apply_attn_res(prefix_sum, block_residual, self.self_attention_res_proj,
                                            self.self_attention_res_norm, self.prev_valid_blocks, prefix_delta,
                                            self.input_layernorm,
                                            self.block_write_idx if self.is_block_write_layer else -1)
            if self.is_block_write_layer:
                prefix_sum = None
            hidden_states = self._run_self_attn(positions, hidden_states)
            if prefix_sum is None:
                prefix_sum, prefix_delta = hidden_states, None
            else:
                prefix_delta = hidden_states
            hidden_states = _apply_attn_res(prefix_sum, block_residual, self.mlp_res_proj, self.mlp_res_norm,
                                            self.prev_valid_blocks + (1 if self.is_block_write_layer else 0),
                                            prefix_delta, self.post_attention_layernorm)
            return prefix_sum, block_residual, self.mlp(hidden_states)

    mod = types.ModuleType("vllm.models.kimi_k3.amd.linear")
    setattr(mod, class_name, Layer)
    return mod, Layer


@pytest.mark.parametrize("class_name", kda.KIMI_DECODER_LAYER_CLASSES)
def test_install_patches_the_decoder_layer_the_model_calls_and_removes_input_mutations(class_name):
    mod, cls = _upstream_linear_module(class_name)
    _, _, params = _toy_params(2, 16, 140)
    g = torch.Generator().manual_seed(1)
    hidden_states, blocks = torch.randn(5, 16, generator=g), torch.randn(5, 3, 16, generator=g)
    delta = torch.randn(5, 16, generator=g)
    layer = cls(1, 4, params[1])

    def probe():
        return _mutated_placeholders(lambda backend: torch.compile(layer, backend=backend),
                                     (None, hidden_states.clone(), blocks.clone(), delta))

    assert probe()  # upstream in-place attn_res: copy_ into the graph inputs
    assert kda.install_functional_attn_res(mod) is cls
    patched = cls.forward_attn_residual
    assert kda.install_functional_attn_res(mod) is cls and cls.forward_attn_residual is patched  # idempotent
    assert probe() == []
    expected = kda.forward_attn_residual_functional(layer, None, hidden_states, blocks, delta)
    for a, b in zip(layer(None, hidden_states, blocks, delta), expected):
        torch.testing.assert_close(a, b)


def test_install_reports_a_missing_decoder_layer():
    assert kda.install_functional_attn_res(types.ModuleType("empty")) is None
    renamed = types.ModuleType("renamed")
    renamed.SomeDecoderLayer = _upstream_linear_module()[1]
    assert kda.install_functional_attn_res(renamed) is None


def test_attn_res_layer_variants_of_kimi_k3():
    layers = [
        types.SimpleNamespace(use_attn_residuals=True,
                              is_block_write_layer=i % 12 == 0,
                              block_write_idx=i // 12,
                              prev_valid_blocks=-(-i // 12)) for i in range(93)
    ]
    assert kda.attn_res_layer_variants(layers + [torch.nn.Linear(2, 2)]) == 16
    assert kda.attn_res_layer_variants([torch.nn.Linear(2, 2)]) == 1
