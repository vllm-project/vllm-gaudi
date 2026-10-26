# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the pure PyTorch Kimi-K3 KDA (vllm_gaudi/ops/hpu_kda_pytorch.py).

The module is loaded by file path (it has no vLLM / Habana dependencies) and compared with a naive reference
written straight from the KDA math: per head, per token loops with explicit outer products.
"""

import importlib.util
import math
import pathlib

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
                         torch.tensor([0]), mask, H, D, block=4, step_hook=lambda: calls.append(1))
    assert len(calls) == 2  # after tokens 4 and 8 (not after the last token)
