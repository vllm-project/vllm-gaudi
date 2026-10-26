# SPDX-License-Identifier: Apache-2.0
"""Exact SiTU routed-expert MLP for Kimi-K3 on HPU (pure PyTorch, no vLLM / Habana imports).

Kimi-K3 uses ``hidden_act = "situ"`` (SituGLU)::

    act(gate, up) = beta * tanh(gate / beta) * sigmoid(gate) * up                    (linear_beta unset)
    act(gate, up) = beta * tanh(gate / beta) * sigmoid(gate) * linear_beta * tanh(up / linear_beta)

(``beta = activation_situ_beta`` = 4.0, ``linear_beta = activation_situ_linear_beta`` = 25.0 in the checkpoint
config; vllm.model_executor.layers.activation.SituAndMul.forward_native). The Habana fused
``torch.ops.hpu.mixture_of_experts`` op only implements silu / gelu, and mapping "situ" to "silu" is not
equivalent (it removes the soft clipping of the gate and of ``up``), so the routed experts are computed unfused:
the packed MXFP4 weights of the selected experts are dequantized to bf16 chunk by chunk and run through batched
matmuls with static shapes, like ``_unfused_swigluoai_moe`` (ops/hpu_fused_moe.py) does for MiniMax-M3.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable

import torch

# E2M1 magnitudes; the high bit of a nibble is the sign. Low nibble = even element, high nibble = odd element.
_FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
MXFP4_BLOCK = 32


def resolve_situ_impl(value: str | None) -> str:
    """HPU_KIMI_SITU_IMPL: "exact" (default) or "silu" (approximate, fused kernel). Raises on anything else."""
    impl = (value or "exact").strip().lower()
    if impl not in ("exact", "silu"):
        raise ValueError(f"HPU_KIMI_SITU_IMPL must be 'exact' or 'silu', got {value!r}")
    return impl


def situ_impl_or_exact(value: str | None, log_error: Callable[[str], None] | None = None) -> str:
    """Like ``resolve_situ_impl`` but an invalid value is reported through ``log_error`` and replaced by "exact"."""
    try:
        return resolve_situ_impl(value)
    except ValueError as exc:
        if log_error is not None:
            log_error(f"{exc}; using the exact SiTU path")
        return "exact"


def situ_fused_alias() -> str | None:
    """Activation the fused Habana MoE op must use for "situ", read from HPU_KIMI_SITU_IMPL at call time.

    "silu" only with HPU_KIMI_SITU_IMPL=silu (approximate); None otherwise, so that a "situ" reaching the fused
    op in "exact" mode fails loudly. Evaluated lazily (no import-time work) so that it cannot hit the circular
    imports of the vLLM plugin load.
    """
    import vllm_gaudi.envs as gaudi_envs
    return "silu" if situ_impl_or_exact(gaudi_envs.HPU_KIMI_SITU_IMPL) == "silu" else None


def situ_activation(gate_up: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """SituAndMul.forward_native on a fused [..., 2 * d] tensor (first half = gate, second half = up)."""
    d = gate_up.shape[-1] // 2
    gate = gate_up[..., :d].float()
    up = gate_up[..., d:].float()
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(gate_up.dtype)


_LUT_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _luts(dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(byte -> [low, high] E2M1 values [256, 2], E8M0 byte -> 2^(byte - 127) [256]) in ``dtype``.

    E2M1 values and powers of two are exact in bf16, so scaling in bf16 is exact (no fp32 round trip needed).
    """
    key = (dtype, str(device))
    if key not in _LUT_CACHE:
        values = torch.tensor(_FP4_VALUES, dtype=torch.float32)
        codes = torch.arange(256)
        byte_lut = torch.stack((values[codes & 0x0F], values[codes >> 4]), dim=-1)
        exp2_lut = torch.ldexp(torch.ones(256), torch.arange(256, dtype=torch.int32) - 127)
        _LUT_CACHE[key] = (byte_lut.to(dtype=dtype, device=device), exp2_lut.to(dtype=dtype, device=device))
    return _LUT_CACHE[key]


def dequant_mxfp4(packed: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """packed uint8 [..., rows, cols // 2] + E8M0 scales uint8 [..., rows, cols // 32] -> [..., rows, cols].

    Two table lookups (a 256 x 2 byte table gives both E2M1 values of a byte, a 256-entry table the E8M0 power of
    two) and one multiply, all directly in ``dtype``; int32 indices keep the temporaries small.
    """
    *lead, rows, half = packed.shape
    groups = scales.shape[-1]
    if half * 2 != groups * MXFP4_BLOCK or scales.shape[:-1] != packed.shape[:-1]:
        raise ValueError(f"packed {tuple(packed.shape)} and scales {tuple(scales.shape)} do not match")
    byte_lut, exp2_lut = _luts(dtype, packed.device)
    values = torch.nn.functional.embedding(packed.to(torch.int32), byte_lut)  # [..., rows, cols / 2, 2]
    values = values.reshape(*lead, rows, groups, MXFP4_BLOCK)
    factor = torch.nn.functional.embedding(scales.to(torch.int32), exp2_lut.unsqueeze(-1))  # [..., rows, groups, 1]
    values.mul_(factor)  # in place: no second weight-sized temporary (the table lookup result is ours)
    return values.reshape(*lead, rows, groups * MXFP4_BLOCK)


def dequant_mxfp4_reference(packed: torch.Tensor,
                            scales: torch.Tensor,
                            dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Straightforward fp32 stack + ldexp version (as in models/gptoss_mxfp4.py); the LUT version must equal it."""
    *lead, rows, half = packed.shape
    groups = scales.shape[-1]
    lut = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    values = torch.stack((lut[(packed & 0x0F).long()], lut[(packed >> 4).long()]), dim=-1)
    values = values.reshape(*lead, rows, groups, MXFP4_BLOCK)
    values = torch.ldexp(values, (scales.to(torch.int32) - 127).unsqueeze(-1))
    return values.reshape(*lead, rows, groups * MXFP4_BLOCK).to(dtype)


BITS_MIN_SCALE, BITS_MAX_SCALE = 2, 252  # E8M0 bytes for which the bit assembly is exact


def dequant_mxfp4_bits(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Gather-free bf16 dequant: the bf16 bit pattern of every value is assembled with int16 elementwise ops.

    E2M1 nibble ``s e1 e0 m``: for ``e > 0`` the value is ``(1 + m / 2) * 2^(e - 1)``, so with the E8M0 scale byte
    ``S`` the bf16 exponent field is ``e - 1 + S`` and the mantissa ``m << 6``; for ``e == 0`` it is ``0.5 * m``,
    i.e. exponent field ``S - 1`` when ``m == 1`` and a (signed) zero otherwise. No table lookups (TPC gathers) and
    no int32 temporaries; equal to ``dequant_mxfp4_reference`` bit for bit for scale bytes 2 .. 252. Scale bytes
    outside that range would leave the normal bf16 range (wrong subnormals, NaN, int16 overflow into the sign bit),
    so they are clamped to [2, 252] (no host sync); real checkpoints use scales around 2^-20 .. 2^8.
    """
    *lead, rows, half = packed.shape
    groups = scales.shape[-1]
    if half * 2 != groups * MXFP4_BLOCK or scales.shape[:-1] != packed.shape[:-1]:
        raise ValueError(f"packed {tuple(packed.shape)} and scales {tuple(scales.shape)} do not match")
    p = packed.to(torch.int16)
    nib = torch.stack((p & 0x0F, p >> 4), dim=-1).reshape(*lead, rows, groups, MXFP4_BLOCK)
    s = scales.to(torch.int16).clamp(BITS_MIN_SCALE, BITS_MAX_SCALE).unsqueeze(-1)  # [..., rows, groups, 1]
    e = (nib >> 1) & 3
    m = nib & 1
    normal = e > 0
    exponent = torch.where(normal, e + s - 1, m * (s - 1))
    bits = (exponent << 7) | ((m * normal) << 6)
    bits = torch.where(nib >= 8, bits | -0x8000, bits)  # sign bit (0x8000 as int16)
    return bits.view(torch.bfloat16).reshape(*lead, rows, groups * MXFP4_BLOCK)


def situ_moe_experts(x: torch.Tensor,
                     topk_ids: torch.Tensor,
                     topk_weights: torch.Tensor,
                     get_w13: Callable[[torch.Tensor], torch.Tensor],
                     get_w2: Callable[[torch.Tensor], torch.Tensor],
                     num_local_experts: int,
                     experts_min: int,
                     beta: float,
                     linear_beta: float | None,
                     expert_chunk: int = 8,
                     token_tile: int = 512,
                     gather_max_tokens: int | None = None,
                     step_hook: Callable[[], None] | None = None,
                     step_min_tokens: int = 0,
                     phase_hook: Callable[[str], None] | None = None) -> torch.Tensor:
    """Per-rank partial routed-expert output (before the cross-rank reduction), same contract as the fused op.

    x [T, H]; topk_ids / topk_weights [T, K] with GLOBAL expert ids; this rank owns the experts
    ``experts_min .. experts_min + num_local_experts - 1``. ``get_w13(ids)`` returns the dequantized fused
    gate/up weights ``[c, 2 * I, H]`` and ``get_w2(ids)`` the down weights ``[c, H, I]`` of the local experts
    ``ids`` (a LongTensor). Shapes are static (HPU graphs): few tokens (decode) compute only the
    ``G = T * K`` experts that can be hit whenever that is fewer than the local experts (optionally only up to
    ``gather_max_tokens`` tokens), otherwise all local experts are computed densely and masked by the routing
    weight. Everything is accumulated in fp32. ``step_hook`` (``htorch.core.mark_step`` in lazy mode) runs after
    every expert chunk when more than ``step_min_tokens`` tokens are processed. The caller passes None whenever
    the forward runs under HPU graphs (a mark_step would break the capture), see extension/graph_state.py.
    ``phase_hook(name)`` (diagnostics, see ``PhaseTimer``) runs at the end of every phase: "route", then per chunk
    "dequant_w13", "dequant_w2" and "experts".
    """
    mark = phase_hook if phase_hook is not None else (lambda name: None)
    tokens, hidden = x.shape
    k = topk_ids.shape[-1]
    local = topk_ids.long() - experts_min
    in_range = (local >= 0) & (local < num_local_experts)
    safe = torch.where(in_range, local, torch.zeros_like(local))
    gate_w = torch.zeros(tokens, num_local_experts, dtype=torch.float32, device=x.device)
    gate_w.scatter_add_(1, safe, torch.where(in_range, topk_weights.float(), torch.zeros_like(topk_weights.float())))

    count = tokens * k
    if count < num_local_experts and (gather_max_tokens is None or tokens <= gather_max_tokens):
        # at most T * K distinct experts can be hit: dequantize and run only those (static count)
        hit = (gate_w > 0).float().sum(0)
        ids = torch.sort(torch.topk(hit, count, sorted=False).indices).values
    else:
        ids = torch.arange(num_local_experts, device=x.device)
    total = ids.shape[0]

    tile = tokens if token_tile <= 0 or token_tile >= tokens else token_tile
    starts = list(range(0, tokens, tile))
    accs = [torch.zeros(min(tile, tokens - t0), hidden, dtype=torch.float32, device=x.device) for t0 in starts]
    # Expert chunks are the OUTER loop: the weights of a chunk are dequantized once and reused by every token tile
    # (tile-outer order repeated the whole dequant per tile: 16x the ops for an 8k prefill).
    step = step_hook if step_hook is not None and tokens > step_min_tokens else None
    mark("route")
    for c0 in range(0, total, expert_chunk):
        chunk_ids = ids[c0:c0 + expert_chunk]
        w13 = get_w13(chunk_ids)  # [c, 2I, H]
        mark("dequant_w13")
        w2 = get_w2(chunk_ids)  # [c, H, I]
        mark("dequant_w2")
        for i, t0 in enumerate(starts):
            xt = x[t0:t0 + tile]
            xe = xt.unsqueeze(0).expand(chunk_ids.shape[0], -1, -1)
            hidden_act = situ_activation(torch.bmm(xe, w13.transpose(1, 2)), beta, linear_beta)  # [c, t, I]
            y = torch.bmm(hidden_act, w2.transpose(1, 2))  # [c, t, H]
            weights = gate_w[t0:t0 + tile].index_select(1, chunk_ids).t().unsqueeze(-1)  # [c, t, 1]
            accs[i] = accs[i] + (y.float() * weights).sum(0)
        mark("experts")
        if step is not None:
            step()  # bound the lazy graph: one identical, cacheable graph per chunk
    outputs = accs
    out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
    return out.to(x.dtype)


DEQUANT_MODES = ("lut", "bits")


def _read_dequant_env() -> str:
    """HPU_KIMI_SITU_DEQUANT, validated once at import (envs.py documents it; this module stays importable without
    vLLM, hence os.environ)."""
    value = os.environ.get("HPU_KIMI_SITU_DEQUANT", "lut")
    mode = (value or "lut").strip().lower()
    if mode not in DEQUANT_MODES:
        raise ValueError(f"HPU_KIMI_SITU_DEQUANT must be one of {DEQUANT_MODES}, got {value!r}")
    return mode


def _read_timing_budget() -> int:
    """HPU_KIMI_SITU_TIMING=N (0 = off), validated once at import."""
    value = os.environ.get("HPU_KIMI_SITU_TIMING", "0")
    try:
        budget = int(value)
    except ValueError:
        budget = -1
    if budget < 0:
        raise ValueError(f"HPU_KIMI_SITU_TIMING must be a non-negative integer, got {value!r}")
    return budget


DEQUANT_MODE_ENV = _read_dequant_env()
TIMING_BUDGET = _read_timing_budget()


def resolve_dequant_mode(value: str | None, dtype: torch.dtype) -> str:
    """HPU_KIMI_SITU_DEQUANT: "lut" (default, table lookups) or "bits" (gather-free, bf16 only)."""
    mode = (value or "lut").strip().lower()
    if mode not in DEQUANT_MODES:
        raise ValueError(f"HPU_KIMI_SITU_DEQUANT must be one of {DEQUANT_MODES}, got {value!r}")
    return "lut" if dtype != torch.bfloat16 else mode


def situ_moe_mxfp4(x: torch.Tensor,
                   topk_ids: torch.Tensor,
                   topk_weights: torch.Tensor,
                   w13: torch.Tensor,
                   w13_scale: torch.Tensor,
                   w2: torch.Tensor,
                   w2_scale: torch.Tensor,
                   experts_min: int,
                   beta: float,
                   linear_beta: float | None,
                   dequant: str = "lut",
                   **kwargs) -> torch.Tensor:
    """``situ_moe_experts`` over the stacked packed MXFP4 expert weights of this rank (dequantized per chunk)."""
    if resolve_dequant_mode(dequant, x.dtype) == "bits":
        deq = lambda packed, scales: dequant_mxfp4_bits(packed, scales)  # noqa: E731
    else:
        deq = lambda packed, scales: dequant_mxfp4(packed, scales, x.dtype)  # noqa: E731

    def get_w13(ids):
        return deq(w13.index_select(0, ids), w13_scale.index_select(0, ids))

    def get_w2(ids):
        return deq(w2.index_select(0, ids), w2_scale.index_select(0, ids))

    return situ_moe_experts(x, topk_ids, topk_weights, get_w13, get_w2, w13.shape[0], experts_min, beta, linear_beta,
                            **kwargs)


class PhaseTimer:
    """Wall time per phase of ``situ_moe_experts`` (``phase_hook``); ``sync`` waits for the device at every mark."""

    def __init__(self, sync: Callable[[], None]):
        self.sync = sync
        self.totals: dict[str, float] = {}
        self.sync()  # do not charge the work queued before the experts to the first phase
        self._last = time.perf_counter()

    def __call__(self, name: str) -> None:
        self.sync()
        now = time.perf_counter()
        self.totals[name] = self.totals.get(name, 0.0) + now - self._last
        self._last = now

    def summary(self) -> str:
        total = sum(self.totals.values())
        parts = " ".join(f"{name}={sec * 1e3:.1f}ms" for name, sec in self.totals.items())
        return f"{parts} total={total * 1e3:.1f}ms"


_timed_calls = 0  # plain int: the compiled path checks it before entering the disabled timed path


@torch.compiler.disable
def _timed_situ_moe_mxfp4(budget: int, *args, **kwargs) -> torch.Tensor:
    """``situ_moe_mxfp4`` with per-phase device timings, run eagerly; logged via the vllm.gaudi.situ logger."""
    global _timed_calls
    from vllm_gaudi.extension.graph_state import device_sync_hook
    sync = device_sync_hook() if _timed_calls < budget else None
    if sync is None:  # budget used up, or lazy mode under HPU graphs (no mark_step allowed)
        return situ_moe_mxfp4(*args, **kwargs)
    _timed_calls += 1
    timer = PhaseTimer(sync)
    out = situ_moe_mxfp4(*args, phase_hook=timer, **kwargs)
    x, topk_ids = args[0], args[1]
    logging.getLogger("vllm.gaudi.situ").info("SiTU experts call %d/%d: tokens=%d topk=%d dequant=%s %s", _timed_calls,
                                              budget, x.shape[0], topk_ids.shape[-1],
                                              resolve_dequant_mode(kwargs.get("dequant"), x.dtype), timer.summary())
    return out


def _lazy_step_hook() -> Callable[[], None] | None:
    """Shared mark_step policy (extension/graph_state.py); imported lazily, the vLLM plugin load must stay light."""
    from vllm_gaudi.extension.graph_state import lazy_step_hook
    return lazy_step_hook()


def maybe_situ_moe(layer, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, impl: str,
                   group_size: int = MXFP4_BLOCK, token_tile: int = 512) -> torch.Tensor | None:
    """Exact SiTU experts for a layer holding packed MXFP4 weights, or None when the fused op should be used.

    Handles layers whose ``activation`` is "situ" while ``impl`` is "exact"; reads ``beta`` / ``linear_beta`` from
    ``layer.moe_config`` and the stacked per-rank tensors ``w13_weight``, ``w13_weight_scale``, ``w2_weight``,
    ``w2_weight_scale`` (the ones the fused MXFP4 op is built from). Shared by both MXFP4 MoE methods.
    """
    activation = getattr(layer.activation, "value", layer.activation)
    if activation != "situ" or impl != "exact":
        return None
    if group_size != MXFP4_BLOCK:
        raise ValueError(f"SiTU experts expect MXFP4 groups of {MXFP4_BLOCK}, got {group_size}")
    beta = layer.moe_config.activation_situ_beta
    if beta is None:
        raise ValueError("SiTU experts need activation_situ_beta in the MoE config")
    args = (x, topk_ids, topk_weights, layer.w13_weight.data, layer.w13_weight_scale.data,
            layer.w2_weight.data, layer.w2_weight_scale.data, int(layer.moe_config.ep_rank * layer.local_num_experts),
            float(beta), layer.moe_config.activation_situ_linear_beta)
    kwargs = dict(dequant=DEQUANT_MODE_ENV,
                  token_tile=token_tile,
                  step_hook=_lazy_step_hook())
    # once the budget is spent the normal (compilable) path is used again: no graph break per layer
    if _timed_calls < TIMING_BUDGET:
        return _timed_situ_moe_mxfp4(TIMING_BUDGET, *args, **kwargs)
    return situ_moe_mxfp4(*args, **kwargs)
