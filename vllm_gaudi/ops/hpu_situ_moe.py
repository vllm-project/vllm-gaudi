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
                     gather_max_tokens: int | None = None) -> torch.Tensor:
    """Per-rank partial routed-expert output (before the cross-rank reduction), same contract as the fused op.

    x [T, H]; topk_ids / topk_weights [T, K] with GLOBAL expert ids; this rank owns the experts
    ``experts_min .. experts_min + num_local_experts - 1``. ``get_w13(ids)`` returns the dequantized fused
    gate/up weights ``[c, 2 * I, H]`` and ``get_w2(ids)`` the down weights ``[c, H, I]`` of the local experts
    ``ids`` (a LongTensor). Shapes are static (HPU graphs): few tokens (decode) compute only the
    ``G = T * K`` experts that can be hit whenever that is fewer than the local experts (optionally only up to
    ``gather_max_tokens`` tokens), otherwise all local experts are computed densely and masked by the routing
    weight. Everything is accumulated in fp32.
    """
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
    for c0 in range(0, total, expert_chunk):
        chunk_ids = ids[c0:c0 + expert_chunk]
        w13 = get_w13(chunk_ids)  # [c, 2I, H]
        w2 = get_w2(chunk_ids)  # [c, H, I]
        for i, t0 in enumerate(starts):
            xt = x[t0:t0 + tile]
            xe = xt.unsqueeze(0).expand(chunk_ids.shape[0], -1, -1)
            hidden_act = situ_activation(torch.bmm(xe, w13.transpose(1, 2)), beta, linear_beta)  # [c, t, I]
            y = torch.bmm(hidden_act, w2.transpose(1, 2))  # [c, t, H]
            weights = gate_w[t0:t0 + tile].index_select(1, chunk_ids).t().unsqueeze(-1)  # [c, t, 1]
            accs[i] = accs[i] + (y.float() * weights).sum(0)
    outputs = accs
    out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
    return out.to(x.dtype)


def situ_moe_mxfp4(x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, w13: torch.Tensor,
                   w13_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor, experts_min: int, beta: float,
                   linear_beta: float | None, **kwargs) -> torch.Tensor:
    """``situ_moe_experts`` over the stacked packed MXFP4 expert weights of this rank (dequantized per chunk)."""

    def get_w13(ids):
        return dequant_mxfp4(w13.index_select(0, ids), w13_scale.index_select(0, ids), x.dtype)

    def get_w2(ids):
        return dequant_mxfp4(w2.index_select(0, ids), w2_scale.index_select(0, ids), x.dtype)

    return situ_moe_experts(x, topk_ids, topk_weights, get_w13, get_w2, w13.shape[0], experts_min, beta, linear_beta,
                            **kwargs)


def maybe_situ_moe(layer,
                   x: torch.Tensor,
                   topk_ids: torch.Tensor,
                   topk_weights: torch.Tensor,
                   impl: str,
                   group_size: int = MXFP4_BLOCK,
                   token_tile: int = 512) -> torch.Tensor | None:
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
    return situ_moe_mxfp4(*args, token_tile=token_tile)
