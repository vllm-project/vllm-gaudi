# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch gathered-expert MoE combine for silu + FP8-per-channel weights.

Replaces the Habana ``mixture_of_experts`` combine (a fixed per-layer launch
pipeline) with a leaner active-expert gather + GEMM + weighted-reduce path,
mirroring ``vllm_gaudi.ops.hpu_fused_moe._gather_swigluoai_moe`` but for a silu
gated activation and the FP8 per-channel weight layout
(``extension/ops.py:fp8_channel_moe_prepare_weights``).

Only ``g = min(E_local, tokens * K)`` distinct routed experts are read from HBM
and computed (<= 8 at BS=1 for K=8), instead of the Habana op's fixed stage
pipeline. The gathered count is static (no ``torch.nonzero``/host branch), so the
path captures cleanly into a compiled HPU graph.

Numeric fidelity: ``x`` and the weights are FP8; ``x`` is quantized the same way
the op does (``dynamic_quant``), everything is dequantized to fp32, and the two
GEMMs + silu + weighted sum accumulate in fp32, rounding to the output dtype only
at the end. Exact bit-equality with the black-box op is NOT expected; correctness
is assessed via an FP8-ULP bar.
"""
from __future__ import annotations

import torch


def _dynamic_quant(data):
    # import lazily to avoid pulling heavy deps at module import
    from vllm_gaudi.extension.ops import dynamic_quant
    return dynamic_quant(data)


def gather_silu_fp8_moe(layer, x, topk_ids, topk_weights, activation="silu"):
    """x [T,H] bf16 -> MoE output [T,H] bf16.

    topk_ids [T,K] int64 (global expert ids), topk_weights [T,K] bf16.
    Mirrors the Habana op's data flow: fp8-quantize x, per-token weighted sum
    over the routed experts, each expert computed as silu(w13 x) w2.
    """
    assert activation == "silu", f"custom combine supports silu, got {activation}"

    T, H = x.shape
    K = topk_ids.shape[-1]
    w13 = layer.w13_weight  # [E, 2I, H] fp8
    w2 = layer.w2_weight    # [E, H, I] fp8
    s13 = layer.w13_weight_scale_inv  # [E, 2I]
    s2 = layer.w2_weight_scale_inv    # [E, H]
    E = w13.shape[0]
    I = w13.shape[1] // 2

    # ---- FP8-quantize x per token (match the op's input quantization) ----
    x_fp8, x_scale = _dynamic_quant(x)   # x_fp8 [T,H], x_scale [T,1] f32
    x_f = x_fp8.to(torch.float32) * x_scale  # [T,H] f32

    # ---- per-token expert combine weights (scatter over E) ----
    # topk_ids are GLOBAL expert ids; remap to this rank's local ids and mask
    # experts owned by other EP ranks (they contribute +0.0 to this rank's
    # partial, which is then reduce-scattered by the runner). At TP=1
    # ep_rank=0 -> experts_min=0, identical to the unremapped path.
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    local_ids = topk_ids - experts_min                                   # [T,K]
    in_range = (local_ids >= 0) & (local_ids < E)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    safe_w = torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32)
    gate_w = x.new_zeros(T, E, dtype=torch.float32)
    gate_w.scatter_add_(1, safe_ids, safe_w)                             # [T,E] f32

    # ---- static gathered-expert count (>= # distinct hit experts) ----
    # The number of distinct routed experts is provably <= tokens * K, so with
    # g = min(E, tokens*K) every real hit is included and the extra (zero-weight)
    # padding experts contribute exactly +0.0 to the weighted sum. Keeping `g`
    # static (no torch.nonzero, no `if G == 0`) lets this capture cleanly into a
    # compiled HPU graph (mirrors _gather_swigluoai_moe).
    g = min(E, T * K)
    hit = (gate_w != 0).float().sum(0)                       # [E]
    gather_ids = torch.topk(hit, g, sorted=False).indices    # [G]
    gather_ids, _ = torch.sort(gather_ids)                   # ascending ids

    # ---- gather only the active experts' weights + per-channel scales ----
    w13_g = w13.index_select(0, gather_ids)      # [G, 2I, H] fp8
    w2_g = w2.index_select(0, gather_ids)        # [G, H, I] fp8
    s13_g = s13.index_select(0, gather_ids)      # [G, 2I]
    s2_g = s2.index_select(0, gather_ids)        # [G, H]
    w13_f = w13_g.to(torch.float32) * s13_g.unsqueeze(-1)  # [G, 2I, H] f32
    w2_f = w2_g.to(torch.float32) * s2_g.unsqueeze(-1)     # [G, H, I] f32

    # ---- per-expert MLP ----
    xe = x_f.unsqueeze(0).expand(g, T, H)                 # [G, T, H]
    h = torch.bmm(xe, w13_f.transpose(1, 2))              # [G, T, 2I] f32
    gate, up = h[..., :I], h[..., I:]
    act = gate * torch.sigmoid(gate) * up                 # silu(gate)*up, [G,T,I]
    y = torch.bmm(act, w2_f.transpose(1, 2))              # [G, T, H] f32

    # ---- weighted sum over experts -> [T, H] ----
    gate_wg = gate_w.index_select(1, gather_ids).t()      # [G, T]
    out = (y * gate_wg.unsqueeze(-1)).sum(0)              # [T, H] f32
    return out.to(x.dtype)
