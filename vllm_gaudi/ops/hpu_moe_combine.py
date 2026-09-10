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

The UP/GATE projection runs in native FP8 via ``torch.ops.hpu.fp8_gemm_v2``
(fp32 internal accumulation): ``x`` is shared across all routed experts, so the G
per-expert GEMMs collapse into ONE wide fp8 GEMM over concatenated expert weights
``[T,H] x [H, G*2I] -> [T, G*2I]`` (expert ``g`` at columns
``[g*2I:(g+1)*2I]`` via ``permute(2,0,1)``), and the gathered ``w13`` weights are
NEVER dequantized to fp32.

The DOWN projection keeps ``act`` high-precision (fp32): ``act`` is the silu
output with a wide dynamic range, so fp8-quantizing it would lose ~2-6% precision
(3 mantissa bits) and blow the 2-ULP bar. HPU's fp8 GEMM ops accept only fp8
activations, so ``w2`` is dequantized to fp32 for a plain fp32 bmm (as the
baseline did). Only the up/gate GEMM benefits from native fp8.

Numeric fidelity: ``x``/``w13`` are FP8 with per-channel scales folded into
``fp8_gemm_v2`` (fp32 internal accumulation); the silu and weighted sum run in
fp32, rounding to the output dtype only at the end. Exact bit-equality with the
black-box op is NOT expected; correctness is assessed via an FP8-ULP bar.
"""
from __future__ import annotations

import torch


def _dynamic_quant(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # import lazily to avoid pulling heavy deps at module import
    from vllm_gaudi.extension.ops import dynamic_quant
    return dynamic_quant(data)


def gather_silu_fp8_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the rank-local FP8 silu MoE partial for routed experts only.

    topk_ids [T,K] int64 (global expert ids), topk_weights [T,K] bf16.
    Mirrors the Habana op's data flow: fp8-quantize x, per-token weighted sum
    over the routed experts, each expert computed as silu(w13 x) w2.
    """
    tokens, hidden_size = x.shape
    num_topk = topk_ids.shape[-1]
    w13 = layer.w13_weight  # [E, 2I, H] fp8
    w2 = layer.w2_weight  # [E, H, I] fp8
    scale13 = layer.w13_weight_scale_inv  # [E, 2I]
    scale2 = layer.w2_weight_scale_inv  # [E, H]
    local_experts = w13.shape[0]
    intermediate_size = w13.shape[1] // 2

    # ---- FP8-quantize x per token (match the op's input quantization) ----
    # x_scale is kept as [T,1] (NOT squeezed): fp8_gemm_v2 requires the row
    # scale in 2-D form; squeezing only works at T=1 and silently mis-scales at
    # larger T.
    x_fp8, x_scale = _dynamic_quant(x)  # x_fp8 [T,H], x_scale [T,1] f32

    # ---- per-token expert combine weights (scatter over E) ----
    # topk_ids are GLOBAL expert ids; remap to this rank's local ids and mask
    # experts owned by other EP ranks (they contribute +0.0 to this rank's
    # partial, which is then reduce-scattered by the runner). At TP=1
    # ep_rank=0 -> experts_min=0, identical to the unremapped path.
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)
    local_ids = topk_ids - experts_min  # [T,K]
    in_range = (local_ids >= 0) & (local_ids < local_experts)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    safe_weights = torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).float()
    gate_weights = x.new_zeros(tokens, local_experts, dtype=torch.float32)
    gate_weights.scatter_add_(1, safe_ids, safe_weights)  # [T,E] f32

    # ---- static gathered-expert count (>= # distinct hit experts) ----
    # The number of distinct routed experts is provably <= tokens * K, so with
    # g = min(E, tokens*K) every real hit is included and the extra (zero-weight)
    # padding experts contribute exactly +0.0 to the weighted sum. Keeping `g`
    # static (no torch.nonzero, no `if G == 0`) lets this capture cleanly into a
    # compiled HPU graph (mirrors _gather_swigluoai_moe).
    gathered_experts = min(local_experts, tokens * num_topk)
    hit_counts = (gate_weights != 0).float().sum(0)  # [E]
    gather_ids = torch.topk(hit_counts, gathered_experts, sorted=False).indices  # [G]
    gather_ids, _ = torch.sort(gather_ids)  # ascending ids

    # ---- gather only the active experts' weights + per-channel scales ----
    w13_gathered = w13.index_select(0, gather_ids)  # [G, 2I, H] fp8
    w2_gathered = w2.index_select(0, gather_ids)  # [G, H, I] fp8
    scale13_gathered = scale13.index_select(0, gather_ids)  # [G, 2I]
    scale2_gathered = scale2.index_select(0, gather_ids)  # [G, H]

    # ---- per-expert MLP, all in FP8 (no dequant to fp32) ----
    #
    # UP/GATE projection: x is shared across all routed experts, so the G
    # per-expert GEMMs `x @ w13[g]` collapse into ONE wide fp8 GEMM over the
    # concatenated expert weights [T,H] x [H, G*2I] -> [T, G*2I]. Expert g's
    # block lands on columns [g*2I:(g+1)*2I] via the permute(2,0,1) ordering
    # (NOT permute(1,0,2), which interleaves experts and is wrong). The
    # per-channel B_scale_inv is concatenated to match, with B_scale_shape
    # declaring per-channel (not per-block) scaling. fp8_gemm_v2 accumulates in
    # fp32 internally and returns out_dtype; A_scale_inv MUST be passed as [T,1].
    w13_columns = w13_gathered.permute(2, 0, 1).reshape(hidden_size, -1)  # [H, G*2I] fp8
    scale13_columns = scale13_gathered.reshape(-1)  # [G*2I]
    projected = torch.ops.hpu.fp8_gemm_v2(
        A=x_fp8,
        trans_A=False,
        B=w13_columns,
        trans_B=False,
        D=None,
        out_dtype=torch.float32,
        A_scale_inv=x_scale,
        B_scale_inv=scale13_columns,
        B_scale_shape=[gathered_experts * 2 * intermediate_size],
        bias=None,
        accumulate=False,
    )  # [T, G*2I] f32
    projected = projected.reshape(tokens, gathered_experts, 2 * intermediate_size).permute(1, 0, 2)  # [G, T, 2I] f32
    gate, up = projected[..., :intermediate_size], projected[..., intermediate_size:]
    activations = gate * torch.sigmoid(gate) * up  # silu(gate)*up, [G,T,I] f32

    # DOWN projection: act has a wide dynamic range (silu output), so it CANNOT
    # be fp8-quantized without losing ~2-6% precision (fp8 has only 3 mantissa
    # bits), which would blow the 2-ULP bar. The stock op keeps act high-precision
    # here, and HPU's fp8 GEMM ops (fp8_gemm_v2 / fp8_gemm) only accept fp8
    # activations. So the down projection dequantizes w2 to fp32 and runs a plain
    # fp32 bmm (same as the baseline); only the UP/GATE projection runs in native
    # fp8.
    w2_float = w2_gathered.float() * scale2_gathered.unsqueeze(-1)  # [G, H, I] f32
    expert_outputs = torch.bmm(activations, w2_float.transpose(1, 2))  # [G, T, H] f32

    # ---- weighted sum over experts -> [T, H] ----
    gathered_weights = gate_weights.index_select(1, gather_ids).t()  # [G, T]
    out = (expert_outputs * gathered_weights.unsqueeze(-1)).sum(0)  # [T, H] f32
    return out.to(x.dtype)
