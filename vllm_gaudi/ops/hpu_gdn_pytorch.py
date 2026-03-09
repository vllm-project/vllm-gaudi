# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HPU-native PyTorch implementations for Qwen3.5 GDN ops.

These implementations intentionally avoid Triton/CUDA-only kernels and run
entirely with PyTorch tensor ops on the active device (HPU for Gaudi runs).
Phase 1 scope:
- non-mixed prefill/decode support
- no speculative decode tensor layout support yet
"""

from __future__ import annotations

import torch
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

def _materialize_seq_ranges(cu_seqlens: torch.Tensor, total_tokens: int) -> list[tuple[int, int]]:
    """Convert cu_seqlens to safe [bos, eos) ranges on CPU.

    Lazy-mode HPU tensors can produce unexpected scalar values when accessed
    repeatedly via .item() on-device. Materialize once on CPU and clamp to
    token bounds to keep Python-side loops safe.
    """
    try:
        cu_cpu = cu_seqlens.to(dtype=torch.int64, device="cpu")
    except RuntimeError as exc:
        # In some lazy/graph captures, host transfer of cu_seqlens is not
        # allowed. Fall back to one contiguous sequence to keep execution
        # alive instead of crashing.
        logger.warning(
            "[GDN seq range] Failed to materialize cu_seqlens on CPU (%s). "
            "Falling back to a single contiguous range [0, %d).",
            exc,
            total_tokens,
        )
        return [(0, total_tokens)]

    cu_list = cu_cpu.tolist()

    ranges: list[tuple[int, int]] = []
    for i in range(max(0, len(cu_list) - 1)):
        bos_raw = int(cu_list[i])
        eos_raw = int(cu_list[i + 1])
        bos = min(max(bos_raw, 0), total_tokens)
        eos = min(max(eos_raw, 0), total_tokens)
        if eos < bos:
            eos = bos
        ranges.append((bos, eos))
    return ranges


def _l2norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def hpu_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch replacement for fused_gdn_gating.

    Returns:
      g: [1, num_tokens, num_heads] float32
      beta_out: [1, num_tokens, num_heads] same dtype as b
    """
    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    use_softplus = (beta * x) <= threshold
    softplus_x = torch.where(use_softplus, (1.0 / beta) * torch.log1p(torch.exp(beta * x)), x)
    g = -torch.exp(A_log.to(torch.float32)) * softplus_x
    beta_out = torch.sigmoid(b.to(torch.float32)).to(b.dtype)
    return g.unsqueeze(0), beta_out.unsqueeze(0)


def hpu_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch replacement for fused_recurrent_gated_delta_rule.

    This implementation supports the non-speculative paths used by current
    Gaudi Qwen3.5 integration.
    """
    if num_accepted_tokens is not None:
        raise NotImplementedError("Speculative decode path is not implemented in phase 1.")
    if ssm_state_indices is not None and ssm_state_indices.ndim > 1:
        raise NotImplementedError("2D ssm_state_indices (spec decode) is not implemented in phase 1.")

    if beta is None:
        beta = torch.ones_like(g)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Shapes: q/k [B, T, H, K], v [B, T, HV, V], g/beta [B, T, HV]
    B, T, H, Kdim = q.shape
    _, _, HV, Vdim = v.shape
    device = q.device

    # Match upstream kernel semantics: when HV > H, each q/k head is shared
    # across a group of value heads (grouped-value attention).
    if H != HV:
        if HV % H == 0:
            repeat = HV // H
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
            H = HV
        else:
            raise ValueError(
                f"Unsupported head mapping in hpu_fused_recurrent_gated_delta_rule: "
                f"q/k heads={H}, value heads={HV}. Expected HV % H == 0."
            )

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("When cu_seqlens is used, expected batch size B=1.")
        seq_ranges = _materialize_seq_ranges(cu_seqlens, B * T)
        num_seqs = len(seq_ranges)
    else:
        num_seqs = B
        seq_ranges = [(i * T, (i + 1) * T) for i in range(B)]

    if initial_state is None:
        final_state = torch.zeros((num_seqs, HV, Vdim, Kdim), dtype=torch.float32, device=device)
    else:
        final_state = initial_state if inplace_final_state else initial_state.clone()

    # Always compute in fp32 for stability and cast back on writes.
    state_work = final_state.to(torch.float32)

    # Flatten token axis for varlen path (B is expected to be 1 there).
    qf = q.reshape(-1, H, Kdim).to(torch.float32)
    kf = k.reshape(-1, H, Kdim).to(torch.float32)
    vf = v.reshape(-1, HV, Vdim).to(torch.float32)
    gf = g.reshape(-1, HV).to(torch.float32)
    bf = beta.reshape(-1, HV).to(torch.float32)

    out = torch.empty((qf.shape[0], HV, Vdim), dtype=torch.float32, device=device)

    state_indices_tensor: torch.Tensor | None = None
    state_indices_valid: torch.Tensor | None = None
    if ssm_state_indices is not None:
        state_indices_tensor = ssm_state_indices.reshape(-1).to(
            dtype=torch.long,
            device=state_work.device,
        )
        state_indices_valid = (
            (state_indices_tensor >= 0)
            & (state_indices_tensor < state_work.shape[0])
        )

    num_state_indices = (
        int(state_indices_tensor.shape[0]) if state_indices_tensor is not None else 0
    )
    #print(f"libin debug recur {seq_ranges=}")
    for seq_id, (bos, eos) in enumerate(seq_ranges):
        if eos <= bos:
            continue

        if state_indices_tensor is not None and state_indices_valid is not None:
            if seq_id >= num_state_indices:
                continue

            seq_id_t = torch.tensor([seq_id], dtype=torch.long, device=state_work.device)
            valid_seq = state_indices_valid.index_select(0, seq_id_t)
            raw_idx = state_indices_tensor.index_select(0, seq_id_t)
            safe_idx = torch.where(valid_seq, raw_idx, torch.zeros_like(raw_idx))
            prev_state = state_work.index_select(0, safe_idx)
            h_state = prev_state.squeeze(0)
        else:
            h_state = state_work[seq_id]
        print(f"libin debug recur {seq_id=} {bos=} {eos=} {h_state.shape=}")
        for t in range(bos, eos):
            q_t = qf[t]
            k_t = kf[t]
            v_t = vf[t]
            g_t = gf[t]
            b_t = bf[t]

            if use_qk_l2norm_in_kernel:
                q_t = _l2norm_last_dim(q_t)
                k_t = _l2norm_last_dim(k_t)

            q_t = q_t * scale

            # Gating decay on recurrent state.
            h_state.mul_(torch.exp(g_t).view(HV, 1, 1))

            # v update and recurrent state update.
            proj = torch.sum(h_state * k_t.view(H, 1, Kdim), dim=-1)
            v_new = (v_t - proj) * b_t.view(HV, 1)
            h_state.add_(v_new.unsqueeze(-1) * k_t.view(H, 1, Kdim))

            # Output projection.
            out[t] = torch.sum(h_state * q_t.view(H, 1, Kdim), dim=-1)

        # Persist state back to selected cache line.
        if state_indices_tensor is not None and state_indices_valid is not None:
            # Avoid Python-side scalar branching in graph mode: for invalid
            # indices, write back the unchanged state.
            updated_state = torch.where(
                valid_seq.view(1, 1, 1),
                h_state.unsqueeze(0),
                prev_state,
            )
            state_work.index_copy_(0, safe_idx, updated_state)
        else:
            state_work[seq_id] = h_state

    final_state.copy_(state_work.to(final_state.dtype))
    out = out.to(v.dtype)

    if cu_seqlens is not None:
        out = out.unsqueeze(0)
    else:
        out = out.view(B, T, HV, Vdim)

    return out, final_state


def hpu_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """PyTorch replacement for chunk_gated_delta_rule.

    This path intentionally mirrors upstream prefill call semantics without
    delegating to the fused recurrent helper.
    """

    B, T, H, Kdim = q.shape
    _, _, HV, Vdim = v.shape
    device = q.device
    chunk_size = 64

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("When cu_seqlens is used, expected batch size B=1.")
        seq_ranges = _materialize_seq_ranges(cu_seqlens, B * T)
        num_seqs = len(seq_ranges)
    else:
        num_seqs = B
        seq_ranges = [(i * T, (i + 1) * T) for i in range(B)]

    # Match upstream grouped-value semantics.
    if H != HV:
        if HV % H == 0:
            repeat = HV // H
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
            H = HV
        else:
            raise ValueError(
                "Unsupported head mapping in hpu_chunk_gated_delta_rule: "
                f"q/k heads={H}, value heads={HV}. Expected HV % H == 0."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Match upstream ChunkGatedDeltaRuleFunction behavior: normalize full
    # q/k tensors before the core chunk pipeline.
    if use_qk_l2norm_in_kernel:
        q = _l2norm_last_dim(q.to(torch.float32))
        k = _l2norm_last_dim(k.to(torch.float32))

    # Flatten token axis for shared varlen/non-varlen logic.
    qf = q.reshape(-1, H, Kdim).to(torch.float32)
    kf = k.reshape(-1, H, Kdim).to(torch.float32)
    vf = v.reshape(-1, HV, Vdim).to(torch.float32)
    gf = g.reshape(-1, HV).to(torch.float32)
    bf = beta.reshape(-1, HV).to(torch.float32)

    # Upstream match: `chunk_local_cumsum` in fla/ops/cumsum.py.
    # Stage 1 computes per-chunk cumulative g in log-space.
    g_cumsum = torch.empty_like(gf)
    for bos, eos in seq_ranges:
        for cs in range(bos, eos, chunk_size):
            ce = min(cs + chunk_size, eos)
            # [Stage 1: chunk_local_cumsum] per-chunk prefix sum of g.
            # Resets at each chunk boundary [cs, ce).
            g_cumsum[cs:ce] = torch.cumsum(gf[cs:ce], dim=0)

    # Initial state layout: [num_seqs, H, V, K].
    if initial_state is None:
        init_state = torch.zeros(
            (num_seqs, H, Vdim, Kdim),
            dtype=torch.float32,
            device=device,
        )
    else:
        if initial_state.shape[0] != num_seqs:
            raise ValueError(
                "The number of initial states is expected to equal the number "
                f"of input sequences ({num_seqs}), got {initial_state.shape[0]}."
            )
        init_state = initial_state.to(torch.float32)

    out = torch.empty((qf.shape[0], H, Vdim), dtype=torch.float32, device=device)
    final_state = torch.empty_like(init_state) if output_final_state else None

    # Upstream stage mapping (chunk.py::chunk_gated_delta_rule_fwd):
    # Stage 2  -> `chunk_scaled_dot_kkt_fwd`
    # Stage 3  -> `solve_tril`
    # Stage 4  -> `recompute_w_u_fwd`
    # Stage 5  -> `chunk_gated_delta_rule_fwd_h`
    #   - kernel variant mapping:
    #     `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` is represented by
    #     this PyTorch stage when chunk_size == 64 (see `chunk_size` above).
    #     The loop below computes the same per-chunk state transition and
    #     v_new accumulation semantics as the Triton kernel, but in eager
    #     PyTorch math.
    # Stage 6  -> `chunk_fwd_o`
    # Implemented below with PyTorch math to avoid Triton dependency.
    eye_cache: dict[int, torch.Tensor] = {}
    #print(f"libin debug chunk {seq_ranges=}")
    for seq_id, (bos, eos) in enumerate(seq_ranges):
        if eos <= bos:
            if final_state is not None:
                final_state[seq_id] = init_state[seq_id]
            continue

        state = init_state[seq_id].clone()  # [H, V, K]
        for cs in range(bos, eos, chunk_size):
            ce = min(cs + chunk_size, eos)
            tc = ce - cs

            q_chunk = qf[cs:ce]          # [Tc, H, K]
            k_chunk = kf[cs:ce]          # [Tc, H, K]
            v_chunk = vf[cs:ce]          # [Tc, H, V]
            g_chunk = g_cumsum[cs:ce]    # [Tc, H]
            beta_chunk = bf[cs:ce]       # [Tc, H]

            # Upstream match: `chunk_scaled_dot_kkt_fwd` + `solve_tril`.
            # A_lower[t, j] = beta[t] * <k_t, k_j> * exp(g_t - g_j), j < t
            # A_solve = (I + A_lower)^(-1)
            A_solve = torch.empty((H, tc, tc), dtype=torch.float32, device=device)
            for h in range(H):
                kh = k_chunk[:, h, :]  # [Tc, K]
                bh = beta_chunk[:, h]  # [Tc]
                gh = g_chunk[:, h]     # [Tc]
                dot = kh @ kh.transpose(0, 1)
                coeff = bh[:, None] * torch.exp(gh[:, None] - gh[None, :])
                a_lower = torch.tril(dot * coeff, diagonal=-1)
                if tc not in eye_cache:
                    eye_cache[tc] = torch.eye(tc, dtype=torch.float32, device=device)
                lmat = eye_cache[tc] + a_lower
                A_solve[h] = torch.linalg.solve_triangular(
                    lmat,
                    eye_cache[tc],
                    upper=False,
                )

            # Upstream match: `recompute_w_u_fwd`.
            u_chunk = torch.empty((tc, H, Vdim), dtype=torch.float32, device=device)
            w_chunk = torch.empty((tc, H, Kdim), dtype=torch.float32, device=device)
            for h in range(H):
                rhs_u = v_chunk[:, h, :] * beta_chunk[:, h:h + 1]
                rhs_w = (
                    k_chunk[:, h, :]
                    * (beta_chunk[:, h] * torch.exp(g_chunk[:, h]))[:, None]
                )
                u_chunk[:, h, :] = A_solve[h] @ rhs_u
                w_chunk[:, h, :] = A_solve[h] @ rhs_w

            # Upstream match: `chunk_gated_delta_rule_fwd_h`.
            # This block is the functional mapping of
            # `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` in fallback
            # mode (blockdim64 <-> chunk_size=64).
            # Performs chunk-level state transition and produces v_new.
            v_new_chunk = torch.empty((tc, H, Vdim), dtype=torch.float32, device=device)
            # Stage 6 must use the chunk-start state. Keep a snapshot before
            # Stage 5 mutates `state` in-place.
            h_start = state.clone()
            for h in range(H):
                state_h = h_start[h]  # [V, K]
                # [Tc, V] = [Tc, K] @ [K, V]
                proj = w_chunk[:, h, :] @ state_h.transpose(0, 1)
                # Match upstream chunk_delta_h: v_new consumed by chunk_fwd_o
                # is the unnormalized residual before applying g_last-g_t.
                val_raw = u_chunk[:, h, :] - proj
                v_new_chunk[:, h, :] = val_raw

                g_last = g_chunk[-1, h]
                # Only state update uses the within-chunk g normalization.
                val_state = val_raw * torch.exp(g_last - g_chunk[:, h])[:, None]
                state_h = state_h * torch.exp(g_last)

                # state_h += v_new^T @ k
                state_h = state_h + val_state.transpose(0, 1) @ k_chunk[:, h, :]
                state[h] = state_h

            # Upstream match: `chunk_fwd_o`.
            for h in range(H):
                qh = q_chunk[:, h, :]     # [Tc, K]
                kh = k_chunk[:, h, :]     # [Tc, K]
                vh = v_new_chunk[:, h, :] # [Tc, V]
                hs = h_start[h]           # [V, K]
                gh = g_chunk[:, h]        # [Tc]

                # [Stage 6: chunk_fwd_o] term 1, recurrent base from h_start.
                base = qh @ hs.transpose(0, 1)  # [Tc, V]
                # Match upstream chunk_fwd_o: base is weighted by exp(g_t).
                base = base * torch.exp(gh)[:, None]
                # [Stage 6: chunk_fwd_o] term 2, lower-triangular intra-chunk
                # contribution weighted by exp(g_i - g_j).
                attn = qh @ kh.transpose(0, 1)  # [Tc, Tc]
                attn = attn * torch.exp(gh[:, None] - gh[None, :])
                attn = torch.tril(attn)
                # [Stage 6: chunk_fwd_o] final output for this chunk/head.
                out[cs:ce, h, :] = (base + attn @ vh) * scale

        if final_state is not None:
            final_state[seq_id] = state

    out = out.to(q.dtype)
    if cu_seqlens is not None:
        out = out.unsqueeze(0)
    else:
        out = out.view(B, T, H, Vdim)

    if final_state is None:
        return out, None
    if initial_state is not None:
        final_state = final_state.to(initial_state.dtype)
    return out, final_state
