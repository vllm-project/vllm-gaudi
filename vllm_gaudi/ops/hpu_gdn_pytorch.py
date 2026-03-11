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

import os
import torch
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


def _chunk_vectorized_body(
    q_chunk: torch.Tensor,   # [Tc, H, K]
    k_chunk: torch.Tensor,   # [Tc, H, K]
    v_chunk: torch.Tensor,   # [Tc, H, V]
    g_chunk: torch.Tensor,   # [Tc, H]
    beta_chunk: torch.Tensor,# [Tc, H]
    state: torch.Tensor,     # [H, V, K]
    eye: torch.Tensor,       # [Tc, Tc]
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compilable vectorized chunk body for hpu_chunk_gated_delta_rule.

    Returns (out_chunk [Tc, H, V], new_state [H, V, K]).
    """
    k_h = k_chunk.permute(1, 0, 2).contiguous()      # [H, Tc, K]
    g_h = g_chunk.transpose(0, 1).contiguous()        # [H, Tc]
    beta_h = beta_chunk.transpose(0, 1).contiguous()  # [H, Tc]

    dot = torch.bmm(k_h, k_h.transpose(1, 2))
    coeff = beta_h.unsqueeze(-1) * torch.exp(
        g_h.unsqueeze(-1) - g_h.unsqueeze(-2)
    )
    a_lower = torch.tril(dot * coeff, diagonal=-1)
    lmat = eye.unsqueeze(0) + a_lower
    A_solve = _hpu_solve_lower_triangular_batched(
        lmat, eye, use_vectorized=True,
    )

    rhs_u = v_chunk.permute(1, 0, 2).contiguous() * beta_h.unsqueeze(-1)
    rhs_w = k_h * (beta_h * torch.exp(g_h)).unsqueeze(-1)
    u_chunk = torch.bmm(A_solve, rhs_u).permute(1, 0, 2).contiguous()
    w_chunk = torch.bmm(A_solve, rhs_w).permute(1, 0, 2).contiguous()

    h_start = state.clone()
    v_new_chunk = u_chunk - torch.einsum("thk,hvk->thv", w_chunk, h_start)

    # Prefer reshape/index broadcasting over chained unsqueeze on
    # sliced tensors for HPU graph lowering stability.
    tc = k_chunk.shape[0]
    H = k_h.shape[0]
    g_last_tc_h = g_chunk[-1:, :]  # [1, H]
    decay_tc_h = torch.exp(g_last_tc_h - g_chunk)
    val_state = v_new_chunk * decay_tc_h.reshape(tc, H, 1)
    new_state = (
        h_start * torch.exp(g_last_tc_h[0]).reshape(H, 1, 1)
        + torch.einsum("thv,thk->hvk", val_state, k_chunk)
    )

    q_h = q_chunk.permute(1, 0, 2).contiguous()
    v_new_h = v_new_chunk.permute(1, 0, 2).contiguous()
    base_h = torch.einsum("htk,hvk->htv", q_h, h_start)
    base_h = base_h * torch.exp(g_h).reshape(H, tc, 1)
    attn_h = torch.bmm(q_h, k_h.transpose(1, 2))
    g_h_l = g_h.reshape(H, tc, 1)
    g_h_r = g_h.reshape(H, 1, tc)
    attn_h = attn_h * torch.exp(g_h_l - g_h_r)
    attn_h = torch.tril(attn_h)
    out_chunk = (
        base_h + torch.bmm(attn_h, v_new_h)
    ).permute(1, 0, 2) * scale

    return out_chunk, new_state


def _recurrent_timestep_body(
    q_t: torch.Tensor,     # [H, K]
    k_t: torch.Tensor,     # [H, K]
    v_t: torch.Tensor,     # [HV, V]
    g_t: torch.Tensor,     # [HV]
    b_t: torch.Tensor,     # [HV]
    h_state: torch.Tensor, # [HV, V, K]
    scale: float,
    HV: int, H: int, Kdim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compilable per-timestep body for hpu_fused_recurrent_gated_delta_rule.

    Returns (out_t [HV, V], updated h_state [HV, V, K]).
    """
    q_t = q_t * scale
    h_state = h_state * torch.exp(g_t).view(HV, 1, 1)
    proj = torch.sum(h_state * k_t.view(H, 1, Kdim), dim=-1)
    v_new = (v_t - proj) * b_t.view(HV, 1)
    h_state = h_state + v_new.unsqueeze(-1) * k_t.view(H, 1, Kdim)
    out_t = torch.sum(h_state * q_t.view(H, 1, Kdim), dim=-1)
    return out_t, h_state


def _hpu_solve_lower_triangular_batched(
    lmat: torch.Tensor,
    eye: torch.Tensor,
    use_vectorized: bool,
) -> torch.Tensor:
    """Solve L X = I for lower-triangular matrices.

    Default: uses ``torch.linalg.inv`` (vectorized) or
    ``torch.linalg.solve_triangular`` (per-head).

    Set ``VLLM_GDN_USE_FORWARD_SUB=1`` to switch to the manual
    forward-substitution path (row-by-row with diagonal division via
    ``bmm``).  This path exists for HPU where linalg ops are unsupported,
    but currently has an **accuracy issue** — the model output degrades
    (e.g. only generates "hello").  Keep the linalg default for correctness
    debugging; enable forward-sub once the accuracy issue is resolved.

    Args:
        lmat: [..., N, N] lower-triangular matrix
        eye:  [N, N] identity matrix (pre-cached for efficiency)
        use_vectorized: True  -> ``torch.linalg.inv`` / vectorized sub
                        False -> ``torch.linalg.solve_triangular`` / per-head sub

    Returns:
        [..., N, N] inverse of lmat
    """
    use_forward_sub = os.getenv("VLLM_GDN_USE_FORWARD_SUB", "0") == "1"

    if not use_forward_sub:
        # --- Default: linalg path (accurate, not supported on HPU) ---
        if use_vectorized:
            return torch.linalg.inv(lmat)
        return torch.linalg.solve_triangular(lmat, eye, upper=False)

    # --- Forward-substitution path (HPU-safe, accuracy issue open) ---
    if lmat.ndim < 2 or lmat.shape[-1] != lmat.shape[-2]:
        raise ValueError(f"Expected square matrix [..., N, N], got {tuple(lmat.shape)}")

    n = lmat.shape[-1]
    batch_shape = lmat.shape[:-2]

    lflat = lmat.reshape(-1, n, n)

    rhs = eye
    if rhs.ndim == 2:
        if batch_shape:
            view_shape = (1,) * len(batch_shape) + rhs.shape
            rhs = rhs.reshape(view_shape).expand(batch_shape + rhs.shape)
    elif rhs.ndim == lmat.ndim:
        if rhs.shape[-2:] != (n, n):
            raise ValueError(
                f"RHS trailing shape must be ({n}, {n}), got {tuple(rhs.shape[-2:])}"
            )
        if rhs.shape[:-2] != batch_shape:
            rhs = rhs.expand(batch_shape + (n, n))
    else:
        raise ValueError(
            f"Unsupported RHS rank: rhs.ndim={rhs.ndim}, lmat.ndim={lmat.ndim}."
        )
    rhs_flat = rhs.reshape(-1, n, n)

    # Row-wise forward substitution with diagonal division.
    # NOTE: This path has a known accuracy issue on HPU — outputs degrade
    # compared to the linalg reference.  Under investigation.
    x = torch.zeros_like(rhs_flat)
    for i in range(n):
        rhs_i = rhs_flat[:, i, :]
        if i > 0:
            corr = torch.bmm(lflat[:, i:i + 1, :i], x[:, :i, :]).squeeze(1)
            rhs_i = rhs_i - corr

        diag = lflat[:, i, i].unsqueeze(-1)
        x[:, i, :] = rhs_i / diag

    return x.reshape(lmat.shape)

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


@torch._dynamo.disable
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

        for t in range(bos, eos):
            q_t = qf[t]
            k_t = kf[t]
            v_t = vf[t]
            g_t = gf[t]
            b_t = bf[t]

            if use_qk_l2norm_in_kernel:
                q_t = _l2norm_last_dim(q_t)
                k_t = _l2norm_last_dim(k_t)

            out_t, h_state = _recurrent_timestep_body(
                q_t, k_t, v_t, g_t, b_t, h_state,
                scale, HV, H, Kdim,
            )
            out[t] = out_t

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


@torch._dynamo.disable
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
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """PyTorch replacement for chunk_gated_delta_rule.

    This path intentionally mirrors upstream prefill call semantics without
    delegating to the fused recurrent helper.
    """
    if torch.compiler.is_compiling():
        return _hpu_chunk_gated_delta_rule_eager(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            chunk_size=chunk_size,
        )
    # https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py#L132
    B, T, H, Kdim = q.shape
    _, _, HV, Vdim = v.shape
    device = q.device
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

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
        #TODO: vectorize this loop with a custom scan or by reshaping to [num_chunks, chunk_size, H] and doing a grouped cumsum with resets at chunk boundaries.
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
    # Default vectorized path; set to 0 to use the current per-head path.
    use_vectorized_chunk = (
        os.getenv("VLLM_GAUDI_GDN_CHUNK_VECTORIZED", "1") == "1"
    )
    if torch.compiler.is_compiling():
        use_vectorized_chunk = False

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

            if tc not in eye_cache:
                eye_cache[tc] = torch.eye(tc, dtype=torch.float32, device=device)

            if use_vectorized_chunk:
                out[cs:ce], state = _chunk_vectorized_body(
                    q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk,
                    state, eye_cache[tc], scale,
                )
            else:
                # Per-head reference path (list accumulation for HPU compat).
                a_solve_list: list[torch.Tensor] = []
                for h in range(H):
                    kh = k_chunk[:, h, :]
                    bh = beta_chunk[:, h]
                    gh = g_chunk[:, h]
                    dot = kh @ kh.transpose(0, 1)
                    coeff = bh[:, None] * torch.exp(gh[:, None] - gh[None, :])
                    a_lower = torch.tril(dot * coeff, diagonal=-1)
                    lmat = eye_cache[tc] + a_lower
                    a_solve_h = _hpu_solve_lower_triangular_batched(
                        lmat,
                        eye_cache[tc],
                        use_vectorized=False,
                    )
                    a_solve_list.append(a_solve_h.unsqueeze(0))
                A_solve = torch.cat(a_solve_list, dim=0)

                u_list: list[torch.Tensor] = []
                w_list: list[torch.Tensor] = []
                for h in range(H):
                    rhs_u = v_chunk[:, h, :] * beta_chunk[:, h:h + 1]
                    rhs_w = (
                        k_chunk[:, h, :]
                        * (beta_chunk[:, h] * torch.exp(g_chunk[:, h]))[:, None]
                    )
                    u_list.append((A_solve[h] @ rhs_u).unsqueeze(1))
                    w_list.append((A_solve[h] @ rhs_w).unsqueeze(1))
                u_chunk = torch.cat(u_list, dim=1)
                w_chunk = torch.cat(w_list, dim=1)

                v_new_list: list[torch.Tensor] = []
                h_start = state.clone()
                for h in range(H):
                    state_h = h_start[h]
                    proj = w_chunk[:, h, :] @ state_h.transpose(0, 1)
                    val_raw = u_chunk[:, h, :] - proj
                    v_new_list.append(val_raw.unsqueeze(1))

                    g_last = g_chunk[-1, h]
                    val_state = val_raw * torch.exp(g_last - g_chunk[:, h])[:, None]
                    state_h = state_h * torch.exp(g_last)
                    state_h = state_h + val_state.transpose(0, 1) @ k_chunk[:, h, :]
                    state[h] = state_h
                v_new_chunk = torch.cat(v_new_list, dim=1)

                out_list: list[torch.Tensor] = []
                for h in range(H):
                    qh = q_chunk[:, h, :]
                    kh = k_chunk[:, h, :]
                    vh = v_new_chunk[:, h, :]
                    hs = h_start[h]
                    gh = g_chunk[:, h]

                    base = qh @ hs.transpose(0, 1)
                    base = base * torch.exp(gh)[:, None]
                    attn = qh @ kh.transpose(0, 1)
                    attn = attn * torch.exp(gh[:, None] - gh[None, :])
                    attn = torch.tril(attn)
                    out_list.append(((base + attn @ vh) * scale).unsqueeze(1))
                out[cs:ce] = torch.cat(out_list, dim=1)

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


@torch._dynamo.disable
def _hpu_chunk_gated_delta_rule_eager(
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
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return hpu_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        chunk_size=chunk_size,
    )
