# SPDX-License-Identifier: Apache-2.0
"""Pure PyTorch Kimi Delta Attention (KDA) for Kimi-K3 on HPU (correctness first).

No vLLM / Habana imports here, so the functions can be unit-tested on CPU tensors.

Math (per head, state ``S[v, k]`` of shape ``[head_dim, head_dim]``, ``scale = head_dim ** -0.5``)::

    q, k   <- l2norm(q), l2norm(k)                  (eps 1e-6 inside the sqrt), q *= scale
    g[k]   <- lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))        if lower_bound is set
              -exp(A_log) * softplus(raw_g + dt_bias)                      otherwise
    beta   <- sigmoid(raw_beta)
    S      <- S * exp(g)[None, :]                   (decay along the key axis)
    v'     <- beta * (v - S @ k)
    S      <- S + v' (outer) k
    o      <- S @ q

q, k, v come from a depthwise causal conv (kernel ``W``, silu) over the packed ``[q | k | v]`` channels whose
state is the last ``W - 1`` conv inputs per request.

Shapes are static per bucket: prefill takes a uniform ``[B, L]`` padded batch (padding at the end of every
sequence) and runs the chunk-parallel delta rule ``kda_chunked_steps_vec`` (the state-independent part is batched
over the chunks, only the state recurrence loops over L / chunk chunks). ``kda_chunked_steps`` and the token loop
``kda_recurrent_steps`` are the references it is validated against. Padded tokens are exact no-ops (``g = 0``,
``beta = 0``). There is no host sync and no data dependent Python control flow, so the functions work eagerly and
under torch.compile.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def kda_gate(raw_g: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
             lower_bound: float | None) -> torch.Tensor:
    """Log of the per-key-channel forget gate. raw_g: [..., H, D], A_log: [H], dt_bias: [H, D]. Returns fp32."""
    x = raw_g.float() + dt_bias.float()
    a = torch.exp(A_log.float()).unsqueeze(-1)
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * x)
    softplus = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    return -a * softplus


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


def gated_rmsnorm_sigmoid(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FusedRMSNormGated(activation='sigmoid'): rmsnorm(x) * weight * sigmoid(gate), computed in fp32."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = normed * weight.float()
    return (normed * torch.sigmoid(gate.float())).to(x.dtype)


# ---------------------------------------------------------------------------------------------------------------
# State cache access. The caches have one slot per state index; padded batch rows carry index -1, which maps to
# the last ("garbage") slot via remainder (same convention as the Qwen GDN path, and safe under HPU compile).
# ---------------------------------------------------------------------------------------------------------------
def _slots(state_idx: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
    return torch.remainder(state_idx.long(), cache.shape[0])


def _conv_is_dim_first(conv_cache: torch.Tensor, channels: int, width_m1: int) -> bool:
    """[slots, C, W-1] (dim first) or [slots, W-1, C]."""
    if conv_cache.shape[-2:] == (channels, width_m1):
        return True
    if conv_cache.shape[-2:] == (width_m1, channels):
        return False
    raise ValueError(f"unexpected conv cache shape {tuple(conv_cache.shape)} for C={channels}, W-1={width_m1}")


def read_conv_state(conv_cache: torch.Tensor, slots: torch.Tensor, channels: int, width_m1: int) -> torch.Tensor:
    """Returns [B, W-1, C] fp32 (oldest input first)."""
    state = conv_cache.index_select(0, slots).float()
    if _conv_is_dim_first(conv_cache, channels, width_m1):
        state = state.transpose(-1, -2)
    return state


def write_conv_state(conv_cache: torch.Tensor, slots: torch.Tensor, new_state: torch.Tensor) -> None:
    """new_state: [B, W-1, C]."""
    if _conv_is_dim_first(conv_cache, new_state.shape[-1], new_state.shape[-2]):
        new_state = new_state.transpose(-1, -2)
    conv_cache.index_copy_(0, slots, new_state.to(dtype=conv_cache.dtype).contiguous())


def read_ssm_state(ssm_cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    return ssm_cache.index_select(0, slots).float()


def write_ssm_state(ssm_cache: torch.Tensor, slots: torch.Tensor, state: torch.Tensor) -> None:
    ssm_cache.index_copy_(0, slots, state.to(dtype=ssm_cache.dtype).contiguous())


# ---------------------------------------------------------------------------------------------------------------
# Delta rule
# ---------------------------------------------------------------------------------------------------------------
def kda_recurrent_step(state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor,
                       beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One token. state [B,H,V,K]; q,k [B,H,K] (normalized, q scaled); v [B,H,V]; decay=exp(g) [B,H,K]; beta [B,H].

    Returns (o [B,H,V], new_state).
    """
    state = state * decay.unsqueeze(-2)
    proj = torch.matmul(state, k.unsqueeze(-1)).squeeze(-1)  # S @ k
    delta = (v - proj) * beta.unsqueeze(-1)
    state = state + delta.unsqueeze(-1) * k.unsqueeze(-2)
    out = torch.matmul(state, q.unsqueeze(-1)).squeeze(-1)  # S @ q
    return out, state


def kda_recurrent_steps(state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor,
                        beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Token loop (reference for the chunked forms). q,k [B,L,H,K]; v [B,L,H,V]; decay [B,L,H,K]; beta [B,L,H];
    state [B,H,V,K]."""
    outs = []
    seq_len = q.shape[1]
    for t in range(seq_len):
        out, state = kda_recurrent_step(state, q[:, t], k[:, t], v[:, t], decay[:, t], beta[:, t])
        outs.append(out)
    return torch.stack(outs, dim=1), state


def _unit_lower_inverse(a: torch.Tensor) -> torch.Tensor:
    """(I + A)^-1 for a strictly lower triangular A [..., C, C] by recursive block doubling.

    The 1x1 diagonal blocks of I + A are 1; neighbouring blocks are merged level by level with the exact block
    formula ``[[T, 0], [-B (C T), B]]`` (T, B the inverses of the two diagonal blocks, C the coupling block), all
    pairs of a level in one batched matmul: log2(C) levels instead of C sequential substitution steps, no in-place
    writes. It is algebraically the same as forward substitution and as accurate (checked against a float64
    inverse, including nearly parallel keys with beta ~ 1), whereas a Neumann / repeated-squaring expansion loses
    all precision there. A size that is not a power of two is padded with identity rows.
    """
    size = a.shape[-1]
    padded = 1
    while padded < size:
        padded *= 2
    if padded != size:
        a = torch.nn.functional.pad(a, (0, padded - size, 0, padded - size))
    lead = a.shape[:-2]
    inv = a.new_ones(*lead, padded, 1, 1)  # inverses of the 1 x 1 diagonal blocks
    block, count = 1, padded
    while count > 1:
        blocks = a.reshape(*lead, count, block, count, block)
        # coupling between block 2p+1 (rows) and block 2p (columns), for every pair p
        coupling = torch.diagonal(blocks[..., 1::2, :, 0::2, :], dim1=-4, dim2=-2)  # [..., block, block, count/2]
        coupling = coupling.permute(*range(len(lead)), -1, -3, -2)  # [..., count/2, block, block]
        top, bottom = inv[..., 0::2, :, :], inv[..., 1::2, :, :]
        lower_left = -torch.matmul(bottom, torch.matmul(coupling, top))
        inv = torch.cat((torch.cat((top, torch.zeros_like(top)), dim=-1), torch.cat((lower_left, bottom), dim=-1)),
                        dim=-2)  # [..., count/2, 2 * block, 2 * block]
        block, count = 2 * block, count // 2
    return inv.reshape(*lead, padded, padded)[..., :size, :size]


def _unit_lower_inverse_dense(a: torch.Tensor) -> torch.Tensor:
    """``_unit_lower_inverse`` for a [N, C, C] batch using only dense ops (same arithmetic, same values).

    The HPU compiler rejects the strided view chains of the original (``torch.diagonal`` of a stepped slice,
    ``inv[..., 0::2]``), and the ``index_select`` gathers that replaced them ran as thousands of small gather
    kernels per layer. Here the coupling blocks come from a top-down pyramid: the quadrants of every diagonal
    block of size 2b are moved to a leading axis by one transposing copy, so the diagonal blocks of size b and
    the couplings are plain leading-dim selects of a contiguous tensor; the even / odd blocks of ``inv`` are
    picked the same way. Every tensor is contiguous and no tensor has rank above 5.
    """
    size = a.shape[-1]
    padded = 1
    while padded < size:
        padded *= 2
    if padded != size:
        a = torch.nn.functional.pad(a, (0, padded - size, 0, padded - size))
    batch = a.shape[0]

    def quadrants(x, count, block):  # [N, count, 2b, 2b] -> [4, N * count, b, b], quadrant (r, c) at 2 * r + c
        x = x.reshape(batch * count, 2, block, 2, block).permute(1, 3, 0, 2, 4).contiguous()
        return x.reshape(4, batch * count, block, block)

    # couplings[b]: block 2p+1 (rows) x block 2p (columns) of the size-b partition, [N, pairs, b, b]
    couplings = {}
    diag, block, count = a.reshape(batch, 1, padded, padded), padded // 2, 1
    while block >= 1:
        quad = quadrants(diag, count, block)
        couplings[block] = quad[2].reshape(batch, count, block, block)
        diag = torch.stack((quad[0], quad[3]), dim=1).reshape(batch, 2 * count, block, block)
        block, count = block // 2, 2 * count

    inv = a.new_ones(batch, padded, 1, 1)  # inverses of the 1 x 1 diagonal blocks
    block, count = 1, padded
    while count > 1:
        pairs = count // 2
        coupling = couplings[block]
        halves = inv.reshape(batch, pairs, 2, block, block).permute(2, 0, 1, 3, 4).contiguous()
        top, bottom = halves[0], halves[1]
        lower_left = -torch.matmul(bottom, torch.matmul(coupling, top))
        inv = torch.cat((torch.cat((top, torch.zeros_like(top)), dim=-1), torch.cat((lower_left, bottom), dim=-1)),
                        dim=-2)  # [N, count/2, 2 * block, 2 * block]
        block, count = 2 * block, pairs
    return inv.reshape(batch, padded, padded)[:, :size, :size]


def kda_chunked_steps(state: torch.Tensor,
                      q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      g: torch.Tensor,
                      beta: torch.Tensor,
                      chunk_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel (WY / UT form) delta rule, identical to ``kda_recurrent_steps`` (reference for the batched
    ``kda_chunked_steps_vec`` that serving runs).

    q,k [B,L,H,K] (normalized, q scaled); v [B,L,H,V]; g [B,L,H,K] LOG forget gate (<= 0); beta [B,L,H];
    state [B,H,V,K]. Chunks are processed one after another (state carried), tokens of a chunk in parallel.
    With G_i the cumulative log gate inside the chunk, u_i the pseudo values and S_0 the state before the chunk::

        (I + A) U = beta * (V - (K * exp(G)) S_0^T)       A_ij = beta_i sum_k k_i k_j exp(G_i - G_j)   (j < i)
        O = (Q * exp(G)) S_0^T + Aqk U                    Aqk_ij = sum_k q_i k_j exp(G_i - G_j)        (j <= i)
        S = S_0 * exp(G_C) + U^T (K * exp(G_C - G))

    Only exp(G_i - G_j) with j <= i (always <= 1) is evaluated, so nothing overflows however negative the gate
    gets; the pairwise [C, C, K] tensor exists for one chunk at a time. Tokens past the real length must have
    g = 0 and beta = 0 (exact no-ops); the sequence is zero-padded to a multiple of ``chunk_size`` the same way.
    """
    if chunk_size <= 0 or (chunk_size > 16 and chunk_size % 16 != 0):
        # keep the chunk (and with it every matmul of the inverse) tile friendly: multiples of 16 above 16
        raise ValueError(f"chunk_size must be positive and, above 16, a multiple of 16, got {chunk_size}")
    bsz, seq_len, heads, kdim = q.shape
    vdim = v.shape[-1]
    chunk = chunk_size
    pad = -seq_len % chunk
    if pad:
        zeros = lambda x: x.new_zeros(bsz, pad, *x.shape[2:])  # noqa: E731
        q, k, v = (torch.cat([x, zeros(x)], dim=1) for x in (q, k, v))
        g = torch.cat([g, zeros(g)], dim=1)
        beta = torch.cat([beta, zeros(beta)], dim=1)
    num_chunks = (seq_len + pad) // chunk

    def to_chunks(x):  # [B, N*C, H, D] -> [B, H, N, C, D]
        return x.reshape(bsz, num_chunks, chunk, heads, -1).permute(0, 3, 1, 2, 4)

    qc, kc, vc, gc = to_chunks(q), to_chunks(k), to_chunks(v), to_chunks(g)
    bc = beta.reshape(bsz, num_chunks, chunk, heads).permute(0, 3, 1, 2).unsqueeze(-1)  # [B,H,N,C,1]
    cum = torch.cumsum(gc, dim=3)  # G, [B,H,N,C,K]
    last = cum[:, :, :, -1:, :]  # G_C
    q_gamma = qc * torch.exp(cum)
    k_gamma = kc * torch.exp(cum)
    k_tail = kc * torch.exp(last - cum)
    gamma_last = torch.exp(last.squeeze(3))  # [B,H,N,K]

    strict = torch.tril(torch.ones(chunk, chunk, dtype=q.dtype, device=q.device), -1)
    lower = torch.tril(torch.ones(chunk, chunk, dtype=q.dtype, device=q.device))

    outs = []
    for n in range(num_chunks):
        g_n, k_n, q_n = cum[:, :, n], kc[:, :, n], qc[:, :, n]  # [B,H,C,K]
        # exp(G_i - G_j) for j <= i; the clamp keeps the masked-out upper triangle finite (0 * inf would be NaN)
        decay = torch.exp(torch.clamp(g_n.unsqueeze(3) - g_n.unsqueeze(2), max=0.0))  # [B,H,C,C,K]
        kk = k_n.unsqueeze(3) * k_n.unsqueeze(2)
        qk = q_n.unsqueeze(3) * k_n.unsqueeze(2)
        a_mat = (kk * decay).sum(-1) * strict * bc[:, :, n]  # beta_i on the row
        aqk = (qk * decay).sum(-1) * lower
        inv = _unit_lower_inverse(a_mat)
        s_t = state.transpose(-1, -2)  # [B,H,K,V]
        rhs = bc[:, :, n] * (vc[:, :, n] - torch.matmul(k_gamma[:, :, n], s_t))
        pseudo = torch.matmul(inv, rhs)  # U, [B,H,C,V]
        out_n = torch.matmul(q_gamma[:, :, n], s_t) + torch.matmul(aqk, pseudo)
        state = state * gamma_last[:, :, n].unsqueeze(-2) + torch.matmul(pseudo.transpose(-1, -2), k_tail[:, :, n])
        outs.append(out_n)
    out = torch.stack(outs, dim=2)  # [B,H,N,C,V]
    out = out.permute(0, 2, 3, 1, 4).reshape(bsz, num_chunks * chunk, heads, vdim)
    return out[:, :seq_len], state


def parse_prefill_settings(value: str) -> tuple[int, int]:
    """Parse HPU_KIMI_KDA_PREFILL_CHUNK ``"<chunk_size>,<vec_chunks>"``."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"HPU_KIMI_KDA_PREFILL_CHUNK must be '<chunk_size>,<vec_chunks>', got {value!r}")
    chunk_size, vec_chunks = int(parts[0]), int(parts[1])
    if chunk_size <= 0 or (chunk_size > 16 and chunk_size % 16 != 0):
        raise ValueError(f"HPU_KIMI_KDA_PREFILL_CHUNK chunk_size must be positive and, above 16, a multiple of 16, "
                         f"got {chunk_size}")
    return chunk_size, vec_chunks


def parse_intra_sub_chunk(value: str) -> int:
    """Parse HPU_KIMI_KDA_INTRA_SUB_CHUNK (empty or 0 = off)."""
    value = value.strip()
    if not value:
        return 0
    if not value.isdigit():
        raise ValueError(f"HPU_KIMI_KDA_INTRA_SUB_CHUNK must be a non-negative integer, got {value!r}")
    return int(value)


def effective_intra_sub_chunk(chunk_size: int, sub_chunk: int) -> int:
    """``sub_chunk`` if it splits ``chunk_size`` into at least two equal blocks, else 0 (full pairwise path)."""
    return sub_chunk if 0 < sub_chunk < chunk_size and chunk_size % sub_chunk == 0 else 0


def _pairwise_terms(cum: torch.Tensor, kc: torch.Tensor, qc: torch.Tensor):
    """Raw A and Aqk ``sum_k x_i k_j exp(G_i - G_j)`` of [N,C,K] inputs from the full [N,C,C,K] pairwise decay."""
    # exp(G_i - G_j) for j <= i; the clamp keeps the masked-out upper triangle finite (0 * inf would be NaN)
    decay = torch.exp(torch.clamp(cum.unsqueeze(-2) - cum.unsqueeze(-3), max=0.0))  # [N,C(i),C(j),K]
    k_decayed = decay * kc.unsqueeze(-3)  # k_j * exp(G_i - G_j)
    # contract with k_i (A) and q_i (Aqk) over K elementwise: as matmuls these are width-1 batched GEMVs
    a_raw = (k_decayed * kc.unsqueeze(-2)).sum(-1)  # [N,C(i),C(j)]
    aqk_raw = (k_decayed * qc.unsqueeze(-2)).sum(-1)
    return a_raw, aqk_raw


def _pairwise_terms_blocked(cum: torch.Tensor, kc: torch.Tensor, qc: torch.Tensor, sub_chunk: int):
    """``_pairwise_terms`` with the chunk split into M sub-chunks of ``sub_chunk`` rows.

    Diagonal blocks use the pairwise form. Block (I, J), J < I, factors through r_J = G at the last row of J:
    ``exp(G_i - G_j) = exp(G_i - r_J) exp(r_J - G_j)``, both exponents <= 0 since G is non-increasing.
    Blocks J > I are zero.
    """
    batch, chunk, kdim = cum.shape
    m = chunk // sub_chunk
    diags = _pairwise_terms(*(x.reshape(batch * m, sub_chunk, kdim) for x in (cum, kc, qc)))
    cum_b = cum.reshape(batch, m, sub_chunk, kdim)
    last = cum_b[:, :, -1:, :]  # r_J, [N,M,1,K]
    right_t = (kc.reshape(batch, m, sub_chunk, kdim) * torch.exp(last - cum_b)).transpose(-1, -2)  # [N,M,K,S]
    # all M block columns at once; rows of block J and above are replaced below
    left_decay = torch.exp(torch.clamp(cum.unsqueeze(1) - last, max=0.0))  # [N,M(J),C,K]
    blk = torch.arange(m, device=cum.device)
    on_diag = (blk.unsqueeze(1) == blk).reshape(m, 1, m, 1)
    below = (blk.unsqueeze(1) > blk).reshape(m, 1, m, 1)
    outs = []
    for x, diag in zip((kc, qc), diags):
        off = torch.matmul(x.unsqueeze(1) * left_decay, right_t)  # [N,M(J),C,S]
        off = off.transpose(1, 2).reshape(batch, m, sub_chunk, m, sub_chunk)  # [N,M(I),S,M(J),S]
        diag = diag.reshape(batch, m, sub_chunk, 1, sub_chunk)
        full = torch.where(on_diag, diag, torch.where(below, off, 0.0))
        outs.append(full.reshape(batch, chunk, chunk))
    return outs[0], outs[1]


def _intra_chunk_terms(cum: torch.Tensor,
                       kc: torch.Tensor,
                       qc: torch.Tensor,
                       vc: torch.Tensor,
                       bc: torch.Tensor,
                       k_gamma: torch.Tensor,
                       q_gamma: torch.Tensor,
                       strict: torch.Tensor,
                       lower: torch.Tensor,
                       sub_chunk: int = 0):
    """State-independent terms of a group of chunks (all inputs [G,B,H,C,*], bc [G,B,H,C,1]).

    With U = Uv - Wk S_0^T the pseudo values of the chunk, returns ``(uv, wk, q_eff, o_intra)`` with
    ``Uv = (I + A)^-1 (beta V)``, ``Wk = (I + A)^-1 (beta K exp(G))``, so that the chunk output is
    ``O = q_eff S_0^T + o_intra`` (``q_eff = Q exp(G) - Aqk Wk``, ``o_intra = Aqk Uv``).
    """
    # Fold B, H and the chunk group into one batch dimension: the HPU compiler rejects (strided_view, Invalid
    # argument) the rank 6/7 views that the pairwise tensors and the block inverse would otherwise create.
    lead = cum.shape[:3]

    def fold(x):  # [B,H,G,C,D] -> [B*H*G,C,D]; reshape copies the non-contiguous slices of a chunk group
        return x.reshape(lead[0] * lead[1] * lead[2], *x.shape[3:])

    cum, kc, qc, vc, bc, k_gamma, q_gamma = (fold(x) for x in (cum, kc, qc, vc, bc, k_gamma, q_gamma))
    chunk = strict.shape[-1]
    if effective_intra_sub_chunk(chunk, sub_chunk):
        a_raw, aqk_raw = _pairwise_terms_blocked(cum, kc, qc, sub_chunk)
    else:
        a_raw, aqk_raw = _pairwise_terms(cum, kc, qc)
    a_mat = a_raw * strict * bc  # beta_i on the row
    aqk = aqk_raw * lower
    inv = _unit_lower_inverse_dense(a_mat)
    uv = torch.matmul(inv, bc * vc)
    wk = torch.matmul(inv, bc * k_gamma)
    q_eff = q_gamma - torch.matmul(aqk, wk)
    o_intra = torch.matmul(aqk, uv)
    return tuple(x.reshape(*lead, *x.shape[1:]) for x in (uv, wk, q_eff, o_intra))


_SCAN_BLOCK = 8  # chunks per block of the state recurrence scan


def _chunk_start_states(s_t: torch.Tensor, tw: torch.Tensor, tu: torch.Tensor,
                        gamma_last: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-initial states [N,B,H,K,V] and the final state of ``s <- s * gl + Tu - Tw s`` from s_t [B,H,K,V].

    The affine chunk maps ``s -> A s + Tu`` with ``A = diag(gl) - Tw`` are composed within blocks of
    ``_SCAN_BLOCK`` chunks, batched over the blocks, so only the block-initial states are stepped sequentially;
    the carries are added by ``baddbmm``.
    """
    num_chunks, bsz, heads, kdim, vdim = tu.shape
    scan_block = _SCAN_BLOCK
    # not min() and no division by num_chunks: no Min / symbolic divisor in the traced shapes
    if scan_block < num_chunks:
        block, num_blocks = scan_block, (num_chunks + scan_block - 1) // scan_block
    else:
        block, num_blocks = num_chunks, 1
    rows = num_blocks * bsz * heads  # explicit sizes: no reshape(-1) divisions in the traced shapes
    eye = torch.eye(kdim, dtype=tw.dtype, device=tw.device)

    # chunk j * block + i goes to row i * num_blocks + j: one gather instead of pad + transpose copy; the padded
    # tail repeats the last chunk and only feeds discarded starts and the unused carry of the last block
    order = torch.arange(num_blocks * block, device=tw.device).reshape(num_blocks, block).t()
    order = order.clamp_max(num_chunks - 1).reshape(num_blocks * block)

    # [block, num_blocks * B*H, K, D]: the position in the block leads, so every step reads a leading-dim slice
    def to_blocks(x):
        return x.index_select(0, order).reshape(block, rows, kdim, x.shape[-1])

    # maps A = diag(gl) - Tw [N,B,H,K,K] built inline so the natural-order buffer is freed after the gather
    a_blk, b_blk = to_blocks(eye * gamma_last - tw), to_blocks(tu)
    # prefix maps within the blocks: P_i = A_i P_{i-1}, c_i = A_i c_{i-1} + b_i
    p, c = [a_blk[0]], [b_blk[0]]
    for a_i, b_i in zip(a_blk[1:], b_blk[1:]):
        p.append(torch.bmm(a_i, p[-1]))
        c.append(torch.baddbmm(b_i, a_i, c[-1]))
    x = s_t.reshape(bsz * heads, kdim, vdim)
    xs = [x]
    for p_j, c_j in zip(p[-1].reshape(num_blocks, bsz * heads, kdim, kdim)[:-1],
                        c[-1].reshape(num_blocks, bsz * heads, kdim, vdim)[:-1]):
        x = torch.baddbmm(c_j, p_j, x)
        xs.append(x)
    xs = torch.stack(xs)  # [num_blocks, B*H, K, V]
    xs_rows = xs.flatten(0, 1)
    starts = [xs] + [
        torch.baddbmm(c_i, p_i, xs_rows).reshape(num_blocks, bsz * heads, kdim, vdim)
        for p_i, c_i in zip(p[:-1], c[:-1])
    ]
    starts = torch.stack(starts, 1)
    starts = starts.reshape(num_blocks * block, bsz, heads, kdim, vdim)[:num_chunks]
    # final state: one step from the last chunk start (the padded maps are not identities)
    a_last = eye * gamma_last[-1] - tw[-1]
    s_t = torch.baddbmm(tu[-1].reshape(bsz * heads, kdim, vdim), a_last.reshape(bsz * heads, kdim, kdim),
                        starts[-1].reshape(bsz * heads, kdim, vdim))
    return starts, s_t.reshape(bsz, heads, kdim, vdim)


def kda_chunked_steps_vec(state: torch.Tensor,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          g: torch.Tensor,
                          beta: torch.Tensor,
                          chunk_size: int = 64,
                          vec_chunks: int = 16,
                          sub_chunk: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel (WY / UT form) delta rule, equal to ``kda_recurrent_steps``.

    q,k [B,L,H,K] (normalized, q scaled); v [B,L,H,V]; g [B,L,H,K] LOG forget gate (<= 0); beta [B,L,H];
    state [B,H,V,K]. Tokens past the real length must have g = 0 and beta = 0 (exact no-ops); the sequence is
    zero-padded to a multiple of ``chunk_size`` the same way. With G_i the cumulative log gate inside the chunk,
    U the pseudo values and S_0 the state before the chunk::

        (I + A) U = beta * (V - (K * exp(G)) S_0^T)       A_ij = beta_i sum_k k_i k_j exp(G_i - G_j)   (j < i)
        O = (Q * exp(G)) S_0^T + Aqk U                    Aqk_ij = sum_k q_i k_j exp(G_i - G_j)        (j <= i)
        S = S_0 * exp(G_C) + U^T (K * exp(G_C - G))

    Only exp(G_i - G_j) with j <= i (always <= 1) is evaluated, so nothing overflows however negative the gate
    gets. Everything that does not depend on the state is batched over the chunks. The pairwise decay, A, Aqk,
    the exact (I + A)^-1 and the terms derived from it (``_intra_chunk_terms``) are computed for ``vec_chunks``
    chunks at a time (0 = all at once; the pairwise tensor is [vec_chunks, C, C, K] per head), so only the state
    recurrence stays sequential, kept in the transposed layout s = S^T [K, V] (5 ops per chunk)::

        U   = Uv - Wk s                  s <- s * exp(G_C)^T + (K exp(G_C - G))^T U

    and the outputs of all chunks come from the stacked chunk-initial states in one matmul at the end; the
    chunk transitions are composed with a scan (see ``_chunk_start_states``). ``sub_chunk`` > 0 computes A and Aqk
    blockwise (see ``_intra_chunk_terms``).
    """
    if chunk_size <= 0 or (chunk_size > 16 and chunk_size % 16 != 0):
        raise ValueError(f"chunk_size must be positive and, above 16, a multiple of 16, got {chunk_size}")
    bsz, seq_len, heads, kdim = q.shape
    vdim = v.shape[-1]
    chunk = chunk_size
    # num_chunks is one floor division of the (dynamic) sequence length by a constant, and the padded length is
    # written as num_chunks * chunk (not seq_len + (-seq_len % chunk)): the traced shapes then stay products of
    # plain dynamic dims and ints, with no Mod and no division by a symbolic size, which the HPU compiler's
    # symbolic-size evaluation does not handle
    num_chunks = (seq_len + chunk - 1) // chunk
    pad = num_chunks * chunk - seq_len
    # always padded (also by 0): the padded length must be the product num_chunks * chunk, not seq_len
    q, k, v, g = (torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v, g))
    beta = torch.nn.functional.pad(beta, (0, 0, 0, pad))
    group = vec_chunks if 0 < vec_chunks < num_chunks else num_chunks  # not min(): no Min in the traced shapes

    # chunk axis leading and contiguous: chunk groups and the recurrence steps are dense leading-dim slices
    def to_chunks(x):  # [B, N*C, H, D] -> [N, B, H, C, D]
        return x.reshape(bsz, num_chunks, chunk, heads, -1).permute(1, 0, 3, 2, 4).contiguous()

    qc, kc, vc, gc = to_chunks(q), to_chunks(k), to_chunks(v), to_chunks(g)
    bc = beta.reshape(bsz, num_chunks, chunk, heads).permute(1, 0, 3, 2).contiguous().unsqueeze(-1)  # [N,B,H,C,1]
    cum = torch.cumsum(gc, dim=3)  # G, [N,B,H,C,K]
    last = cum[:, :, :, -1:, :]  # G_C
    q_gamma = qc * torch.exp(cum)
    k_gamma = kc * torch.exp(cum)
    k_tail_t = (kc * torch.exp(last - cum)).transpose(-1, -2)  # [B,H,N,K,C]
    gamma_last = torch.exp(last.squeeze(3)).unsqueeze(-1)  # [B,H,N,K,1]

    strict = torch.tril(torch.ones(chunk, chunk, dtype=q.dtype, device=q.device), -1)
    lower = torch.tril(torch.ones(chunk, chunk, dtype=q.dtype, device=q.device))

    terms = []
    for start in range(0, num_chunks, group):
        sl = slice(start, start + group)
        terms.append(
            _intra_chunk_terms(cum[sl], kc[sl], qc[sl], vc[sl], bc[sl], k_gamma[sl], q_gamma[sl], strict, lower,
                               sub_chunk))
    uv, wk, q_eff, o_intra = (torch.cat(parts) if len(parts) > 1 else parts[0] for parts in zip(*terms))

    tw = torch.matmul(k_tail_t, wk)  # [N,B,H,K,K]
    tu = torch.matmul(k_tail_t, uv)  # [N,B,H,K,V]

    starts, s_t = _chunk_start_states(state.transpose(-1, -2), tw, tu, gamma_last)
    out = torch.matmul(q_eff, starts) + o_intra  # [N,B,H,C,V]
    out = out.permute(1, 0, 3, 2, 4).reshape(bsz, num_chunks * chunk, heads, vdim)
    # exact shape and a dense layout: this output crosses a graph break (storage offset 0, no padded tail);
    # clone, since with B = 1 the slice of the reshaped buffer is already contiguous
    return out[:, :seq_len].clone(memory_format=torch.contiguous_format), s_t.transpose(-1, -2)


def _prepare_qkv_gates(conv_out: torch.Tensor, raw_g: torch.Tensor, raw_beta: torch.Tensor, A_log: torch.Tensor,
                       dt_bias: torch.Tensor, lower_bound: float | None, num_heads: int, head_dim: int,
                       token_mask: torch.Tensor | None):
    """conv_out [B,L,3P] (after silu), raw_g [B,L,P], raw_beta [B,L,H] -> q,k,v,g,beta (fp32, [B,L,H,*]).

    g is the LOG of the forget gate (<= 0)."""
    bsz, seq_len, _ = conv_out.shape
    proj = num_heads * head_dim
    q, k, v = conv_out.split(proj, dim=-1)
    q = q.reshape(bsz, seq_len, num_heads, head_dim)
    k = k.reshape(bsz, seq_len, num_heads, head_dim)
    v = v.reshape(bsz, seq_len, num_heads, head_dim)
    q = l2norm(q) * (head_dim**-0.5)
    k = l2norm(k)
    g = kda_gate(raw_g.reshape(bsz, seq_len, num_heads, head_dim), A_log, dt_bias.reshape(num_heads, head_dim),
                 lower_bound)
    beta = torch.sigmoid(raw_beta.float())
    if token_mask is not None:  # padded tokens: decay 1, beta 0 -> exact no-op on the state
        m = token_mask.reshape(bsz, seq_len, 1).float()
        g = g * m.unsqueeze(-1)
        beta = beta * m
    return q, k, v, g, beta


# ---------------------------------------------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------------------------------------------
def kda_prefill_step(mixed_qkv: torch.Tensor,
                     raw_g: torch.Tensor,
                     raw_beta: torch.Tensor,
                     conv_weight: torch.Tensor,
                     conv_bias: torch.Tensor | None,
                     A_log: torch.Tensor,
                     dt_bias: torch.Tensor,
                     lower_bound: float | None,
                     conv_cache: torch.Tensor,
                     ssm_cache: torch.Tensor,
                     state_idx: torch.Tensor,
                     has_initial_state: torch.Tensor | None,
                     token_mask: torch.Tensor | None,
                     num_heads: int,
                     head_dim: int,
                     chunk_size: int = 64,
                     vec_chunks: int = 16,
                     sub_chunk: int = 0):
    """Out-of-place prefill: only READS the caches, returns ``(o, slots, conv_rows, ssm_rows)`` like
    ``kda_decode_step`` (store the rows with ``write_kda_decode_state``), so it is traceable by torch.compile.

    mixed_qkv [B,L,3P] (pre-conv), raw_g [B,L,P], raw_beta [B,L,H], token_mask [B,L] (1 = real token, padding at
    the end of each sequence) or None, state_idx [B] (-1 = padded row), has_initial_state [B] or None.
    o is [B,L,H,V] (fp32). The delta rule is ``kda_chunked_steps_vec`` with the given ``chunk_size`` and
    ``vec_chunks``.
    """
    bsz, seq_len, channels = mixed_qkv.shape
    width = conv_weight.shape[-1]
    width_m1 = width - 1
    device = mixed_qkv.device
    slots = _slots(state_idx, conv_cache)

    use_init = (state_idx >= 0).float()
    if has_initial_state is not None:
        use_init = use_init * has_initial_state.reshape(-1).float()[:bsz]
    use_init = use_init.reshape(bsz, 1, 1)

    if token_mask is None:
        lens = torch.full((bsz, ), seq_len, dtype=torch.long, device=device)
    else:
        lens = token_mask.reshape(bsz, seq_len).float().sum(dim=1).round().long()

    # --- causal depthwise conv + silu ---
    init = read_conv_state(conv_cache, slots, channels, width_m1) * use_init  # [B, W-1, C]
    seq = torch.cat([init, mixed_qkv.float()], dim=1)  # [B, W-1+L, C]
    w = conv_weight.reshape(channels, width).float().t().contiguous()  # [W, C]: taps are leading-dim selects
    conv_out = torch.zeros(bsz, seq_len, channels, dtype=torch.float32, device=device)
    for i in range(width):
        conv_out = conv_out + seq[:, i:i + seq_len, :] * w[i]
    if conv_bias is not None:
        conv_out = conv_out + conv_bias.float()
    conv_out = torch.nn.functional.silu(conv_out)
    # final conv state = last W-1 REAL inputs of each sequence
    gather_idx = (lens.unsqueeze(1) + torch.arange(width_m1, device=device).unsqueeze(0)).unsqueeze(-1)
    new_conv = torch.gather(seq, 1, gather_idx.expand(-1, -1, channels))
    if _conv_is_dim_first(conv_cache, channels, width_m1):
        new_conv = new_conv.transpose(-1, -2)
    conv_rows = new_conv.to(dtype=conv_cache.dtype).contiguous()

    # --- delta rule ---
    q, k, v, g, beta = _prepare_qkv_gates(conv_out, raw_g, raw_beta, A_log, dt_bias, lower_bound, num_heads, head_dim,
                                          token_mask)
    state = read_ssm_state(ssm_cache, slots) * use_init.reshape(bsz, 1, 1, 1)
    out, state = kda_chunked_steps_vec(state,
                                       q,
                                       k,
                                       v,
                                       g,
                                       beta,
                                       chunk_size=chunk_size,
                                       vec_chunks=vec_chunks,
                                       sub_chunk=sub_chunk)
    ssm_rows = state.to(dtype=ssm_cache.dtype).contiguous()
    return out, slots, conv_rows, ssm_rows


def kda_decode_step(mixed_qkv: torch.Tensor, raw_g: torch.Tensor, raw_beta: torch.Tensor, conv_weight: torch.Tensor,
                    conv_bias: torch.Tensor | None, A_log: torch.Tensor, dt_bias: torch.Tensor,
                    lower_bound: float | None, conv_cache: torch.Tensor, ssm_cache: torch.Tensor,
                    state_idx: torch.Tensor, num_heads: int, head_dim: int):
    """Out-of-place decode step: only READS the caches, returns ``(o, slots, conv_rows, ssm_rows)``.

    ``o`` [B,H,V] fp32; ``conv_rows`` / ``ssm_rows`` are the new cache rows of ``slots``, already in the layout and
    dtype of the caches, so ``write_kda_decode_state`` is just two ``index_copy_``. Traceable by torch.compile
    (no in-place writes to graph inputs, which HPU compile can silently drop).
    """
    bsz, channels = mixed_qkv.shape
    width = conv_weight.shape[-1]
    slots = _slots(state_idx, conv_cache)

    state_c = read_conv_state(conv_cache, slots, channels, width - 1)  # [B, W-1, C]
    seq = torch.cat([state_c, mixed_qkv.float().unsqueeze(1)], dim=1)  # [B, W, C]
    w = conv_weight.reshape(channels, width).float()
    conv_out = (seq * w.t().unsqueeze(0)).sum(dim=1)
    if conv_bias is not None:
        conv_out = conv_out + conv_bias.float()
    conv_out = torch.nn.functional.silu(conv_out)
    conv_rows = seq[:, 1:, :]
    if _conv_is_dim_first(conv_cache, channels, width - 1):
        conv_rows = conv_rows.transpose(-1, -2)
    conv_rows = conv_rows.to(dtype=conv_cache.dtype).contiguous()

    q, k, v, g, beta = _prepare_qkv_gates(conv_out.unsqueeze(1), raw_g.unsqueeze(1), raw_beta.unsqueeze(1), A_log,
                                          dt_bias, lower_bound, num_heads, head_dim, None)
    state = read_ssm_state(ssm_cache, slots)
    out, state = kda_recurrent_step(state, q[:, 0], k[:, 0], v[:, 0], torch.exp(g[:, 0]), beta[:, 0])
    ssm_rows = state.to(dtype=ssm_cache.dtype).contiguous()
    return out, slots, conv_rows, ssm_rows


def write_kda_decode_state(conv_cache: torch.Tensor, ssm_cache: torch.Tensor, slots: torch.Tensor,
                           conv_rows: torch.Tensor, ssm_rows: torch.Tensor) -> None:
    """Stores the rows returned by ``kda_decode_step`` / ``kda_prefill_step`` (their only in-place part)."""
    conv_cache.index_copy_(0, slots, conv_rows)
    ssm_cache.index_copy_(0, slots, ssm_rows)


# ---------------------------------------------------------------------------------------------------------------
# KV cache group tagging
# ---------------------------------------------------------------------------------------------------------------
def tag_kda_cache_groups(named_modules, kv_cache_groups, is_mamba_group: Callable, make_index: Callable):
    """Set ``cache_group_idx`` on every KDA module from the KV cache group that lists its layer name.

    KDA modules are recognised by their ``in_proj_qkvgfab`` projection and matched by ``module.prefix`` (the key
    they registered in the static forward context, which is what ``kv_cache_groups[i].layer_names`` holds);
    the module path is only a fallback. Returns ``(tagged, untagged, unmatched)``: layers tagged per group id,
    KDA module names without a mamba group, and mamba group layer names without a KDA module.
    """
    layer_to_group: dict[str, int] = {}
    for group_id, group in enumerate(kv_cache_groups):
        if is_mamba_group(group):
            for layer_name in group.layer_names:
                layer_to_group[layer_name] = group_id
    tagged: dict[int, int] = {}
    untagged: list[str] = []
    seen: set[str] = set()
    for module_name, module in named_modules:
        if not hasattr(module, "in_proj_qkvgfab"):
            continue
        prefix = getattr(module, "prefix", None) or module_name
        group_id = layer_to_group.get(prefix, layer_to_group.get(module_name, -1))
        if group_id < 0:
            untagged.append(prefix)
            continue
        module.cache_group_idx = make_index(group_id)
        tagged[group_id] = tagged.get(group_id, 0) + 1
        seen.add(prefix if prefix in layer_to_group else module_name)
    unmatched = [name for name in layer_to_group if name not in seen]
    return tagged, untagged, unmatched


# ---------------------------------------------------------------------------------------------------------------
# Kimi-K3 attention residual (attn_res), pure PyTorch twin of vllm/models/kimi_k3/amd/ops/attn_res.py
# ---------------------------------------------------------------------------------------------------------------
def attn_res_functional(prefix: torch.Tensor, delta: torch.Tensor | None, blocks: torch.Tensor,
                        norm_weight: torch.Tensor, qk_weight: torch.Tensor, output_norm_weight: torch.Tensor | None,
                        num_blocks: int, block_write_idx: int, eps: float,
                        output_norm_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Out-of-place ``attn_res``: returns ``(new_prefix, new_blocks, mixed)``, no input is modified.

    Same numbers as the Triton kernel up to fp32 reduction order (see ``attn_res_torch``, which adds its in-place
    side effects). Needed in torch.compile mode, where regional compilation makes the prefix and the block bank
    graph inputs and HPU compile can silently drop in-place writes to them. ``new_prefix`` is ``prefix`` itself
    without a delta and ``new_blocks`` is ``blocks`` itself without a block write (no copy in those cases). All
    shapes are static.
    """
    new_prefix, new_blocks, _, mixed = attn_res_functional_cached(prefix, delta, blocks, None, norm_weight, qk_weight,
                                                                  output_norm_weight, num_blocks, block_write_idx, eps,
                                                                  output_norm_eps)
    return new_prefix, new_blocks, mixed


def attn_res_bank_mean_square(blocks: torch.Tensor, num_valid_blocks: int) -> torch.Tensor:
    """fp32 ``mean(v^2)`` of the first ``num_valid_blocks`` stored blocks of ``blocks`` [M, R, D]; zeros after."""
    num_tokens, num_bank, hidden = blocks.shape
    valid = torch.square(blocks[:, :num_valid_blocks].float()).sum(dim=-1) / hidden
    return torch.cat((valid, valid.new_zeros(num_tokens, num_bank - num_valid_blocks)), dim=1)


def attn_res_functional_cached(
        prefix: torch.Tensor, delta: torch.Tensor | None, blocks: torch.Tensor, bank_ms: torch.Tensor | None,
        norm_weight: torch.Tensor, qk_weight: torch.Tensor, output_norm_weight: torch.Tensor | None, num_blocks: int,
        block_write_idx: int, eps: float,
        output_norm_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """``attn_res_functional`` that also takes and returns ``bank_ms`` [M, R], the fp32 ``mean(v^2)`` per stored block.

    The bank read uses ``bank_ms`` instead of a pass over the bank; a block write also writes its mean square.
    Same numbers as recomputing it from the stored (rounded) block. ``bank_ms=None``: recomputed per read.
    """
    num_tokens, hidden = prefix.shape
    if delta is not None and delta.shape != prefix.shape:
        raise ValueError(f"delta {tuple(delta.shape)} does not match prefix {tuple(prefix.shape)}")
    if blocks.dim() != 3 or blocks.shape[0] != num_tokens or blocks.shape[-1] != hidden:
        raise ValueError(f"block bank {tuple(blocks.shape)} does not match prefix {tuple(prefix.shape)}")
    if bank_ms is not None and bank_ms.shape != blocks.shape[:2]:
        raise ValueError(f"bank mean squares {tuple(bank_ms.shape)} do not match the bank {tuple(blocks.shape)}")

    if delta is not None:
        new_prefix = (prefix.float() + delta.float()).to(prefix.dtype)
        updated = new_prefix.float()
    else:
        new_prefix = prefix
        updated = prefix.float()
    own_ms = None
    new_blocks, new_ms = blocks, bank_ms
    if block_write_idx >= 0:
        written = updated.to(blocks.dtype)
        new_blocks = torch.cat((blocks[:, :block_write_idx], written.unsqueeze(1), blocks[:, block_write_idx + 1:]),
                               dim=1)
        if bank_ms is not None:
            own_ms = torch.square(updated).sum(dim=-1, keepdim=True) / hidden  # [M, 1]
            written_ms = own_ms if blocks.dtype == prefix.dtype else torch.square(written.float()).sum(
                dim=-1, keepdim=True) / hidden
            new_ms = torch.cat((bank_ms[:, :block_write_idx], written_ms, bank_ms[:, block_write_idx + 1:]), dim=1)

    if num_blocks == 0:
        mixed = updated
    else:
        # The stored blocks are read in their own dtype (no cat with the prefix); the updated prefix is a
        # separate [M, D] term and reductions accumulate in fp32.
        bank = new_blocks[:, :num_blocks]  # [M, S, D]
        qk = norm_weight.float() * qk_weight.float()  # [D]
        bank_sq = torch.square(bank.float()).sum(dim=-1) / hidden if new_ms is None else new_ms[:, :num_blocks]
        bank_rstd = torch.rsqrt(bank_sq + eps)  # [M, S]
        bank_logits = (bank.float() * qk).sum(dim=-1) * bank_rstd
        if own_ms is None:
            own_ms = torch.square(updated).sum(dim=-1, keepdim=True) / hidden
        own_rstd = torch.rsqrt(own_ms + eps)  # [M, 1]
        own_logit = (updated * qk).sum(dim=-1, keepdim=True) * own_rstd
        probs = torch.softmax(torch.cat((bank_logits, own_logit), dim=-1), dim=-1)  # [M, S + 1]
        # fp32 weighted sum over the stored blocks as elementwise ops (no M=1 bmm), in the kernel's order
        mixed = probs[:, :1] * bank[:, 0].float()
        for r in range(1, num_blocks):
            mixed = mixed + probs[:, r:r + 1] * bank[:, r].float()
        mixed = mixed + probs[:, num_blocks:] * updated
    if output_norm_weight is not None:
        mixed = mixed * torch.rsqrt(mixed.pow(2).mean(dim=-1, keepdim=True) + output_norm_eps)
        mixed = mixed * output_norm_weight.float()
    return new_prefix, new_blocks, new_ms, mixed.to(prefix.dtype)


_compiled_attn_res_functional: Callable | None = None


def _hpu_lazy_mode() -> bool:
    try:
        import habana_frameworks.torch.utils.internal as htu
    except ImportError:
        return False
    return htu.is_lazy()


def _eager_attn_res_functional(device_type: str) -> Callable:
    """``attn_res_functional`` for a call outside a compiled region.

    On HPU in torch.compile mode it runs as its own compiled graph: eagerly, every op of the per-block mix is a
    separate recipe.
    """
    global _compiled_attn_res_functional
    if device_type != "hpu" or torch.compiler.is_compiling() or _hpu_lazy_mode():
        return attn_res_functional
    if _compiled_attn_res_functional is None:
        _compiled_attn_res_functional = torch.compile(attn_res_functional, backend="hpu_backend", dynamic=False)
    return _compiled_attn_res_functional


def attn_res_torch(prefix: torch.Tensor,
                   delta: torch.Tensor | None,
                   blocks: torch.Tensor,
                   norm_weight: torch.Tensor,
                   qk_weight: torch.Tensor,
                   output_norm_weight: torch.Tensor | None,
                   num_blocks: int,
                   block_write_idx: int,
                   eps: float,
                   output_norm_eps: float,
                   quant_dtype=None) -> torch.Tensor:
    """Same results AND side effects as the Triton ``attn_res`` kernel.

    Per row, in this order:
      1. with a delta: ``prefix <- round_to_prefix_dtype(prefix + delta)`` is stored back INTO ``prefix`` (the
         callers return that tensor as the running prefix sum and rely on it);
      2. ``block_write_idx >= 0``: the updated prefix is stored into ``blocks[:, block_write_idx]``;
      3. softmax-weighted mix (fp32) of the ``num_blocks`` stored blocks and the updated prefix, with logits
         ``sum(v * norm_weight * qk_weight) * rsqrt(mean(v^2) + eps)``; no sources -> the prefix itself;
      4. optional RMSNorm of the mix with ``output_norm_weight``; the result has the dtype of ``prefix``.
    ``prefix`` [M, D] and ``blocks`` [M, R, D] are mutated in place; a shape mismatch raises (the old fallback
    silently replaced the block bank with zeros). Compiled code should use ``attn_res_functional`` instead.
    """
    del quant_dtype
    if prefix.dim() == 3:  # [B, S, D]: flatten with views so that the in-place update reaches the caller's tensor
        flat_delta = None if delta is None else delta.view(-1, delta.shape[-1])
        out = attn_res_torch(prefix.view(-1, prefix.shape[-1]), flat_delta, blocks, norm_weight, qk_weight,
                             output_norm_weight, num_blocks, block_write_idx, eps, output_norm_eps)
        return out.view(prefix.shape)
    if prefix.dim() != 2:
        raise ValueError(f"attn_res expects a 2D or 3D prefix, got {tuple(prefix.shape)}")
    functional = _eager_attn_res_functional(prefix.device.type)
    new_prefix, new_blocks, mixed = functional(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight,
                                               num_blocks, block_write_idx, eps, output_norm_eps)
    if delta is not None:
        prefix.copy_(new_prefix)  # the kernel's in-place side effect
    if block_write_idx >= 0:
        blocks[:, block_write_idx] = new_blocks[:, block_write_idx]
    return mixed


def forward_attn_residual_functional(self, positions, hidden_states, block_residual, prefix_delta):
    """``KimiLinearDecoderLayer.forward_attn_residual`` without in-place updates of its inputs.

    Upstream relies on ``attn_res`` mutating ``prefix_sum`` and ``block_residual`` across layer boundaries; with
    one compiled graph per decoder layer those are graph inputs. Here the updated tensors are rebound and returned:
    ``(prefix_sum, block_residual, mlp_out)`` like upstream, but ``block_residual`` is a new tensor in block-write
    layers (the bank copy costs ``tokens * bank * hidden`` elements per write layer).
    """
    blocks, bank_ms = block_residual if isinstance(block_residual, tuple) else (block_residual, None)
    if bank_ms is None:
        bank_ms = attn_res_bank_mean_square(blocks, self.prev_valid_blocks)
    prefix_sum = hidden_states
    write_idx = self.block_write_idx if self.is_block_write_layer else -1
    new_prefix, blocks, bank_ms, hidden_states = _layer_attn_res(prefix_sum, blocks, bank_ms,
                                                                 self.self_attention_res_proj,
                                                                 self.self_attention_res_norm, self.prev_valid_blocks,
                                                                 prefix_delta, self.input_layernorm, write_idx)
    # in a block-write layer the updated prefix lives in the bank and the attention output starts the new prefix
    prefix_sum = None if self.is_block_write_layer else new_prefix

    hidden_states = self._run_self_attn(positions, hidden_states)

    if prefix_sum is None:
        prefix_sum, prefix_delta = hidden_states, None
    else:
        prefix_delta = hidden_states

    mlp_valid_blocks = self.prev_valid_blocks + (1 if self.is_block_write_layer else 0)
    prefix_sum, blocks, bank_ms, hidden_states = _layer_attn_res(prefix_sum, blocks, bank_ms, self.mlp_res_proj,
                                                                 self.mlp_res_norm, mlp_valid_blocks, prefix_delta,
                                                                 self.post_attention_layernorm, -1)
    hidden_states = self.mlp(hidden_states)
    return prefix_sum, (blocks, bank_ms), hidden_states


# upstream vllm/models/kimi_k3/amd/linear.py names it KimiDecoderLayer; KimiLinearDecoderLayer is the older name
KIMI_DECODER_LAYER_CLASSES = ("KimiDecoderLayer", "KimiLinearDecoderLayer")


def find_kimi_decoder_layer_class(linear_mod) -> type | None:
    """The decoder layer class of the Kimi-K3 ``linear`` module under any of its known names, or None."""
    return next((getattr(linear_mod, n) for n in KIMI_DECODER_LAYER_CLASSES if hasattr(linear_mod, n)), None)


def kimi_decoder_layer_class(linear_mod) -> type | None:
    """The decoder layer class of the Kimi-K3 ``linear`` module that has ``forward_attn_residual``, or None."""
    for name in KIMI_DECODER_LAYER_CLASSES:
        cls = getattr(linear_mod, name, None)
        if cls is not None and hasattr(cls, "forward_attn_residual"):
            return cls
    return None


def install_functional_attn_res(linear_mod) -> type | None:
    """Replace ``forward_attn_residual`` of the Kimi-K3 decoder layer with the out-of-place version.

    Returns the patched class (idempotent), or None when ``linear_mod`` has no such class; the caller must report
    that loudly: without the patch the in-place upstream version runs inside the per-layer compiled graphs.
    The layers pass the block bank on as ``(bank, bank_ms)`` (see ``attn_res_functional_cached``).
    """
    cls: Any = kimi_decoder_layer_class(linear_mod)
    if cls is None or getattr(cls.forward_attn_residual, "_hpu_functional_attn_res", False):
        return cls

    def forward_attn_residual(self, positions, hidden_states, block_residual, prefix_delta):
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if prefix_delta is not None and prefix_delta.dim() == 3:
            prefix_delta = prefix_delta.reshape(-1, prefix_delta.shape[-1])
        if positions is not None and positions.dim() == 2:
            positions = positions.reshape(-1)
        blocks = attn_res_bank(block_residual)
        if blocks.dim() == 3 and (blocks.shape[0] != hidden_states.shape[0]
                                  or blocks.shape[-1] != hidden_states.shape[-1]):
            # Silently replacing the block bank with zeros would drop the stored blocks and corrupt every
            # later attn_res mix, so fail loudly instead.
            raise RuntimeError(f"Kimi-K3 block residual bank {tuple(blocks.shape)} does not match "
                               f"the hidden states {tuple(hidden_states.shape)}")
        return forward_attn_residual_functional(self, positions, hidden_states, block_residual, prefix_delta)

    forward_attn_residual._hpu_functional_attn_res = True  # type: ignore[attr-defined]
    cls.forward_attn_residual = forward_attn_residual
    return cls


def attn_res_bank(block_residual) -> torch.Tensor:
    """The block bank of a decoder-layer ``block_residual`` output: a plain bank or a ``(bank, bank_ms)`` tuple."""
    return block_residual[0] if isinstance(block_residual, tuple) else block_residual


def attn_res_layer_variants(modules) -> int:
    """Distinct (is_block_write_layer, block_write_idx, prev_valid_blocks) of the attn_res decoder layers, >= 1.

    dynamo specializes these python ints, so with regional compilation every variant is its own cache entry of the
    shared forward code object, per bucket shape (Kimi-K3: 93 layers, block 12 -> 16 variants).
    """
    variants = {(m.is_block_write_layer, m.block_write_idx, m.prev_valid_blocks)
                for m in modules if getattr(m, "use_attn_residuals", False) and hasattr(m, "prev_valid_blocks")}
    return max(1, len(variants))


def _layer_attn_res(prefix_sum, blocks, bank_ms, proj, norm, num_valid_blocks, delta, output_norm, block_write_idx):
    return attn_res_functional_cached(prefix_sum, delta, blocks, bank_ms, norm.weight, proj.weight.squeeze(0),
                                      None if output_norm is None else output_norm.weight, num_valid_blocks,
                                      block_write_idx, norm.variance_epsilon,
                                      0.0 if output_norm is None else output_norm.variance_epsilon)
