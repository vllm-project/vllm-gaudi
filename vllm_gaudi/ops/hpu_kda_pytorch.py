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
sequence); the default delta rule is chunk-parallel (``kda_chunked_steps``, a loop over L / chunk chunks), the
token loop over ``L`` (``kda_recurrent_steps``) is kept for validation; padded tokens are exact no-ops
(``g = 0``, ``beta = 0``). There is no host sync and no data dependent Python control flow, so the functions work
in lazy HPU graph capture and eagerly.
"""

from __future__ import annotations

from typing import Callable

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


def kda_recurrent_steps(state: torch.Tensor,
                        q: torch.Tensor,
                        k: torch.Tensor,
                        v: torch.Tensor,
                        decay: torch.Tensor,
                        beta: torch.Tensor,
                        block: int = 64,
                        step_hook: Callable[[], None] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Token loop. q,k [B,L,H,K]; v [B,L,H,V]; decay [B,L,H,K]; beta [B,L,H]; state [B,H,V,K].

    ``step_hook`` (e.g. ``htorch.core.mark_step`` in lazy mode) is called every ``block`` tokens so the traced
    graph stays small and identical blocks reuse one compiled recipe.
    """
    outs = []
    seq_len = q.shape[1]
    for t in range(seq_len):
        out, state = kda_recurrent_step(state, q[:, t], k[:, t], v[:, t], decay[:, t], beta[:, t])
        outs.append(out)
        if step_hook is not None and (t + 1) % block == 0 and t + 1 < seq_len:
            step_hook()
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


def kda_chunked_steps(state: torch.Tensor,
                      q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor,
                      g: torch.Tensor,
                      beta: torch.Tensor,
                      chunk_size: int = 64,
                      block_chunks: int = 1,
                      step_hook: Callable[[], None] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-parallel (WY / UT form) delta rule, identical to ``kda_recurrent_steps``.

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

    if step_hook is not None and num_chunks > 1:
        step_hook()  # the whole-sequence preparation above is its own graph segment
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
        if step_hook is not None and (n + 1) % block_chunks == 0 and n + 1 < num_chunks:
            step_hook()
    out = torch.stack(outs, dim=2)  # [B,H,N,C,V]
    out = out.permute(0, 2, 3, 1, 4).reshape(bsz, num_chunks * chunk, heads, vdim)
    return out[:, :seq_len], state


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
def kda_core_prefill(mixed_qkv: torch.Tensor,
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
                     block: int = 64,
                     step_hook: Callable[[], None] | None = None,
                     prefill_impl: str = "chunked",
                     chunk_size: int = 64,
                     block_chunks: int = 1) -> torch.Tensor:
    """Uniform padded prefill batch.

    mixed_qkv [B,L,3P] (pre-conv), raw_g [B,L,P], raw_beta [B,L,H], token_mask [B,L] (1 = real token, padding at
    the end of each sequence) or None, state_idx [B] (-1 = padded row), has_initial_state [B] or None.
    Reads the initial conv/ssm state from the caches, writes the final ones back. Returns o [B,L,H,V] (fp32).
    ``prefill_impl`` selects the delta rule: "chunked" (chunk-parallel WY form, default) or "loop" (token loop,
    kept for validation); ``block`` is the loop's hook period in tokens, ``chunk_size`` the chunk length and
    ``block_chunks`` the number of chunks between two ``step_hook`` calls (1 = one lazy graph per chunk).
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
    w = conv_weight.reshape(channels, width).float()
    conv_out = torch.zeros(bsz, seq_len, channels, dtype=torch.float32, device=device)
    for i in range(width):
        conv_out = conv_out + seq[:, i:i + seq_len, :] * w[:, i]
    if conv_bias is not None:
        conv_out = conv_out + conv_bias.float()
    conv_out = torch.nn.functional.silu(conv_out)
    # final conv state = last W-1 REAL inputs of each sequence
    gather_idx = (lens.unsqueeze(1) + torch.arange(width_m1, device=device).unsqueeze(0)).unsqueeze(-1)
    new_conv = torch.gather(seq, 1, gather_idx.expand(-1, -1, channels))
    write_conv_state(conv_cache, slots, new_conv)

    # --- delta rule ---
    q, k, v, g, beta = _prepare_qkv_gates(conv_out, raw_g, raw_beta, A_log, dt_bias, lower_bound, num_heads, head_dim,
                                          token_mask)
    state = read_ssm_state(ssm_cache, slots) * use_init.reshape(bsz, 1, 1, 1)
    if prefill_impl == "loop":
        out, state = kda_recurrent_steps(state, q, k, v, torch.exp(g), beta, block=block, step_hook=step_hook)
    elif prefill_impl == "chunked":
        out, state = kda_chunked_steps(state,
                                       q,
                                       k,
                                       v,
                                       g,
                                       beta,
                                       chunk_size=chunk_size,
                                       block_chunks=block_chunks,
                                       step_hook=step_hook)
    else:
        raise ValueError(f"prefill_impl must be 'chunked' or 'loop', got {prefill_impl!r}")
    write_ssm_state(ssm_cache, slots, state)
    return out


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
    """Stores the rows returned by ``kda_decode_step`` (the only in-place part of a decode step)."""
    conv_cache.index_copy_(0, slots, conv_rows)
    ssm_cache.index_copy_(0, slots, ssm_rows)


def kda_core_decode(mixed_qkv: torch.Tensor, raw_g: torch.Tensor, raw_beta: torch.Tensor, conv_weight: torch.Tensor,
                    conv_bias: torch.Tensor | None, A_log: torch.Tensor, dt_bias: torch.Tensor,
                    lower_bound: float | None, conv_cache: torch.Tensor, ssm_cache: torch.Tensor,
                    state_idx: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """One token per request. mixed_qkv [B,3P], raw_g [B,P], raw_beta [B,H], state_idx [B]. Returns o [B,H,V] fp32.

    ``kda_decode_step`` followed by ``write_kda_decode_state`` (updates the caches in place)."""
    out, slots, conv_rows, ssm_rows = kda_decode_step(mixed_qkv, raw_g, raw_beta, conv_weight, conv_bias, A_log,
                                                      dt_bias, lower_bound, conv_cache, ssm_cache, state_idx, num_heads,
                                                      head_dim)
    write_kda_decode_state(conv_cache, ssm_cache, slots, conv_rows, ssm_rows)
    return out


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
        group_id = layer_to_group.get(prefix, layer_to_group.get(module_name))
        if group_id is None:
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
def attn_res_functional(prefix: torch.Tensor,
                        delta: torch.Tensor | None,
                        blocks: torch.Tensor,
                        norm_weight: torch.Tensor,
                        qk_weight: torch.Tensor,
                        output_norm_weight: torch.Tensor | None,
                        num_blocks: int,
                        block_write_idx: int,
                        eps: float,
                        output_norm_eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Out-of-place ``attn_res``: returns ``(new_prefix, new_blocks, mixed)``, no input is modified.

    Same numbers as the Triton kernel (see ``attn_res_torch``, which adds its in-place side effects). Needed in
    torch.compile mode, where regional compilation makes the prefix and the block bank graph inputs and HPU
    compile can silently drop in-place writes to them. ``new_prefix`` is ``prefix`` itself without a delta and
    ``new_blocks`` is ``blocks`` itself without a block write (no copy in those cases). All shapes are static.
    """
    num_tokens, hidden = prefix.shape
    if delta is not None and delta.shape != prefix.shape:
        raise ValueError(f"delta {tuple(delta.shape)} does not match prefix {tuple(prefix.shape)}")
    if blocks.dim() != 3 or blocks.shape[0] != num_tokens or blocks.shape[-1] != hidden:
        raise ValueError(f"block bank {tuple(blocks.shape)} does not match prefix {tuple(prefix.shape)}")

    if delta is not None:
        new_prefix = (prefix.float() + delta.float()).to(prefix.dtype)
        updated = new_prefix.float()
    else:
        new_prefix = prefix
        updated = prefix.float()
    new_blocks = blocks
    if block_write_idx >= 0:
        new_blocks = torch.cat((blocks[:, :block_write_idx], updated.to(blocks.dtype).unsqueeze(1),
                                blocks[:, block_write_idx + 1:]),
                               dim=1)

    if num_blocks == 0:
        mixed = updated
    else:
        values = torch.cat((new_blocks[:, :num_blocks].float(), updated.unsqueeze(1)), dim=1)  # [M, S, D]
        rstd = torch.rsqrt(values.pow(2).mean(dim=-1) + eps)  # [M, S]
        logits = (values * (norm_weight.float() * qk_weight.float())).sum(dim=-1) * rstd
        probs = torch.softmax(logits, dim=-1)
        mixed = (probs.unsqueeze(-1) * values).sum(dim=1)
    if output_norm_weight is not None:
        mixed = mixed * torch.rsqrt(mixed.pow(2).mean(dim=-1, keepdim=True) + output_norm_eps)
        mixed = mixed * output_norm_weight.float()
    return new_prefix, new_blocks, mixed.to(prefix.dtype)


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
    new_prefix, new_blocks, mixed = attn_res_functional(prefix, delta, blocks, norm_weight, qk_weight,
                                                        output_norm_weight, num_blocks, block_write_idx, eps,
                                                        output_norm_eps)
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
    prefix_sum = hidden_states
    write_idx = self.block_write_idx if self.is_block_write_layer else -1
    new_prefix, block_residual, hidden_states = _layer_attn_res(prefix_sum, block_residual,
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
    prefix_sum, block_residual, hidden_states = _layer_attn_res(prefix_sum, block_residual, self.mlp_res_proj,
                                                                 self.mlp_res_norm, mlp_valid_blocks, prefix_delta,
                                                                 self.post_attention_layernorm, -1)
    hidden_states = self.mlp(hidden_states)
    return prefix_sum, block_residual, hidden_states


def _layer_attn_res(prefix_sum, block_residual, proj, norm, num_valid_blocks, delta, output_norm, block_write_idx):
    return attn_res_functional(prefix_sum, delta, block_residual, norm.weight, proj.weight.squeeze(0),
                               None if output_norm is None else output_norm.weight, num_valid_blocks,
                               block_write_idx, norm.variance_epsilon,
                               0.0 if output_norm is None else output_norm.variance_epsilon)
