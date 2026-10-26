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
sequence) and loops over ``L``; padded tokens are exact no-ops (``g = 0``, ``beta = 0``). There is no host sync
and no data dependent Python control flow, so the functions work in lazy HPU graph capture and eagerly.
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


def _prepare_qkv_gates(conv_out: torch.Tensor, raw_g: torch.Tensor, raw_beta: torch.Tensor, A_log: torch.Tensor,
                       dt_bias: torch.Tensor, lower_bound: float | None, num_heads: int, head_dim: int,
                       token_mask: torch.Tensor | None):
    """conv_out [B,L,3P] (after silu), raw_g [B,L,P], raw_beta [B,L,H] -> q,k,v,decay,beta (fp32, [B,L,H,*])."""
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
    return q, k, v, torch.exp(g), beta


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
                     step_hook: Callable[[], None] | None = None) -> torch.Tensor:
    """Uniform padded prefill batch.

    mixed_qkv [B,L,3P] (pre-conv), raw_g [B,L,P], raw_beta [B,L,H], token_mask [B,L] (1 = real token, padding at
    the end of each sequence) or None, state_idx [B] (-1 = padded row), has_initial_state [B] or None.
    Reads the initial conv/ssm state from the caches, writes the final ones back. Returns o [B,L,H,V] (fp32).
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
    q, k, v, decay, beta = _prepare_qkv_gates(conv_out, raw_g, raw_beta, A_log, dt_bias, lower_bound, num_heads,
                                              head_dim, token_mask)
    state = read_ssm_state(ssm_cache, slots) * use_init.reshape(bsz, 1, 1, 1)
    out, state = kda_recurrent_steps(state, q, k, v, decay, beta, block=block, step_hook=step_hook)
    write_ssm_state(ssm_cache, slots, state)
    return out


def kda_core_decode(mixed_qkv: torch.Tensor, raw_g: torch.Tensor, raw_beta: torch.Tensor, conv_weight: torch.Tensor,
                    conv_bias: torch.Tensor | None, A_log: torch.Tensor, dt_bias: torch.Tensor,
                    lower_bound: float | None, conv_cache: torch.Tensor, ssm_cache: torch.Tensor,
                    state_idx: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """One token per request. mixed_qkv [B,3P], raw_g [B,P], raw_beta [B,H], state_idx [B]. Returns o [B,H,V] fp32."""
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
    write_conv_state(conv_cache, slots, seq[:, 1:, :])

    q, k, v, decay, beta = _prepare_qkv_gates(conv_out.unsqueeze(1), raw_g.unsqueeze(1), raw_beta.unsqueeze(1), A_log,
                                              dt_bias, lower_bound, num_heads, head_dim, None)
    state = read_ssm_state(ssm_cache, slots)
    out, state = kda_recurrent_step(state, q[:, 0], k[:, 0], v[:, 0], decay[:, 0], beta[:, 0])
    write_ssm_state(ssm_cache, slots, state)
    return out
