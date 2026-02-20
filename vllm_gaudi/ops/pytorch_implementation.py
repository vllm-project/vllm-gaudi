# Copyright (C) 2024-2026 Habana Labs, Ltd. an Intel Company.

import torch
import torch.nn.functional as F


def new_chunk_cumsum(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """
    Arguments:
        dt: Tensor - (seqlen, nheads)
        A: Tensor - (nheads)
        chunk_size: int
        dt_bias: Optional Tensor - (nheads)
        dt_softplus: bool
        dt_limit: tuple - (min: float, max: float)

    Return:
        dA_cumsum: Tensor - (nheads, nchunks, chunk_size)
        dt_out: Tensor - (nheads, nchunks, chunk_size)
    """
    seqlen, nheads = dt.shape
    nchunks = seqlen // chunk_size
    dt_min, dt_max = dt_limit

    dt = dt.float()
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, )
        dt += dt_bias.view(1, nheads).float()

    if dt_softplus:
        dt = F.softplus(dt)

    dt = torch.clamp(dt, dt_min, dt_max)
    dA = dt * A.view(1, nheads)
    dA = dA.transpose(0, 1).reshape(nheads, nchunks, chunk_size)
    dt = dt.transpose(0, 1).reshape(nheads, nchunks, chunk_size)

    dA_cumsum = dA.cumsum(dim=-1)

    return dA_cumsum, dt


def new_chunk_state(B_expanded, x_chunked, dt_t, dA_cumsum_t, states_in_fp32=True):
    """
    Arguments:
        B_expanded: Tensor - pre-expanded B (nchunks, chunk_size, nheads, dstate)
        x_chunked: Tensor - pre-chunked x (nchunks, chunk_size, nheads, hdim)
        dt_t: Tensor - pre-transposed dt (nchunks, nheads, chunk_size), float32
        dA_cumsum_t: Tensor - pre-transposed dA_cumsum (nchunks, nheads, chunk_size), float32
        states_in_fp32: bool

    Return:
        states: Tensor - (nchunks, nheads, hdim, dstate)
    """
    _, _, nheads, hdim = x_chunked.shape
    dstate = B_expanded.shape[-1]
    states_dtype = torch.float32 if states_in_fp32 else B_expanded.dtype
    x_dtype = x_chunked.dtype

    dA_cs_last = dA_cumsum_t[:, :, -1]
    scale = torch.exp(dA_cs_last.unsqueeze(2) - dA_cumsum_t) * dt_t
    scale = scale.transpose(1, 2).unsqueeze(3)

    B_scaled = (B_expanded * scale).to(x_dtype)
    x_for_bmm = x_chunked.permute(0, 2, 3, 1).flatten(0, 1)
    B_for_bmm = B_scaled.permute(0, 2, 1, 3).flatten(0, 1)
    state = torch.bmm(x_for_bmm, B_for_bmm).view(-1, nheads, hdim, dstate).to(states_dtype)
    return state


def new_chunk_scan(cb, x_chunked, dt_t, dA_cumsum_t, C, states, output, D=None, z=None, initial_states=None):
    """
    Arguments:
        cb: Tensor - (nchunks, ngroups, chunk_size, chunk_size) - already causally masked
        x_chunked: Tensor - pre-chunked x (nchunks, chunk_size, nheads, hdim)
        dt_t: Tensor - pre-transposed dt (nchunks, nheads, chunk_size), float32
        dA_cumsum_t: Tensor - pre-transposed dA_cumsum (nchunks, nheads, chunk_size), float32
        C: Tensor - (seqlen, ngroups, dstate)
        states: Tensor - (nchunks, nheads, hdim, dstate)
        output: Tensor - (seqlen, nheads, hdim)
        D: Optional Tensor - (nheads, hdim) or (nheads)
        z: Optional Tensor - (seqlen, nheads, hdim)
        initial_states: Optional Tensor - (1, nheads, hdim, dstate)

    Return:
        output: Tensor - (seqlen, nheads, hdim)
    """
    nchunks, ngroups, chunk_size, _ = cb.shape
    seqlen = nchunks * chunk_size
    _, _, dstate = C.shape
    _, _, nheads, hdim = x_chunked.shape
    assert nheads % ngroups == 0
    nheads_ngroups_ratio = nheads // ngroups
    mm_dtype = x_chunked.dtype

    x_chunked = x_chunked.transpose(1, 2)
    C = (C.view(nchunks, chunk_size, ngroups, 1, dstate).expand(nchunks, chunk_size, ngroups, nheads_ngroups_ratio,
                                                                dstate).reshape(nchunks, chunk_size, nheads,
                                                                                dstate).transpose(1, 2))

    cb = (cb.view(nchunks, ngroups, 1, chunk_size,
                  chunk_size).expand(nchunks, ngroups, nheads_ngroups_ratio, chunk_size,
                                     chunk_size).reshape(nchunks, nheads, chunk_size, chunk_size))
    states = states.float()
    init = torch.zeros_like(states[:1]) if initial_states is None else initial_states.float()
    prev_states = torch.cat([init, states[:-1]], dim=0)
    if D is not None:
        D = D.float()
    if z is not None:
        z = z.float()

    scale = torch.exp(dA_cumsum_t)
    acc = (C @ prev_states.to(mm_dtype).transpose(-1, -2)).float() * scale.unsqueeze(-1)

    decay = torch.exp(torch.clamp(dA_cumsum_t.unsqueeze(-1) - dA_cumsum_t.unsqueeze(-2), -30.0, 30))
    cb_scaled = (cb * decay * dt_t.unsqueeze(-2)).to(mm_dtype)
    acc = acc + (cb_scaled @ x_chunked).float()
    if D is not None:
        if D.dim() == 1:
            D = D[:, None]
        acc = acc + x_chunked * D.unsqueeze(0).unsqueeze(2)
    if z is not None:
        z = z.view(nchunks, chunk_size, nheads, hdim).transpose(1, 2)
        acc = acc * F.silu(z)
    out = acc.transpose(1, 2).reshape(seqlen, nheads, hdim)
    output.copy_(out)


def new_ssd_state_passing(states, dA_cumsum, initial_states=None, out_dtype=None):
    """
    Arguments:
        states: Tensor - (nchunks, nheads, hdim)
        dA_cumsum: Tensor - (nheads, nchunks, chunk_size)
        initial_states: Optional Tensor - (1, nheads, hdim)
        out_dtype: Optional dtype
    Return:
        output: Tensor - (nchunks, nheads, hdim)

    Note:
        This implementation uses a parallel prefix-sum approach via a full
        (nheads, nchunks+1, nchunks+1) decay matrix and batched matmul,
        trading O(nchunks^2) memory for O(1) sequential depth (fully parallel).
        This is intentional for performancel. For extremely large nchunks,
        memory usage may become significant; in such cases, consider chunking
        the sequence into smaller segments and processing sequentially.
    """
    nchunks, nheads, hdim = states.shape

    out_dtype = states.dtype if out_dtype is None else out_dtype
    device = states.device

    # Per-chunk total decay: dA_cumsum[:, c, -1] gives the cumulative
    # log-decay across all positions within chunk c.
    dA_last = dA_cumsum[:, :, -1]  # (nheads, nchunks)

    # Prepend initial state as position 0
    if initial_states is not None:
        init = initial_states[0].unsqueeze(0)
    else:
        init = torch.zeros(1, nheads, hdim, device=device, dtype=states.dtype)
    all_states = torch.cat([init, states], dim=0)  # (nchunks+1, nheads, hdim)

    # Build the (nchunks+1) x (nchunks+1) decay matrix via segment sums.
    # Prepend 0 for the initial-state position so the decay from
    # position 0 to itself is exp(0) = 1.
    dA_padded = F.pad(dA_last, (1, 0))  # (nheads, nchunks+1)
    cumsum = torch.cumsum(dA_padded, dim=-1)  # (nheads, nchunks+1)

    # segsum[h, i, j] = cumsum[h, i] - cumsum[h, j]
    #   i >= j (causal):  values <= 0  →  exp ∈ (0, 1]
    #   i <  j (acausal): values >  0  →  exp >  1
    segsum = cumsum.unsqueeze(-1) - cumsum.unsqueeze(-2)  # (nheads, n, n)
    decay = torch.tril(torch.exp(segsum))  # (nheads, n, n)

    # Parallel matmul replaces the sequential loop:
    #   new_states[h, i, :] = Σ_j  decay[h, i, j] · all_states[j, h, :]
    all_states_t = all_states.permute(1, 0, 2)  # (nheads, nchunks+1, hdim)
    new_states = torch.bmm(decay, all_states_t)  # (nheads, nchunks+1, hdim)

    # Positions 1..nchunks are the states after each chunk
    out = new_states[:, 1:, :].permute(1, 0, 2)  # (nchunks, nheads, hdim)

    return out.to(out_dtype)


def new_ssd_bmm(a, b, chunk_size, causal=False, output_dtype=None):
    """
    Arguments:
        a: Tensor - (seqlen, ngroups, k)
        b: Tensor - (seqlen, ngroups, k)
        chunk_size: int
        causal: bool
        out_dtype: Optional dtype
    Return:
        output: Tensor - (nchunks, ngroups, chunk_size, chunk_size)
    """
    seqlen, ngroups, k = a.shape
    nchunks = seqlen // chunk_size
    if a.stride(-1) != 1 and a.stride(0) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(0) != 1:
        b = b.contiguous()
    out_dtype = output_dtype if output_dtype is not None else a.dtype

    a = a.view(nchunks, chunk_size, ngroups, k).permute(0, 2, 1, 3)
    b = b.view(nchunks, chunk_size, ngroups, k).permute(0, 2, 3, 1)

    out = torch.matmul(a, b)
    if causal:
        out = torch.tril(out)

    return out.to(out_dtype)


# Based on https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/ops/triton/selective_state_update.py#L219
# Added support for softplus threshold which is applied by default in the triton kernel.
def selective_state_update_ref(state,
                               x,
                               dt,
                               A,
                               B,
                               C,
                               D=None,
                               z=None,
                               dt_bias=None,
                               dt_softplus=False,
                               softplus_thres=20.0):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    if dt_softplus:
        dt = torch.where(dt <= softplus_thres, F.softplus(dt), dt)
    dA = torch.exp(dt.unsqueeze(-1) * A)  # (batch, nheads, dim, dstate)
    B = B.repeat_interleave(nheads // ngroups, dim=1)  # (batch, nheads, dstate)
    C = C.repeat_interleave(nheads // ngroups, dim=1)  # (batch, nheads, dstate)
    dB = dt.unsqueeze(-1) * B.unsqueeze(-2)  # (batch, nheads, dim, dstate)
    state.copy_(state * dA + dB * x.unsqueeze(-1))  # (batch, dim, dstate)
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out
