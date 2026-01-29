# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


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
        dt = torch.where(dt <= 20.0, F.softplus(dt), dt)

    dt = torch.clamp(dt, dt_min, dt_max)
    dA = dt * A.view(1, nheads)
    dA = dA.transpose(0, 1).reshape(nheads, nchunks, chunk_size)
    dt = dt.transpose(0, 1).reshape(nheads, nchunks, chunk_size)

    dA_cumsum = dA.cumsum(dim=-1)

    return dA_cumsum, dt


def new_chunk_state(B, x, dt, dA_cumsum, states=None, states_in_fp32=True):
    """
    Arguments:
        B: Tensor - (seqlen, ngroups, dstate)
        x: Tensor - (seqlen, nheads, hdim)
        dt: Tensor - (nheads, nchunks, chunk_size)
        dA_cumsum: Tensor - (nheads, nchunks, chunk_siz)
        states: Optional Tensor - (nchunks, nheads, hdim, dstate)
        states_in_fp32: bool

    Return:
        states: Tensor - (nchunks, nheads, hdim, dstate)
    """
    seqlen, nheads, hdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    nheads_ngroups_ratio = nheads // ngroups
    states_dtype = torch.float32 if states_in_fp32 else B.dtype
    x_dtype = x.dtype
    B = B.float().view(nchunks, chunk_size, ngroups, 1, dstate).expand(-1, -1, -1, nheads_ngroups_ratio,
                                                                       -1).reshape(nchunks, chunk_size, nheads, dstate)
    x = x.view(nchunks, chunk_size, nheads, hdim)
    dt = dt.float().permute(1, 0, 2)
    dA_cumsum = dA_cumsum.float().permute(1, 0, 2)

    dA_cs_last = dA_cumsum[:, :, -1]
    scale = torch.exp(dA_cs_last.unsqueeze(2) - dA_cumsum) * dt
    scale = scale.transpose(1, 2).unsqueeze(3)

    B_scaled = (B * scale).to(x_dtype)
    x_for_bmm = x.permute(0, 2, 3, 1).reshape(nchunks * nheads, hdim, chunk_size)
    B_for_bmm = B_scaled.permute(0, 2, 1, 3).reshape(nchunks * nheads, chunk_size, dstate)
    state = torch.bmm(x_for_bmm, B_for_bmm).view(nchunks, nheads, hdim, dstate).to(states_dtype)
    if states is not None:
        states[:] = state
        return states
    return state


def new_chunk_scan(cb, x, dt, dA_cumsum, C, states, output, D=None, z=None, initial_states=None):
    """
    Arguments:
        cb: Tensor - (nchunks, ngroups, chunk_size, chunk_size)
        x: Tensor - (seqlen, nheads, hdim)
        dt: Tensor - (nheads, nchunks, chunk_size)
        dA_cumsum: Tensor - (nheads, nchunks, chunk_size)
        C: Tensor - (seqlen, ngroups, dstate)
        states: Tensor - (nchunks, nheads, hdim, dstate)
        output: Tensor - (seqlen, nheads, hdim)
        D: Optional Tensor - (nheads, hdim) or (nheads)
        z: Optional Tensor - (seqlen, nheads, hdim)
        initial_states: Optional Tensor - (1, nheads, hdim, dstate)

    Return:
        output: Tensor - (seqlen, nheads, hdim)
    """
    device = x.device
    seqlen, nheads, hdim = x.shape
    nchunks, ngroups, chunk_size, _ = cb.shape
    _, _, dstate = C.shape
    assert nheads % ngroups == 0
    nheads_ngroups_ratio = nheads // ngroups

    x = x.float().view(nchunks, chunk_size, nheads, hdim).transpose(1, 2)
    C = (C.float().view(nchunks, chunk_size, ngroups, 1,
                        dstate).expand(nchunks, chunk_size, ngroups, nheads_ngroups_ratio,
                                       dstate).reshape(nchunks, chunk_size, nheads, dstate).transpose(1, 2))
    cb = (cb.float().view(nchunks, ngroups, 1, chunk_size,
                          chunk_size).expand(nchunks, ngroups, nheads_ngroups_ratio, chunk_size,
                                             chunk_size).reshape(nchunks, nheads, chunk_size, chunk_size))
    dt = dt.float().transpose(0, 1)
    dA_cs = dA_cumsum.float().transpose(0, 1)
    states = states.float()
    if initial_states is not None:
        init = initial_states.float()
    else:
        init = torch.zeros(1, nheads, hdim, dstate, device=device, dtype=torch.float32)
    prev_states = torch.cat([init, states[:-1]], dim=0)
    if D is not None:
        D = D.float()
    if z is not None:
        z = z.float()

    scale = torch.exp(dA_cs)
    acc = (C @ prev_states.transpose(-1, -2)) * scale.unsqueeze(-1)

    decay = torch.exp(torch.clamp(dA_cs.unsqueeze(-1) - dA_cs.unsqueeze(-2), -30.0, 30))
    causal_mask = torch.tril(torch.ones((chunk_size, chunk_size), device=device))
    cb_scaled = cb * decay * dt.unsqueeze(-2) * causal_mask
    acc = acc + (cb_scaled @ x)
    if D is not None:
        if D.dim() == 1:
            D = D[:, None]
        acc = acc + x * D.unsqueeze(0).unsqueeze(2)
    if z is not None:
        z = z.view(nchunks, chunk_size, nheads, hdim).transpose(1, 2)
        acc = acc * z * torch.sigmoid(z)
    out = acc.transpose(1, 2).reshape(seqlen, nheads, hdim)
    output.copy_(out)


def new_ssd_state_passing(states, dA_cumsum, initial_states=None, out_dtype=None):
    """
    Arguments:
        states: Tensor - (nchunks, nheads, hdim)
        dA_cumsum: Tensor - (nheads, nchunks, chunk_size)
        initial_states: Optional Tensor - (1, nheads, hdim,)
        out_dtype: Optional dtype
    Return:
        output: Tensor - (nchunks, nheads, hdim)
    """
    nchunks, nheads, hdim = states.shape

    out_dtype = states.dtype if out_dtype is None else out_dtype
    device = states.device

    compute_dtype = torch.float32
    states_t = initial_states[0].to(dtype=compute_dtype, device=device) if initial_states is not None else torch.zeros(
        (nheads, hdim), device=device, dtype=compute_dtype)
    states = states.to(compute_dtype)
    out = torch.empty((nchunks, nheads, hdim), device=device, dtype=out_dtype)
    dA_cumsum = dA_cumsum.to(dtype=compute_dtype)
    last_pos = dA_cumsum.shape[-1] - 1
    dA_cumsum = dA_cumsum[:, :, last_pos]
    scale = torch.exp(dA_cumsum).unsqueeze(2)
    for c in range(nchunks):
        states_t = scale[:, c] * states_t + states[c]
        out[c] = states_t.to(dtype=out_dtype)
    return out


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

    a = a.float().view(nchunks, chunk_size, ngroups, k).permute(0, 2, 1, 3)
    b = b.float().view(nchunks, chunk_size, ngroups, k).permute(0, 2, 3, 1)

    out = torch.matmul(a, b)
    if causal:
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=a.device, dtype=torch.bool))
        out *= mask

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
    dA = torch.exp(rearrange(dt, "b h d -> b h d 1") * A)  # (batch, nheads, dim, dstate)
    B = repeat(B, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, "b g n -> b (g h) n", h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, "b h d -> b h d 1") * rearrange(B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
    state.copy_(state * dA + dB * rearrange(x, "b h d -> b h d 1"))  # (batch, dim, dstate)
    out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out
