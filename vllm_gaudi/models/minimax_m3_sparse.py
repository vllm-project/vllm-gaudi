# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU (Gaudi) building blocks for MiniMax-M3.

Upstream ships only CUDA/ROCm implementations of Gemma-style RMSNorm and
SwiGLU-OAI; this module re-implements that math in pure ``torch`` for HPU
(Triton is disabled on HPU and the CUDA custom ops are absent from the
``+empty`` build).

Contents
--------
* ``gemma_rmsnorm`` / ``GemmaRMSNorm`` -- Gemma-style RMSNorm ``x * (1 + w)``
  computed in fp32 (matches ``amd/ops/gemma_rmsnorm.py``).
* ``swiglu_oai`` -- SwiGLU-OAI on a split ``[*, 2I]`` layout (matches
  ``amd/ops/swiglu_oai.py``).

MiniMax-M3's MSA ("sparse") attention layers currently run as full **dense**
causal attention on the existing HPU paged-attention stack; exact block-sparse
top-k parity is future work and is intentionally not implemented here.
"""

from __future__ import annotations

import torch


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma-style RMSNorm: ``normalize(x) * (1 + weight)`` computed in fp32.

    Normalizes over the last dim and broadcasts ``weight`` (shape ``[N]``) over
    it, so it serves both full-hidden norms and per-head q/k norms.
    """
    orig_dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    out = xf * (1.0 + weight.float())
    return out.to(orig_dtype)


class GemmaRMSNorm(torch.nn.Module):
    """Gemma-style RMSNorm with an optional fused residual add.

    ``forward(x)`` returns the normalized tensor. ``forward(x, residual)``
    returns ``(normed(x + residual), x + residual)`` -- the pre-norm sum is the
    residual consumed by the next layer, matching vLLM's ``RMSNorm`` contract.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x = x + residual
            residual = x
            out = gemma_rmsnorm(x, self.weight, self.variance_epsilon)
            return out, residual
        return gemma_rmsnorm(x, self.weight, self.variance_epsilon)


def swiglu_oai(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float | None,
) -> torch.Tensor:
    """SwiGLU-OAI on a split-layout ``[*, 2I]`` tensor -> ``[*, I]``.

    ``gate`` is the first half, ``up`` the second half::

        gate = clamp(gate, max=limit)
        up   = clamp(up, -limit, +limit)
        out  = gate * sigmoid(alpha * gate) * (up + beta)

    Computed in fp32 for accuracy, cast back to the input dtype.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    g = gate.float()
    u = up.float()
    if limit is not None:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    out = g * torch.sigmoid(alpha * g) * (u + beta)
    return out.to(gate_up.dtype)
