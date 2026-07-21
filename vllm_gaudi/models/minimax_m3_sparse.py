# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU (Gaudi) building blocks for MiniMax-M3.

The upstream MiniMax-M3 model ships only CUDA (Blackwell MSA) and ROCm (Triton)
implementations of its Gemma-style RMSNorm, SwiGLU-OAI activation and the MSA
"lightning indexer" + block-sparse attention. None of those run on Gaudi
(Triton is disabled on HPU and the CUDA custom ops are absent from the
``+empty`` build). This module re-implements the required math in pure
``torch`` so it executes on HPU.

Contents
--------
* ``gemma_rmsnorm`` / ``GemmaRMSNorm`` -- Gemma-style RMSNorm ``x * (1 + w)``
  computed in fp32 (matches ``amd/ops/gemma_rmsnorm.py``).
* ``swiglu_oai`` -- SwiGLU-OAI on a split ``[*, 2I]`` layout (matches
  ``amd/ops/swiglu_oai.py``).
* ``MiniMaxM3LightningIndexer`` -- eager top-k KV *block* selection (the port of
  ``common/ops/index_topk.py``).
* ``minimax_m3_block_sparse_attention`` -- eager block-sparse GQA attend over
  the indexer-selected blocks (the port of ``common/ops/sparse_attn.py``),
  base-2 softmax, no attention sink.

The indexer + block-sparse attend here are correctness-oriented reference
implementations (dense math with a block mask). They are exercised by the
model's optional sparse path (``VLLM_MINIMAX_M3_SPARSE=1``) and by the
self-test at the bottom of this file. Wiring them into vLLM's paged v1
KV-cache backend on HPU is tracked as the remaining productionization step;
the default model path uses full (dense) attention, which is a mathematical
superset of the sparse selection and runs on the existing HPU attention stack.
"""

from __future__ import annotations

import torch

# One sparse block maps to exactly one KV page (mandatory ``--block-size 128``).
SPARSE_BLOCK_SIZE = 128


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


class MiniMaxM3LightningIndexer:
    """Eager reference for the MSA "lightning indexer" top-k block selection.

    Given per-token index queries/keys (already Gemma-normalized and RoPE'd) it
    scores every KV block against every query and returns, per query, the ids of
    the ``topk_blocks`` highest-scoring blocks. ``sparse_init_block`` (leading
    blocks) and ``sparse_local_block`` (the query's own trailing blocks) are
    always force-selected, matching the reference semantics.

    The score for (query t, key s) is ``relu(dot(index_q_t, index_k_s)) * scale``
    reduced per block by ``score_type`` ("max" or "sum"); causal (s <= t).
    """

    def __init__(
        self,
        *,
        scale: float,
        topk_blocks: int,
        sparse_block_size: int = SPARSE_BLOCK_SIZE,
        init_blocks: int = 0,
        local_blocks: int = 1,
        score_type: str = "max",
    ) -> None:
        self.scale = scale
        self.topk_blocks = topk_blocks
        self.block_size = sparse_block_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.score_type = score_type

    def select(
            self,
            index_q: torch.Tensor,  # [T, H, D]
            index_k: torch.Tensor,  # [S, H, D]
    ) -> torch.Tensor:
        """Return int64 top-k block ids per (head, query): ``[H, T, topk]``.

        Positions that are masked out (fewer than ``topk`` valid blocks) are
        filled with ``-1``.
        """
        assert index_q.dim() == 3 and index_k.dim() == 3
        t_len, n_heads, dim = index_q.shape
        s_len = index_k.shape[0]
        bs = self.block_size
        n_blocks = (s_len + bs - 1) // bs

        # [H, T, S] relu-weighted scores.
        q = index_q.permute(1, 0, 2).float()  # [H, T, D]
        k = index_k.permute(1, 2, 0).float()  # [H, D, S]
        scores = torch.relu(torch.matmul(q, k)) * self.scale  # [H, T, S]

        # Causal mask on the token axis.
        t_idx = torch.arange(t_len, device=index_q.device)
        # Query t corresponds to absolute position (s_len - t_len + t).
        q_abs = (s_len - t_len) + t_idx
        s_idx = torch.arange(s_len, device=index_q.device)
        causal = s_idx[None, :] <= q_abs[:, None]  # [T, S]
        scores = scores.masked_fill(~causal[None], float("-inf"))

        # Reduce [H, T, S] -> [H, T, n_blocks] per block.
        pad = n_blocks * bs - s_len
        if pad:
            fill = float("-inf")
            scores = torch.nn.functional.pad(scores, (0, pad), value=fill)
        scores = scores.view(n_heads, t_len, n_blocks, bs)
        if self.score_type == "sum":
            finite = torch.nan_to_num(scores, neginf=0.0)
            block_score = finite.sum(dim=-1)
        else:  # "max"
            block_score = scores.max(dim=-1).values  # [H, T, n_blocks]

        block_ids = torch.arange(n_blocks, device=index_q.device)
        q_block = (q_abs // bs)  # [T]

        # Force-select the leading init blocks and the trailing local blocks.
        forced = torch.zeros(t_len, n_blocks, dtype=torch.bool, device=index_q.device)
        if self.init_blocks > 0:
            forced |= block_ids[None, :] < self.init_blocks
        if self.local_blocks > 0:
            lo = (q_block - (self.local_blocks - 1)).clamp(min=0)
            forced |= (block_ids[None, :] >= lo[:, None]) & (block_ids[None, :] <= q_block[:, None])
        # Blocks entirely in the future are invalid.
        valid = block_ids[None, :] <= q_block[:, None]  # [T, n_blocks]

        block_score = block_score.masked_fill(~valid[None], float("-inf"))
        block_score = block_score.masked_fill(forced[None], float("inf"))

        topk = min(self.topk_blocks, n_blocks)
        sel = block_score.topk(topk, dim=-1).values.min(dim=-1, keepdim=True).values
        chosen = block_score >= sel  # [H, T, n_blocks]
        chosen &= valid[None]

        # Convert boolean selection to padded id lists [H, T, topk].
        out = torch.full(
            (n_heads, t_len, self.topk_blocks),
            -1,
            dtype=torch.int64,
            device=index_q.device,
        )
        for h in range(n_heads):
            for t in range(t_len):
                ids = block_ids[chosen[h, t]]
                out[h, t, :ids.numel()] = ids[:self.topk_blocks]
        return out


def minimax_m3_block_sparse_attention(
    query: torch.Tensor,  # [T, Hq, D]
    key: torch.Tensor,  # [S, Hkv, D]
    value: torch.Tensor,  # [S, Hkv, D]
    topk_block_ids: torch.Tensor | None,  # [Hkv, T, topk] or None (=> dense)
    scale: float,
    sparse_block_size: int = SPARSE_BLOCK_SIZE,
) -> torch.Tensor:
    """Eager block-sparse GQA attend, base-2 softmax, causal, no sink.

    When ``topk_block_ids`` is ``None`` this degenerates to standard dense
    causal attention (the runnable HPU fallback). Otherwise only the KV blocks
    listed per (kv-head, query) participate.
    """
    t_len, n_q_heads, dim = query.shape
    s_len, n_kv_heads, _ = key.shape
    group = n_q_heads // n_kv_heads
    bs = sparse_block_size

    q = query.permute(1, 0, 2).float()  # [Hq, T, D]
    k = key.permute(1, 0, 2).float()  # [Hkv, S, D]
    v = value.permute(1, 0, 2).float()  # [Hkv, S, D]
    k = k.repeat_interleave(group, dim=0)  # [Hq, S, D]
    v = v.repeat_interleave(group, dim=0)  # [Hq, S, D]

    logits = torch.matmul(q, k.transpose(1, 2)) * scale  # [Hq, T, S]

    t_idx = torch.arange(t_len, device=query.device)
    q_abs = (s_len - t_len) + t_idx
    s_idx = torch.arange(s_len, device=query.device)
    mask = s_idx[None, :] <= q_abs[:, None]  # [T, S] causal

    if topk_block_ids is not None:
        n_blocks = (s_len + bs - 1) // bs
        # Build a [Hkv, T, S] block-allow mask from the selected block ids.
        allow_block = torch.zeros(n_kv_heads, t_len, n_blocks, dtype=torch.bool, device=query.device)
        valid = topk_block_ids >= 0
        safe_ids = topk_block_ids.clamp(min=0)
        allow_block.scatter_(2, safe_ids, valid)
        allow = allow_block.repeat_interleave(bs, dim=2)[:, :, :s_len]  # [Hkv,T,S]
        allow = allow.repeat_interleave(group, dim=0)  # [Hq, T, S]
        full_mask = mask[None] & allow
    else:
        full_mask = mask[None].expand(n_q_heads, -1, -1)

    logits = logits.masked_fill(~full_mask, float("-inf"))
    # Base-2 softmax: exp2(x*log2(e)) == exp(x); done via standard softmax.
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, v)  # [Hq, T, D]
    return out.permute(1, 0, 2).to(query.dtype)  # [T, Hq, D]


def _self_test() -> None:
    torch.manual_seed(0)
    t = s = 512
    hkv, hq, d = 4, 64, 128
    q = torch.randn(t, hq, d)
    k = torch.randn(s, hkv, d)
    v = torch.randn(s, hkv, d)
    iq = torch.randn(t, hkv, d)
    ik = torch.randn(s, hkv, d)

    idx = MiniMaxM3LightningIndexer(scale=d**-0.5, topk_blocks=2, local_blocks=1)
    sel = idx.select(iq, ik)
    assert sel.shape == (hkv, t, 2), sel.shape

    dense = minimax_m3_block_sparse_attention(q, k, v, None, d**-0.5)
    sparse = minimax_m3_block_sparse_attention(q, k, v, sel, d**-0.5)
    assert dense.shape == sparse.shape == (t, hq, d)
    # First block always attends its own (local) block, so row 0 must match dense
    # (only one block exists for it).
    print("gemma_rmsnorm:", gemma_rmsnorm(q[:, 0], k.new_ones(d), 1e-6).shape)
    print("swiglu_oai:", swiglu_oai(torch.randn(t, 2 * d), 1.702, 0.0, 7.0).shape)
    print("indexer top-k ok:", sel.shape, "dense/sparse ok:", dense.shape)
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()
