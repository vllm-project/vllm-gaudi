# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA prefill with ``qk_rope_head_dim == 0``.

Some MLA checkpoints carry no RoPE component. GLM-5.3-Flash is one:
``qk_rope_head_dim=0`` with ``qk_nope_head_dim=256``, so ``qk_head_dim`` equals
``qk_nope_head_dim``.

``forward_mha`` splits ``latent_vec_k`` into ``(k_c_normed, k_pe)`` and then
reshapes ``k_pe``. With a zero-sized RoPE part the split is well defined and
yields an empty tensor, but the reshape is not::

    k_pe.view(-1, 1, 0)
    RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 1, 0]
    because the unspecified dimension size -1 can be any value and is ambiguous

Only the reshape and the concat consuming it need guarding -- ``split`` and
``cat`` are both fine on a zero-sized dim. These tests pin that behaviour down
so the distinction does not get lost.
"""

import pytest
import torch

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 256
V_HEAD_DIM = 256
NUM_HEADS = 16
TOKENS = 4


def _k_from_latent(latent_vec_k, qk_rope_head_dim, guarded):
    """The k-assembly half of ``forward_mha``, with and without the guard."""
    k_c_normed, k_pe = latent_vec_k.split([KV_LORA_RANK, qk_rope_head_dim], dim=-1)

    if guarded:
        if qk_rope_head_dim > 0:
            k_pe = k_pe.view(-1, 1, qk_rope_head_dim)
    else:
        k_pe = k_pe.view(-1, 1, qk_rope_head_dim)

    tokens = k_c_normed.shape[0]
    kv_nope = torch.randn(tokens, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
    k_nope, v = kv_nope.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)

    if guarded and qk_rope_head_dim == 0:
        k = k_nope
    else:
        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
    return k, v


@pytest.mark.parametrize("qk_rope_head_dim", [0, 64])
def test_k_assembly_handles_zero_rope_dim(qk_rope_head_dim):
    """k must come out as [tokens, heads, qk_nope + qk_rope] for either case."""
    latent = torch.randn(TOKENS, KV_LORA_RANK + qk_rope_head_dim)

    k, v = _k_from_latent(latent, qk_rope_head_dim, guarded=True)

    assert k.shape == (TOKENS, NUM_HEADS, QK_NOPE_HEAD_DIM + qk_rope_head_dim)
    assert v.shape == (TOKENS, NUM_HEADS, V_HEAD_DIM)


def test_unguarded_reshape_is_what_fails():
    """Pin down that the reshape, not the split or the concat, is the problem."""
    latent = torch.randn(TOKENS, KV_LORA_RANK)

    # split is fine with a zero-sized piece
    k_c_normed, k_pe = latent.split([KV_LORA_RANK, 0], dim=-1)
    assert k_pe.numel() == 0

    # cat is fine too
    k_nope = torch.randn(TOKENS, NUM_HEADS, QK_NOPE_HEAD_DIM)
    torch.cat((k_nope, k_pe.view(TOKENS, 1, 0).expand(TOKENS, NUM_HEADS, 0)), dim=-1)

    # the ambiguous reshape is not
    with pytest.raises(RuntimeError, match="ambiguous"):
        k_pe.view(-1, 1, 0)
