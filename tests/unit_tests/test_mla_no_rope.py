# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA prefill with ``qk_rope_head_dim == 0``.

Some MLA checkpoints carry no RoPE component. GLM-5.3-Flash is one:
``qk_rope_head_dim=0`` with ``qk_nope_head_dim=256``, so ``qk_head_dim`` equals
``qk_nope_head_dim``.

``forward_mha`` splits ``latent_vec_k`` into ``(k_c_normed, k_pe)``, gives
``k_pe`` a head axis, and concatenates it onto ``k_nope``. ``latent_vec_k`` is
2-D there, so that head axis is an unsqueeze; spelling it as a reshape is what
breaks once ``k_pe`` is empty::

    k_pe.view(-1, 1, 0)
    RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 1, 0]
    because the unspecified dimension size -1 can be any value and is ambiguous

Nothing else on the path minds a zero-sized dim -- ``split``, ``expand`` and
``cat`` are all well defined, and ``k`` comes out contiguous either way. These
tests pin that down so the reshape does not come back.
"""

import pytest
import torch

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 256
V_HEAD_DIM = 256
NUM_HEADS = 16
TOKENS = 4


def _assemble_k(latent_vec_k, qk_rope_head_dim, kv_nope):
    """The k-assembly half of ``forward_mha``."""
    k_c_normed, k_pe = latent_vec_k.split([KV_LORA_RANK, qk_rope_head_dim], dim=-1)
    k_pe = k_pe.unsqueeze(1)

    k_nope, v = kv_nope.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
    k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
    return k_c_normed, k, v


@pytest.mark.parametrize("qk_rope_head_dim", [0, 64])
def test_k_assembly_handles_zero_rope_dim(qk_rope_head_dim):
    """k must be [tokens, heads, qk_nope + qk_rope] and contiguous either way."""
    latent = torch.randn(TOKENS, KV_LORA_RANK + qk_rope_head_dim)
    kv_nope = torch.randn(TOKENS, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)

    k_c_normed, k, v = _assemble_k(latent, qk_rope_head_dim, kv_nope)

    assert k_c_normed.shape == (TOKENS, KV_LORA_RANK)
    assert k.shape == (TOKENS, NUM_HEADS, QK_NOPE_HEAD_DIM + qk_rope_head_dim)
    assert v.shape == (TOKENS, NUM_HEADS, V_HEAD_DIM)
    assert k.is_contiguous()


def test_unsqueeze_is_the_reshape_without_the_ambiguity():
    """The head axis must come from unsqueeze: the reshape is what fails at 0."""
    k_pe = torch.randn(TOKENS, 0)
    with pytest.raises(RuntimeError, match="ambiguous"):
        k_pe.view(-1, 1, 0)
    assert k_pe.unsqueeze(1).shape == (TOKENS, 1, 0)

    # For a non-empty RoPE part the two spellings are the same tensor, which is
    # why the unsqueeze is safe to use unconditionally.
    k_pe = torch.randn(TOKENS, 64)
    assert torch.equal(k_pe.view(-1, 1, 64), k_pe.unsqueeze(1))
