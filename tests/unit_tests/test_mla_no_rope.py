# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA prefill with ``qk_rope_head_dim == 0``.

Some MLA checkpoints carry no RoPE component. GLM-5.3-Flash is one:
``qk_rope_head_dim=0`` with ``qk_nope_head_dim=256``, so ``qk_head_dim`` equals
``qk_nope_head_dim``.

``forward_mha`` builds its key and value with ``assemble_prefill_kv``, which
splits ``latent_vec_k`` into ``(k_c_normed, k_pe)``, gives ``k_pe`` a head
axis, and concatenates it onto ``k_nope``. ``latent_vec_k`` is 2-D there, so
that head axis is an unsqueeze; spelling it as a reshape is what breaks once
``k_pe`` is empty::

    k_pe.view(-1, 1, 0)
    RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 1, 0]
    because the unspecified dimension size -1 can be any value and is ambiguous

Nothing else on the path minds a zero-sized dim -- ``split``, ``expand`` and
``cat`` are all well defined, and ``k`` comes out contiguous either way. These
tests run the helper ``forward_mha`` calls, so the reshape cannot come back
without one of them failing.
"""

import pytest
import torch

from vllm_gaudi.attention.backends.mla_utils import assemble_prefill_kv

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 256
V_HEAD_DIM = 256
NUM_HEADS = 16
TOKENS = 4


class _Projection:
    """Stands in for ``kv_b_proj``: returns ``(output, bias)`` and records its input."""

    def __init__(self):
        self.weight = torch.randn(NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK)
        self.inputs = []

    def __call__(self, x):
        self.inputs.append(x)
        return x @ self.weight.t(), None


def _assemble(latent, qk_rope_head_dim, proj):
    return assemble_prefill_kv(latent, proj, KV_LORA_RANK, qk_rope_head_dim, NUM_HEADS, QK_NOPE_HEAD_DIM, V_HEAD_DIM)


@pytest.mark.parametrize("qk_rope_head_dim", [0, 64])
def test_k_assembly_handles_zero_rope_dim(qk_rope_head_dim):
    """k must be [tokens, heads, qk_nope + qk_rope] and contiguous either way."""
    latent = torch.randn(TOKENS, KV_LORA_RANK + qk_rope_head_dim)
    proj = _Projection()

    k, v = _assemble(latent, qk_rope_head_dim, proj)

    assert k.shape == (TOKENS, NUM_HEADS, QK_NOPE_HEAD_DIM + qk_rope_head_dim)
    assert v.shape == (TOKENS, NUM_HEADS, V_HEAD_DIM)
    assert k.is_contiguous()
    # The projection sees the compressed half of the latent and nothing else.
    assert len(proj.inputs) == 1
    assert torch.equal(proj.inputs[0], latent[:, :KV_LORA_RANK])


@pytest.mark.parametrize("qk_rope_head_dim", [0, 64])
def test_k_assembly_values(qk_rope_head_dim):
    """k is k_nope followed by k_pe broadcast across heads; v is the rest."""
    latent = torch.randn(TOKENS, KV_LORA_RANK + qk_rope_head_dim)
    proj = _Projection()

    k, v = _assemble(latent, qk_rope_head_dim, proj)

    kv_nope = (latent[:, :KV_LORA_RANK] @ proj.weight.t()).view(TOKENS, NUM_HEADS, -1)
    k_pe = latent[:, KV_LORA_RANK:]
    assert torch.equal(k[..., :QK_NOPE_HEAD_DIM], kv_nope[..., :QK_NOPE_HEAD_DIM])
    for head in range(NUM_HEADS):
        assert torch.equal(k[:, head, QK_NOPE_HEAD_DIM:], k_pe)
    assert torch.equal(v, kv_nope[..., QK_NOPE_HEAD_DIM:])


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
