# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA helpers that need nothing but torch.

Kept apart from ``hpu_attn`` so they can be imported, and tested, on a machine
without ``habana_frameworks``.
"""

from collections.abc import Callable

import torch


def assemble_prefill_kv(
    latent_vec_k: torch.Tensor,
    kv_b_proj: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]],
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the prefill key and value from the MLA latent.

    ``latent_vec_k`` is ``[tokens, kv_lora_rank + qk_rope_head_dim]``, and
    ``kv_b_proj`` returns ``(output, bias)`` like ``ColumnParallelLinear``.
    Returns ``k`` as ``[tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]``
    and ``v`` as ``[tokens, num_heads, v_head_dim]``.
    """
    k_c_normed, k_pe = latent_vec_k.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    # latent_vec_k is 2-D, so this is an unsqueeze. Spelling it as one keeps
    # MLA variants without a RoPE component working: they leave k_pe empty,
    # and view(-1, 1, 0) is ambiguous because a zero in the shape makes the
    # element count 0 and -1 unsolvable.
    k_pe = k_pe.unsqueeze(1)

    kv_nope = kv_b_proj(k_c_normed)[0].view(-1, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope, v = kv_nope.split([qk_nope_head_dim, v_head_dim], dim=-1)

    k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)
    return k, v
