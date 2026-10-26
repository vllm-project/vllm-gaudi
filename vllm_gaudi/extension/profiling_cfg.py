# SPDX-License-Identifier: Apache-2.0
"""Sizing of the scheduler token budget for the profile-one-synthetic-batch mode (no vLLM / Habana imports)."""

from __future__ import annotations


def profiling_max_batched_tokens(prompt_cfg: tuple | None, decode_cfg: tuple | None, current: int) -> int:
    """Token budget (``max_num_batched_tokens``) that fits the dummy batch of VLLM_PROFILE_PROMPT / _DECODE.

    ``prompt_cfg`` is ``(bs, query, ctx_blocks)``, ``decode_cfg`` ``(bs, 1, blocks)``. The prompt requests
    schedule ``bs * query`` tokens and every decode request one more token, so a mixed step (both variables set)
    needs ``bs * query + decode_bs``; sizing it as ``bs * query`` alone made the 8193rd token overflow the
    position buffer in ``_prepare_inputs``. A decode-only profile never shrinks the budget below its batch size.
    """
    tokens = current
    if prompt_cfg:
        tokens = prompt_cfg[0] * prompt_cfg[1]
    if decode_cfg:
        tokens = tokens + decode_cfg[0] if prompt_cfg else max(tokens, decode_cfg[0])
    return tokens
