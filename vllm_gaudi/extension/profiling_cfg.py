# SPDX-License-Identifier: Apache-2.0
"""Sizing of the scheduler token budget for the profile-one-synthetic-batch mode (no vLLM / Habana imports)."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import time
from typing import Callable

# under the 'vllm' logger tree: vLLM attaches its handler there, other names would drop the INFO lines
_logger = logging.getLogger('vllm.gaudi.profiling')


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


def bucketing_max_batched_tokens(tokens: int, mamba_chunk_size: int, profiling: bool) -> int:
    """Token budget handed to the bucketing manager.

    HPUBucketingManager.initialize raises unless ``max_num_batched_tokens`` is divisible by the mamba-like chunk
    size (128 for KDA/GDN models without an explicit one). A mixed profile budget (``bs * query + decode_bs``,
    e.g. 8192 + 1) is not, which killed the engine right after start (build #859). Only the profile mode rounds
    up; a misconfigured normal serve still fails loudly.
    """
    if not profiling or mamba_chunk_size <= 0:
        return tokens
    return -(-tokens // mamba_chunk_size) * mamba_chunk_size


def make_trace_handler(output_dir: str, rank: int, use_gzip: bool = True) -> Callable:
    """``on_trace_ready`` callback writing ``<output_dir>/<rank>_<pid>_<ms>.pt.trace.json[.gz]``.

    ``torch.profiler.tensorboard_trace_handler(..., use_gzip=True)`` calls ``prof.export_chrome_trace(path, True)``
    which the Habana profiler wrapper does not accept (TypeError, no trace written). This handler calls
    ``export_chrome_trace`` with the path only and gzips in Python. Failures are logged, never raised.
    """

    def handler(prof) -> None:
        path = os.path.join(output_dir, f"{rank}_{os.getpid()}_{int(time.time() * 1000)}.pt.trace.json")
        try:
            os.makedirs(output_dir, exist_ok=True)
            prof.export_chrome_trace(path)
            if use_gzip:
                with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.remove(path)
                path += ".gz"
            _logger.info("Profiler trace written on rank %d: %s (%d bytes)", rank, path, os.path.getsize(path))
        except Exception as exc:
            _logger.warning("Cannot write the profiler trace on rank %d (%s): %r", rank, path, exc)

    return handler
