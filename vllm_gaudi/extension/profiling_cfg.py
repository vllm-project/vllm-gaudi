# SPDX-License-Identifier: Apache-2.0
"""Sizing of the scheduler token budget for the profile-one-synthetic-batch mode (no vLLM / Habana imports)."""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import time
from typing import Callable, Mapping, Optional

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


# Paths of the traces written by make_trace_handler in this process (logged by hold_before_exit).
WRITTEN_TRACES: list = []


def profile_exit_delay_s(environ: Optional[Mapping[str, str]] = None) -> float:
    """VLLM_PROFILE_EXIT_DELAY_S: seconds every rank holds after the traces are written, before the run ends with
    'Finished profiling' (0 = no hold). Gives time to copy the traces off the pod; invalid/negative values mean 0."""
    environ = os.environ if environ is None else environ
    try:
        return max(0.0, float(environ.get('VLLM_PROFILE_EXIT_DELAY_S', '0') or 0))
    except ValueError:
        return 0.0


def hold_before_exit(delay_s: float, sleep: Callable[[float], None] = time.sleep) -> None:
    """Log the written trace paths at WARNING and sleep ``delay_s`` s (no-op for 0).

    Called on every rank after the post-profiling barrier with the same delay, so ranks stay aligned.
    """
    if delay_s <= 0:
        return
    _logger.warning("Profiling traces ready: %s; holding %g s before exit",
                    ", ".join(WRITTEN_TRACES) or "none written on this rank", delay_s)
    sleep(delay_s)


def _remove_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _export_trace(prof, path: str) -> None:
    """Write the chrome trace of ``prof`` to ``path`` with a fallback chain.

    ``torch.profiler.profile.export_chrome_trace`` always calls ``self.profiler.export_chrome_trace(path,
    self._trace_metadata)`` (torch/profiler/profiler.py), but the Habana ``habana_autograd_profile_wrapper``
    (habana_frameworks/torch/activity_profiler/activity_profiler.py) only takes ``(self, path)``, hence
    ``TypeError: ... takes 2 positional arguments but 3 were given`` for any outer call. So call the inner
    profiler directly with the path only; fall back to the outer method and to the raw kineto results.
    """
    errors = []
    inner = getattr(prof, "profiler", None)
    attempts = []
    if inner is not None and hasattr(inner, "export_chrome_trace"):
        attempts.append(("profiler.export_chrome_trace", lambda: inner.export_chrome_trace(path)))
    attempts.append(("export_chrome_trace", lambda: prof.export_chrome_trace(path)))
    kineto = getattr(inner, "kineto_results", None)
    if kineto is not None and hasattr(kineto, "save"):
        attempts.append(("kineto_results.save", lambda: kineto.save(path)))
    for name, attempt in attempts:
        _remove_if_exists(path)  # a partial file of a failed attempt must never count as a trace
        try:
            attempt()
            if os.path.exists(path):
                return
            errors.append(f"{name}: no file written")
        except Exception as exc:
            errors.append(f"{name}: {exc!r}")
    _remove_if_exists(path)
    raise RuntimeError("; ".join(errors))


def make_trace_handler(output_dir: str, rank: int, use_gzip: bool = True) -> Callable:
    """``on_trace_ready`` callback writing ``<output_dir>/<rank>_<pid>_<ms>.pt.trace.json[.gz]``.

    ``torch.profiler.tensorboard_trace_handler`` goes through ``prof.export_chrome_trace``, which passes extra
    arguments the Habana profiler wrapper does not accept (see ``_export_trace``). This handler writes the plain
    trace through ``_export_trace`` and gzips in Python. Failures are logged, never raised.
    """

    def handler(prof) -> None:
        path = os.path.join(output_dir, f"{rank}_{os.getpid()}_{int(time.time() * 1000)}.pt.trace.json")
        try:
            os.makedirs(output_dir, exist_ok=True)
            _export_trace(prof, path)
            if use_gzip:
                with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.remove(path)
                path += ".gz"
            WRITTEN_TRACES.append(path)
            _logger.info("Profiler trace written on rank %d: %s (%d bytes)", rank, path, os.path.getsize(path))
        except Exception as exc:
            _logger.warning("Cannot write the profiler trace on rank %d (%s): %r", rank, path, exc)

    return handler
