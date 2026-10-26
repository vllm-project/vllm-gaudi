# SPDX-License-Identifier: Apache-2.0
"""Whether the current model forward runs outside of HPU graphs (pure python, no vLLM / Habana imports).

``mark_step`` inside an HPU graph capture breaks the capture, so model code that wants to cut the lazy graph
(Kimi-K3 SiTU experts, chunked KDA prefill) asks ``lazy_step_hook`` for the hook. The runner publishes the
result of ``_use_graphs`` around every ``model.forward`` with ``forward_graph_mode``; when nothing was published
(unknown) no mark_step is issued.
"""

from __future__ import annotations

import contextlib
import os
from typing import Callable

_without_graphs: bool | None = None  # None = unknown


@contextlib.contextmanager
def forward_graph_mode(use_graphs: bool):
    """Publish for the duration of one forward whether it runs under HPU graphs."""
    global _without_graphs
    previous = _without_graphs
    _without_graphs = not use_graphs
    try:
        yield
    finally:
        _without_graphs = previous


def forward_without_graphs() -> bool:
    """True only when the runner said that this forward bypasses HPU graphs."""
    return _without_graphs is True


def mark_step_enabled() -> bool:
    """HPU_KIMI_SITU_MARK_STEP=0 turns the mark_step cuts off (default on)."""
    return os.environ.get("HPU_KIMI_SITU_MARK_STEP", "1").strip().lower() not in ("0", "false", "off")


def lazy_step_hook() -> Callable[[], None] | None:
    """``htorch.core.mark_step`` in lazy mode and outside HPU graphs, else None.

    Habana frameworks are imported on first use at runtime, never at plugin load.
    """
    if not mark_step_enabled() or not forward_without_graphs():
        return None
    if os.environ.get("PT_HPU_LAZY_MODE", "1").strip() == "0":
        return None  # torch.compile mode: no lazy graph to cut, and nothing Habana-specific to trace under dynamo
    try:
        import habana_frameworks.torch as htorch
        return htorch.core.mark_step if htorch.utils.internal.is_lazy() else None
    except ImportError:
        return None


def device_sync_hook() -> Callable[[], None] | None:
    """Host-side wait for all queued HPU work (diagnostic timings), or None where waiting is not possible.

    Lazy mode needs a mark_step to launch the pending graph, which is only allowed outside HPU graphs; compile /
    eager mode only waits (callers must run outside the compiled region, e.g. under ``torch.compiler.disable``).
    """
    try:
        import habana_frameworks.torch as htorch
    except ImportError:
        return None
    if not htorch.utils.internal.is_lazy():
        return htorch.hpu.synchronize
    if not forward_without_graphs():
        return None

    def sync():
        htorch.core.mark_step()
        htorch.hpu.synchronize()

    return sync
