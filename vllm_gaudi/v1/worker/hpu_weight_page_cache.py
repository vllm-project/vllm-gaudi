# SPDX-License-Identifier: Apache-2.0
"""Release the host page cache of checkpoint shards while the model weights are loaded.

The default loader reads every *.safetensors shard through mmap, so the page cache keeps all pages that were read
(for a 1.5 TB checkpoint ~700 GB per node with EP). That memory is reclaimable, but MemFree stays low until the
kernel needs it, and the Habana PyTorch bridge treats MemTotal - MemFree above
PT_HPU_HOST_MEMORY_THRESHOLD_PERCENT (90 by default) as host memory pressure and evicts its recipe cache, so every
graph is recompiled afterwards. The weights live on the HPU after loading and the shards are not read again, so
their cached pages are dropped with posix_fadvise(POSIX_FADV_DONTNEED) as soon as the loader is done with them.

Only vLLM's DefaultModelLoader safetensors path is covered (not the torchao / prefetch load strategies,
enable_multithread_load, fastsafetensors, instanttensor or .pt checkpoints).

Environment:
  VLLM_HPU_DROP_WEIGHT_PAGE_CACHE      unset/auto: active only when the safetensors shards together are larger
                                       than 25% of MemTotal (small checkpoints keep the upstream iterator,
                                       including its auto-prefetch); 1 forces it on, 0 keeps the upstream loader

The fadvise calls run in one daemon thread per process, so a slow call (all ranks of a node advise the same shard
at once, on network file systems that takes seconds) does not delay the loading of the next shard.
"""
import contextlib
import functools
import os
import queue
import threading
import time
from collections.abc import Callable, Generator, Iterator

import torch

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

_ENV_DROP = "VLLM_HPU_DROP_WEIGHT_PAGE_CACHE"
# Shards kept cached behind the current one: the ranks of a node read the same shards at about the same time, the lag
# lets slower ranks still hit the cache instead of re-reading a shard that a faster rank dropped.
_LAG = 4
_WAIT_S = 600.0  # the end of loading waits this long for the background drops
_AUTO_MIN_FRACTION = 0.25  # auto mode: active when the shards are larger than this fraction of MemTotal
# Strategies whose iteration state does not cross shard boundaries, so the iterator can be called once per shard.
# Kept on the upstream iterator: "torchao" (carries leftover tensors from one shard to the next) and "prefetch"
# (its background thread warms the cache of all shards at once, which is the opposite of this module).
_PER_SHARD_STRATEGIES = (None, "lazy", "eager")


def drop_mode() -> str:
    """'on' (forced), 'off' or 'auto' (decided from the checkpoint size) from VLLM_HPU_DROP_WEIGHT_PAGE_CACHE."""
    value = os.getenv(_ENV_DROP, "").strip().lower()
    if value in ("0", "false", "no", "off"):
        return "off"
    if value in ("1", "true", "yes", "on"):
        return "on"
    if value not in ("", "auto"):
        logger.warning("%s=%r is not 0/1/auto, using auto", _ENV_DROP, value)
    return "auto"


def _mem_total_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _files_size_bytes(files: list[str]) -> int:
    total = 0
    for path in files:
        with contextlib.suppress(OSError):
            total += os.path.getsize(path)
    return total


def drop_file_page_cache(path: str) -> bool:
    """Ask the kernel to forget the cached, unmapped pages of a file. Pages still mapped elsewhere stay."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        logger.debug("page cache of %s not dropped: %s", path, exc)
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError as exc:
        logger.debug("page cache of %s not dropped: %s", path, exc)
        return False
    finally:
        os.close(fd)


class _DropBatch:
    """Results of the background drops requested by one loader run."""

    def __init__(self):
        self._cond = threading.Condition()
        self.pending = 0
        self.dropped = 0
        self.failed = 0
        self.total_s = 0.0
        self.max_s = 0.0

    def submit(self):
        with self._cond:
            self.pending += 1

    def done(self, ok: bool, seconds: float):
        with self._cond:
            self.dropped += ok
            self.failed += not ok
            self.total_s += seconds
            self.max_s = max(self.max_s, seconds)
            self.pending -= 1
            self._cond.notify_all()

    def wait(self, timeout: float) -> bool:
        """Block until every submitted drop is done; False when the timeout expired first."""
        with self._cond:
            return self._cond.wait_for(lambda: self.pending == 0, timeout)


class _BackgroundDropper:
    """One lazily started daemon thread that runs drop_file_page_cache for queued shard paths."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._fallback_logged = False

    def submit(self, path: str, batch: _DropBatch) -> None:
        batch.submit()
        try:
            self._ensure_thread()
        except Exception as exc:  # no thread (e.g. interpreter shutdown): drop in place rather than lose the drop
            if not self._fallback_logged:  # once per process, not per shard
                self._fallback_logged = True
                logger.warning("Weight page cache: background drop thread not started (%s), dropping synchronously",
                               exc)
            self._run(path, batch)
            return
        self._queue.put((path, batch))

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="hpu-weight-page-cache-drop", daemon=True)
                self._thread.start()

    @staticmethod
    def _run(path: str, batch: _DropBatch) -> None:
        start, ok = time.monotonic(), False
        try:
            ok = drop_file_page_cache(path)
        except Exception as exc:  # never crash loading
            logger.warning("Weight page cache: drop of %s failed: %s", path, exc)
        batch.done(ok, time.monotonic() - start)

    def _loop(self) -> None:
        while True:
            path, batch = self._queue.get()
            try:
                self._run(path, batch)
            finally:
                self._queue.task_done()

    def wait_idle(self, timeout: float) -> bool:
        """Block until the queue is drained and the in-flight drop is done; False on timeout."""
        with self._queue.all_tasks_done:
            return self._queue.all_tasks_done.wait_for(lambda: self._queue.unfinished_tasks == 0, timeout)


_dropper = _BackgroundDropper()


def wrap_safetensors_iterator(original: Callable[..., Iterator[tuple[str, torch.Tensor]]],
                              seen: list[str] | None = None,
                              force: bool = True):
    """Return a drop-in replacement of safetensors_weights_iterator that drops the cache of consumed shards.

    Shards handled by the replacement are appended to ``seen``. Unless ``force`` is set, checkpoints not larger than
    _AUTO_MIN_FRACTION of MemTotal are passed to the upstream iterator untouched."""
    from vllm.model_executor.model_loader import weight_utils

    sort_key = getattr(weight_utils, "_natural_sort_key", None)
    use_tqdm = getattr(weight_utils, "enable_tqdm", lambda flag: flag)
    bar_format = getattr(weight_utils, "_BAR_FORMAT", None)

    @functools.wraps(original)
    def iterator(hf_weights_files: list[str],
                 use_tqdm_on_load: bool,
                 safetensors_load_strategy: str | None = None,
                 *args,
                 **kwargs) -> Generator[tuple[str, torch.Tensor], None, None]:
        if safetensors_load_strategy not in _PER_SHARD_STRATEGIES:
            yield from original(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy, *args, **kwargs)
            return
        if not force:
            total, mem_total = _files_size_bytes(hf_weights_files), _mem_total_bytes()
            if mem_total and total <= _AUTO_MIN_FRACTION * mem_total:
                logger.info(
                    "Weight page cache: checkpoint shards %.1f GiB <= %d%% of MemTotal (%.1f GiB), "
                    "upstream loader kept", total / 2**30, 100 * _AUTO_MIN_FRACTION, mem_total / 2**30)
                yield from original(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy, *args, **kwargs)
                return
            if not mem_total:
                logger.info("Weight page cache: MemTotal unknown, dropping shards (%.1f GiB)", total / 2**30)
        from tqdm.auto import tqdm

        # None lets the upstream iterator size the whole checkpoint and may start a prefetch (plus an info_once log)
        # per call; "lazy" is the plain safe_open path it takes otherwise, without that per-shard side effect.
        if safetensors_load_strategy is None:
            safetensors_load_strategy = "lazy"
        files = sorted(hf_weights_files, key=sort_key) if sort_key else sorted(hf_weights_files)
        if seen is not None:
            seen.extend(files)
        lag = _LAG
        logger.info("Weight page cache: dropping shards after use (lag %d of %d shards, %.1f GiB)", lag, len(files),
                    _files_size_bytes(files) / 2**30)
        batch = _DropBatch()
        started = 0  # shards opened so far
        next_drop = 0  # first shard whose cache is not dropped yet
        try:
            for i, st_file in enumerate(
                    tqdm(files,
                         desc="Loading safetensors checkpoint shards",
                         disable=not use_tqdm(use_tqdm_on_load),
                         bar_format=bar_format)):
                started = i + 1
                # One shard per call: the shard's safe_open handle (and its mmap) is closed when the call ends.
                yield from original([st_file], False, safetensors_load_strategy, *args, **kwargs)
                while next_drop <= i - lag:
                    _dropper.submit(files[next_drop], batch)
                    next_drop += 1
        finally:
            # Also runs when loading stops early, so no opened shard stays cached because of this process.
            for st_file in files[next_drop:started]:
                _dropper.submit(st_file, batch)
            if not batch.wait(_WAIT_S):
                logger.warning("Weight page cache: %d background drops still pending after %.0f s", batch.pending,
                               _WAIT_S)
            logger.info(
                "Weight page cache: loader finished, fadvise(DONTNEED) called for %d shards (%d failed); "
                "background fadvise time total %.1f s, max %.2f s per shard", batch.dropped, batch.failed,
                batch.total_s, batch.max_s)

    return iterator


@contextlib.contextmanager
def drop_weight_page_cache_during_load():
    """Use the page-cache-dropping iterator in vLLM's default loader for the duration of the block.

    DONTNEED skips pages that are still mapped, so the shards seen are advised once more when the block ends, i.e.
    after the loader's post-processing released its mappings and CPU tensors."""
    mode = drop_mode()
    if mode == "off":
        yield
        return
    try:
        from vllm.model_executor.model_loader import default_loader
        original = default_loader.safetensors_weights_iterator
    except (ImportError, AttributeError) as exc:
        logger.warning("Weight page cache drop disabled, default loader not patchable: %s", exc)
        yield
        return
    seen: list[str] = []
    default_loader.safetensors_weights_iterator = wrap_safetensors_iterator(original, seen, force=mode == "on")
    try:
        yield
    finally:
        default_loader.safetensors_weights_iterator = original
        if seen:
            try:  # must never replace an exception raised by get_model
                calls = sum(drop_file_page_cache(path) for path in seen)
                logger.info("Weight page cache: final fadvise(DONTNEED) called for %d/%d shards", calls, len(seen))
            except Exception as exc:
                logger.warning("Weight page cache: final drop failed: %s", exc)
