"""Runtime monkey-patches applied when the HPU plugin is loaded.

Currently:

* ``torch.accelerator.empty_cache`` — HPU's allocator does not implement the
  ``c10::DeviceAllocator`` interface, so the upstream helper raises
  ``RuntimeError: Allocator for hpu is not a DeviceAllocator``.  We replace
  it with an HPU-safe variant that routes through
  ``current_platform.empty_cache()`` (a no-op on HPU).  This also makes the
  ``cleanup_dist_env_and_memory`` patch resilient to import-order issues.

* ``torch._C._host_emptyCache`` — does not exist on HPU; we install a no-op
  stub to prevent ``AttributeError`` in ``cleanup_dist_env_and_memory``.

* ``vllm.distributed.parallel_state.cleanup_dist_env_and_memory`` — upstream
  (since vllm PR #34328) calls ``torch.accelerator.empty_cache()``, which
  requires the device's allocator to be a ``c10::DeviceAllocator``.  We
  replace it with an HPU-safe variant that uses
  ``current_platform.empty_cache()`` instead (see GAUDISW-247825).

* ``vllm.v1.sample.ops.logprobs.batched_count_greater_than`` — upstream
  decorates this function with ``@torch.compile(dynamic=True, ...)``.
  Habana's ``recipe_compiler`` backend cannot handle the symbolic shapes
  produced by ``dynamic=True`` (and by ``mark_unbacked`` in the caller),
  raising ``TypeError: Cannot convert symbols to int``.  We replace it
  with a plain (uncompiled) version of the same function.  The replacement
  is deferred to ``load_general_plugins`` time to avoid importing
  ``vllm.v1.sample.sampler`` during early plugin registration, which would
  trigger a heavy import chain that interferes with platform initialisation.
"""

import functools
import gc
from unittest.mock import MagicMock, patch

import torch

from vllm import envs
from vllm.distributed import parallel_state
from vllm.platforms import current_platform


def _hpu_accelerator_empty_cache() -> None:
    """HPU-safe replacement for ``torch.accelerator.empty_cache()``.

    HPU's allocator does not implement the ``c10::DeviceAllocator``
    interface, so the upstream ``torch.accelerator.empty_cache()`` raises
    ``RuntimeError``.  Route through ``current_platform.empty_cache``
    instead (which is ``None`` on HPU, making this a no-op).
    """
    empty_cache = current_platform.empty_cache
    if empty_cache is not None:
        empty_cache()


def _hpu_cleanup_dist_env_and_memory(shutdown_ray: bool = False) -> None:
    """HPU-safe replacement for ``cleanup_dist_env_and_memory``.

    Mirrors the upstream implementation but routes the device-side cache
    release through ``current_platform.empty_cache()`` instead of
    ``torch.accelerator.empty_cache()`` (which is incompatible with the
    HPU allocator).
    """
    # Reset environment variable cache
    envs.disable_envs_cache()
    # Ensure all objects are not frozen before cleanup
    gc.unfreeze()

    parallel_state.destroy_model_parallel()
    parallel_state.destroy_distributed_environment()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    gc.collect()

    empty_cache = current_platform.empty_cache
    if empty_cache is not None:
        empty_cache()
    try:
        if not current_platform.is_cpu():
            torch._C._host_emptyCache()
    except AttributeError:
        parallel_state.logger.warning("torch._C._host_emptyCache() only available in Pytorch >=2.5")


def _hpu_batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """HPU-safe replacement for ``batched_count_greater_than``.

    Identical logic to the upstream implementation but *not* wrapped in
    ``torch.compile``.  The upstream decorator uses ``dynamic=True`` whose
    symbolic shapes are incompatible with Habana's ``recipe_compiler``
    backend, and ``mark_unbacked`` in the caller prevents ``dynamic=False``
    from helping.
    """
    return (x >= values).sum(-1)


def _patch_batched_count_greater_than() -> None:
    """Replace ``batched_count_greater_than`` in the sampler & logprobs modules.

    Called from the ``load_general_plugins`` hook so that the heavy
    ``vllm.v1.sample.*`` import chain runs *after* platform initialisation.
    """
    import vllm.v1.sample.ops.logprobs as _logprobs_mod
    import vllm.v1.sample.sampler as _sampler_mod

    _logprobs_mod.batched_count_greater_than = _hpu_batched_count_greater_than
    _sampler_mod.batched_count_greater_than = _hpu_batched_count_greater_than


def apply() -> None:
    """Install all HPU runtime monkey-patches."""
    # --- torch.accelerator.empty_cache ---
    torch.accelerator.empty_cache = _hpu_accelerator_empty_cache

    # --- torch._C._host_emptyCache ---
    if not hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache = lambda: None

    # --- cleanup_dist_env_and_memory ---
    parallel_state.cleanup_dist_env_and_memory = _hpu_cleanup_dist_env_and_memory
    import vllm.distributed as _vllm_distributed

    _vllm_distributed.cleanup_dist_env_and_memory = _hpu_cleanup_dist_env_and_memory

    # --- batched_count_greater_than (deferred) ---
    # We cannot import the sampler modules here because the import chain
    # triggers platform re-initialisation ("Device string must not be
    # empty").  Instead we hook into ``load_general_plugins`` which runs
    # in every process (parent + EngineCore subprocess) after the platform
    # is ready.
    import vllm.plugins as _plugins_mod

    _original_load_general = _plugins_mod.load_general_plugins

    def _load_general_with_hpu_patches():
        _original_load_general()
        _patch_batched_count_greater_than()

    _plugins_mod.load_general_plugins = _load_general_with_hpu_patches


def patch_hf3fs_mock_client():
    """Guard CUDA sync in the HF3FS mock client on non-CUDA platforms.

    The upstream mock client's ``batch_write`` unconditionally calls
    ``torch.cuda.current_stream().wait_event(event)``, which raises
    ``RuntimeError`` on platforms without CUDA (e.g. HPU).  We wrap
    ``batch_write`` to stub ``torch.cuda.current_stream`` with a no-op
    mock for the duration of the call.

    Called from ``register_utils()`` (general plugin) rather than
    ``apply()`` (platform plugin) to avoid circular imports — the mock
    client transitively imports ``vllm.config`` which is not yet fully
    initialized during platform registration.
    """
    if torch.cuda.is_available():
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils import (
            hf3fs_mock_client, )
    except ImportError:
        return

    _orig_batch_write = hf3fs_mock_client.Hf3fsClient.batch_write

    @functools.wraps(_orig_batch_write)
    def _safe_batch_write(self, offsets, tensors, event):
        with patch("torch.cuda.current_stream", return_value=MagicMock()):
            return _orig_batch_write(self, offsets, tensors, event)

    hf3fs_mock_client.Hf3fsClient.batch_write = _safe_batch_write  # type: ignore[method-assign]
