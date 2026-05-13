"""Runtime monkey-patches applied when the HPU plugin is loaded.

Currently:

* ``vllm.distributed.parallel_state.cleanup_dist_env_and_memory`` — upstream
  (since vllm PR #34328) calls ``torch.accelerator.empty_cache()``, which
  requires the device's allocator to be a ``c10::DeviceAllocator``. HPU's
  allocator does not implement that interface, so the call raises
  ``RuntimeError: Allocator for hpu is not a DeviceAllocator`` during pytest
  fixture teardown (see GAUDISW-247825). We replace it with an HPU-safe
  variant that uses ``current_platform.empty_cache()`` instead.
"""

import functools
import gc
from unittest.mock import MagicMock, patch

import torch

from vllm import envs
from vllm.distributed import parallel_state
from vllm.platforms import current_platform


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


def apply() -> None:
    """Install all HPU runtime monkey-patches."""
    # Patch the canonical definition.
    parallel_state.cleanup_dist_env_and_memory = _hpu_cleanup_dist_env_and_memory
    # Patch the re-export from ``vllm.distributed`` so ``from vllm.distributed
    # import cleanup_dist_env_and_memory`` (used by the upstream pytest
    # conftest) also picks up the HPU-safe version.
    import vllm.distributed as _vllm_distributed

    _vllm_distributed.cleanup_dist_env_and_memory = _hpu_cleanup_dist_env_and_memory


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

    hf3fs_mock_client.Hf3fsClient.batch_write = _safe_batch_write
