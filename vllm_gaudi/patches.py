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

* ``torch.accelerator.empty_host_cache`` — vllm PR #51107 switched
  ``cleanup_dist_env_and_memory`` from ``torch._C._host_emptyCache()`` to
  ``torch.accelerator.empty_host_cache()``.  On torch>=2.9 the attribute exists,
  so vllm's ``except AttributeError`` guard no longer catches it; the call
  dispatches to ``at::accelerator::emptyHostCache()``, which has no HPU host-cache
  hook and segfaults at teardown.  We replace it with a no-op (sibling of the
  ``empty_cache`` patch above).

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

* ``vllm.v1.sample.sampler.Sampler.gather_logprobs`` — upstream PR #38933
  added two ``mark_unbacked()`` calls inside ``gather_logprobs`` to prevent
  0->1 batch-size specialization recompiles for ``batched_count_greater_than``
  on CUDA.  On HPU, ``torch._dynamo`` marks ``mark_unbacked`` as a forbidden
  callable (``is_forbidden=True``), so tracing a compiled ``Sampler`` raises
  ``AssertionError: Attempt to trace forbidden callable mark_unbacked`` for
  any request with ``logprobs=true``.  We replace ``gather_logprobs`` with an
  identical implementation that omits the two ``mark_unbacked`` calls.  HPU
  handles dynamic batch shapes without this hint.

* ``vllm.distributed.device_communicators.shm_broadcast.check_shm_free_space``
  — upstream PR #48879 ("Fail fast when /dev/shm is too small for the shm ring
  buffer") added a pre-allocation guard that compares the ring buffer's *full
  nominal* size (default ``24 MiB x 10 = 240 MiB`` per ``MessageQueue``) against
  the free space on ``/dev/shm`` and raises ``RuntimeError: Insufficient space
  in /dev/shm`` when it does not fit. HPU CI containers expose the Docker
  default 64 MiB ``/dev/shm``, where the pre-#48879 code path worked fine: the
  segment is created with ``SharedMemory(create=True, size=...)`` (an
  ``ftruncate`` on tmpfs, which allocates pages lazily on write), and the TP2 /
  DP / multi-proc EngineCore workers only ever touch a small fraction of the
  nominal ring. The new eager check aborts EngineCore start-up before any page
  is touched, breaking every shared-memory-broadcast config
  (``run_data_parallel_test``, ``run_tp2_load_generate_test``,
  ``run_deepseek_v2_inc_dynamic_tp2_load_generate_test``,
  ``run_gsm8k_qwen3_30b_test``). Restore the pre-#48879 behaviour on HPU by
  replacing the guard with a no-op so the lazy tmpfs allocation proceeds.

  Upstream ref: https://github.com/vllm-project/vllm/pull/48879

* ``vllm.v1.core.block_pool.BlockPool.free_blocks`` — upstream PR #42656
  ("Apply LRU policy only to proper cache entries") made ``free_blocks``
  partition freed blocks into with-hash / without-hash lists and issue two
  queue ops (``prepend_n`` + ``append_n``) on every engine step.  When prefix
  caching is disabled every block's hash is ``None``, so the split is a no-op
  that only adds per-step CPU overhead — a measurable decode-throughput loss
  on small/low-batch models.  We restore a single-pass free for the
  ``enable_caching=False`` case and delegate to the original implementation
  when prefix caching is on.  Remove once fixed upstream.

* ``vllm.model_executor.layers.mamba.abstract.MambaBase.bind_kv_cache`` —
  upstream PR #44456 unpacks a single packed int8 page via
  ``kv_cache.squeeze(dim=(1, 2))``.  The HPU runner allocates a separate
  tensor per Mamba state and hands the layer a ready-made tuple, so the
  upstream method raises ``AttributeError: 'tuple' object has no attribute
  'squeeze'`` at EngineCore init.  We replace it with a variant that assigns
  the pre-split tuple directly (restoring the pre-#44456 contract).  Regressed
  when the 0.26.0 sliding-window port dropped the original patch.

* ``transformers.integrations.sdpa_attention.sdpa_attention_forward`` — the
  transformers library's SDPA attention uses ``F.scaled_dot_product_attention``
  which routes to suboptimal kernels on HPU. We replace it with an HPU-optimized
  version that uses FusedSDPA when the ``fsdpa_impl`` attention implementation
  is configured, providing 1.4-2x performance improvement for models that use
  transformers SDPA attention (e.g., vision encoders in Gemma4 models).
"""

import functools
import gc
import inspect
import os
from typing import Callable, Optional

import torch

from vllm import envs

# NOTE: neither ``vllm.platforms.current_platform`` nor
# ``vllm.distributed.parallel_state`` is imported at module top level — both
# force re-entrant platform resolution *while this module is being imported by
# ``vllm_gaudi.register()``*, leaving ``vllm_gaudi`` partially initialized so
# the HPU platform plugin is silently dropped and vLLM falls back to
# ``UnspecifiedPlatform`` ("RuntimeError: Failed to infer device type / Device
# string must not be empty"). Concretely:
#
# * ``current_platform`` is a lazily-resolved attribute (see
#   ``vllm/platforms/__init__.py.__getattr__``): importing it eagerly runs
#   ``resolve_current_platform_cls_qualname()`` directly.
# * ``parallel_state`` transitively imports ``vllm.utils.torch_utils``, whose
#   module-level ``PIN_MEMORY = is_pin_memory_available()`` (vllm PR #45424)
#   resolves ``current_platform`` at import time — the same re-entry, one hop
#   removed.
#
# Both are therefore imported lazily inside the functions that use them, and
# the ``cleanup_dist_env_and_memory`` patch (which needs ``parallel_state``)
# is deferred to ``load_general_plugins`` time rather than ``apply()``
# (platform-registration time). See GAUDISW-249622.


def _hpu_accelerator_empty_cache() -> None:
    """HPU-safe replacement for ``torch.accelerator.empty_cache()``.

    HPU's allocator does not implement the ``c10::DeviceAllocator``
    interface, so the upstream ``torch.accelerator.empty_cache()`` raises
    ``RuntimeError``.  Route through ``current_platform.empty_cache``
    instead (which is ``None`` on HPU, making this a no-op).
    """
    from vllm.platforms import current_platform

    empty_cache = current_platform.empty_cache
    if empty_cache is not None:
        empty_cache()


def _hpu_accelerator_empty_host_cache() -> None:
    """HPU-safe replacement for ``torch.accelerator.empty_host_cache()``.

    HPU has no host-cache-release hook; the upstream C++ path
    (``at::accelerator::emptyHostCache()``) segfaults at teardown. vLLM PR
    #51107 swapped ``torch._C._host_emptyCache()`` for
    ``torch.accelerator.empty_host_cache()`` in ``cleanup_dist_env_and_memory``,
    and on torch>=2.9 the attribute exists so the upstream
    ``except AttributeError`` guard no longer catches it.  Make it a no-op.
    """
    return


def _patch_hf3fs_mock_client_for_cpu_only() -> None:
    """Patch HF3FS mock client to avoid CUDA stream waits on CPU-only builds.

    Upstream mock client unconditionally calls
    ``torch.cuda.current_stream().wait_event(event)`` in ``batch_write``.
    In environments where PyTorch is not compiled with CUDA, that path throws
    and the method returns ``-1`` for writes, causing connector unit tests to
    fail. This patch keeps the same behavior but skips CUDA synchronization when
    CUDA is unavailable.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils import hf3fs_mock_client as _mock_mod
    except Exception:
        # Keep plugin load resilient if the module path changes or is missing.
        return

    client_cls = getattr(_mock_mod, "Hf3fsClient", None)
    if client_cls is None:
        return

    original_batch_write = getattr(client_cls, "batch_write", None)
    if original_batch_write is None:
        return

    if getattr(original_batch_write, "_vllm_gaudi_cpu_safe_patch", False):
        return

    def _batch_write_cpu_safe(self, offsets, tensors, event):
        if torch.cuda.is_available():
            return original_batch_write(self, offsets, tensors, event)

        results = []
        try:
            data_bytes_list = [self._tensor_to_bytes(tensor) for tensor in tensors]

            with open(self._file_path, "r+b") as f:
                for offset, data_bytes in zip(offsets, data_bytes_list):
                    if offset < 0 or offset + len(data_bytes) > self._size:
                        results.append(-1)
                        continue

                    f.seek(offset)
                    bytes_written = f.write(data_bytes)

                    if bytes_written == len(data_bytes) == self._bytes_per_page:
                        results.append(self._bytes_per_page)
                    else:
                        _mock_mod.logger.error(
                            "Write size mismatch: wrote %d, expected %d",
                            bytes_written,
                            self._bytes_per_page,
                        )
                        results.append(-1)
        except Exception as e:
            _mock_mod.logger.error("Batch write error: %s", e)
            results.extend([-1] * (len(offsets) - len(results)))

        return results

    _batch_write_cpu_safe._vllm_gaudi_cpu_safe_patch = True  # type: ignore[attr-defined]
    client_cls.batch_write = _batch_write_cpu_safe


def _hpu_cleanup_dist_env_and_memory(shutdown_ray: bool = False) -> None:
    """HPU-safe replacement for ``cleanup_dist_env_and_memory``.

    Mirrors the upstream implementation but routes the device-side cache
    release through ``current_platform.empty_cache()`` instead of
    ``torch.accelerator.empty_cache()`` (which is incompatible with the
    HPU allocator).
    """
    from vllm.distributed import parallel_state
    from vllm.platforms import current_platform

    # Re-apply lazy runtime patches that may depend on import timing.
    _patch_hf3fs_mock_client_for_cpu_only()

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


def _hpu_gather_logprobs(
    logprobs: torch.Tensor,
    num_logprobs: int,
    token_ids: torch.Tensor,
):
    """HPU-safe replacement for ``Sampler.gather_logprobs``.

    Identical logic to the upstream implementation (vllm PR #38933) but with
    the two ``mark_unbacked()`` calls removed.  On HPU, ``mark_unbacked`` is a
    forbidden callable in ``torch._dynamo`` (``is_forbidden=True``).  When the
    compiled ``Sampler`` traces ``gather_logprobs``, hitting ``mark_unbacked``
    raises ``AssertionError: Attempt to trace forbidden callable``.  HPU does
    not need this CUDA-specific recompile hint.

    Upstream ref: https://github.com/vllm-project/vllm/pull/38933
    """
    from vllm.v1.outputs import LogprobsTensors
    import vllm.v1.sample.sampler as _sampler_mod

    assert token_ids.dtype == torch.int64
    topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
    token_ids = token_ids.unsqueeze(-1)
    token_logprobs = logprobs.gather(-1, token_ids)
    # mark_unbacked calls intentionally omitted — forbidden on HPU dynamo.
    token_ranks = _sampler_mod.batched_count_greater_than(logprobs, token_logprobs)
    indices = torch.cat((token_ids, topk_indices), dim=1)
    logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
    indices = indices.to(torch.int32)
    return LogprobsTensors(indices, logprobs, token_ranks)


def _patch_gather_logprobs() -> None:
    """Replace ``Sampler.gather_logprobs`` with the HPU-safe variant.

    Called from the ``load_general_plugins`` hook (same as
    ``_patch_batched_count_greater_than``) so that ``vllm.v1.sample.*``
    imports run after platform initialisation.

    Guarded by ``inspect.getsource`` so this is a no-op on vLLM versions
    that predate PR #38933 (i.e. where ``gather_logprobs`` does not call
    ``mark_unbacked``).
    """
    import inspect

    import vllm.v1.sample.sampler as _sampler_mod

    if "mark_unbacked" not in inspect.getsource(_sampler_mod.Sampler.gather_logprobs):
        return  # Not affected — older vLLM without PR #38933.

    _sampler_mod.Sampler.gather_logprobs = staticmethod(  # type: ignore[method-assign]
        _hpu_gather_logprobs)


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


_GRANITE_LEGACY_LAYER_ALIASES = {
    "full_attention": "attention",
    "linear_attention": "mamba",
}


def _patch_granite_hybrid_layer_types() -> None:
    """Accept transformers>=5 remapped hybrid layer-type names for GraniteMoeHybrid.

    transformers>=5 rewrites the legacy hybrid ``layer_types`` values on config
    load ("attention" -> "full_attention", "mamba" -> "linear_attention"; see
    ``remap_legacy_layer_types`` in transformers ``configuration_utils.py``).
    Hub checkpoints still store the legacy names, so the rename happens in
    memory. vLLM's ``ALL_DECODER_LAYER_TYPES`` in ``granitemoehybrid.py`` only
    keys the legacy names, so the remapped values raise
    ``KeyError: 'linear_attention'`` at ``get_layer`` during model construction
    (before any HPU kernel runs).

    Guarded so this is a no-op once the value is already keyed (e.g. upstream
    vLLM PR #47634 merged, or the module shape changed).

    Upstream ref: https://github.com/vllm-project/vllm/pull/47634
    """
    import vllm.model_executor.models.granitemoehybrid as _granite_mod

    layer_types = getattr(_granite_mod, "ALL_DECODER_LAYER_TYPES", None)
    if not isinstance(layer_types, dict):
        return  # Module shape changed — nothing to patch.
    if "linear_attention" in layer_types or "full_attention" in layer_types:
        return  # Already handles the remapped names (e.g. PR #47634 merged).
    if "attention" not in layer_types or "mamba" not in layer_types:
        return  # Legacy keys absent — not the shape this patch targets.

    layer_types["full_attention"] = layer_types["attention"]
    layer_types["linear_attention"] = layer_types["mamba"]


def _hpu_mamba_bind_kv_cache(self, kv_cache) -> None:
    """HPU-safe replacement for ``MambaBase.bind_kv_cache``.

    Upstream vLLM PR #44456 ("[3/N][KV-Cache Layout Refactor] Standardize
    Mamba cache") replaced the old direct assignment
    (``forward_context[layer_name].kv_cache = kv_cache`` in
    ``vllm.v1.worker.utils.bind_kv_cache``) with a per-layer
    ``MambaBase.bind_kv_cache`` that unpacks a single packed ``[B, 1, 1, C]``
    int8 page view via ``kv_cache.squeeze(dim=(1, 2))``.

    The HPU model runner never packs the Mamba state into one int8 page:
    ``HPUModelRunner.initialize_kv_cache`` allocates a separate tensor per
    state (conv/ssm) and always stores a ``tuple``/``list`` in the ``kv_caches``
    dict for every MambaSpec layer (see the standard, hybrid and
    naive-cache-sharing branches — each ends in ``kv_caches[...] = tuple(...)``
    or ``= state_tensors``). ``bind_kv_cache`` therefore only ever receives that
    pre-split sequence on HPU; the upstream ``.squeeze`` path would raise
    ``AttributeError: 'tuple' object has no attribute 'squeeze'`` at EngineCore
    init (crashes ``run_granite_4_h_load_generate_test`` for
    ibm-granite/granite-4.0-h-small).

    Restore the pre-#44456 contract by assigning the pre-split sequence
    directly. This patch is HPU-only (installed from the HPU plugin), so the
    single-int8-page allocation never reaches it; ``tuple(kv_cache)`` would
    silently iterate a raw tensor's leading dim, so guard the contract instead
    of coercing.

    Upstream ref: https://github.com/vllm-project/vllm/pull/44456
    """
    assert isinstance(
        kv_cache,
        (tuple,
         list)), (f"HPU MambaBase.bind_kv_cache expects a pre-split conv/ssm sequence, got {type(kv_cache).__name__}")
    self.kv_cache = tuple(kv_cache)


def _patch_mamba_bind_kv_cache() -> None:
    """Install the HPU-safe ``MambaBase.bind_kv_cache`` replacement.

    Guarded so this is a no-op unless the target actually exists and still
    uses the ``squeeze``-based unpacking introduced by PR #44456:

    * If ``vllm.model_executor.layers.mamba.abstract.MambaBase`` cannot be
      imported or has no ``bind_kv_cache`` attribute (import-path or API
      change), skip silently — nothing to patch.
    * If the upstream method no longer calls ``.squeeze(`` (upstream reverted
      to a direct assignment), skip — the AttributeError this patch works
      around can no longer occur, and overwriting would only risk masking a
      future upstream change.

    Deferred to ``load_general_plugins`` time so the ``vllm.model_executor``
    import chain runs after the platform is ready.

    Upstream ref: https://github.com/vllm-project/vllm/pull/44456
    """
    try:
        from vllm.model_executor.layers.mamba.abstract import MambaBase
    except ImportError:
        return  # Import path changed/removed upstream — nothing to patch.

    upstream = getattr(MambaBase, "bind_kv_cache", None)
    if upstream is None:
        return  # Method dropped upstream — nothing to patch.

    if upstream is _hpu_mamba_bind_kv_cache:
        return  # Already installed (idempotent within this process).

    # Only patch when the upstream body still does the squeeze-based unpacking
    # that trips on HPU's pre-split tuple; otherwise leave upstream untouched.
    try:
        source = inspect.getsource(upstream)
    except (OSError, TypeError):
        source = ""  # Builtin/C or source unavailable — assume it needs the patch.
    if source and ".squeeze(" not in source:
        return  # Upstream no longer squeezes — the tuple crash cannot occur.

    MambaBase.bind_kv_cache = _hpu_mamba_bind_kv_cache


def _hpu_mla_bind_kv_cache(self, kv_cache) -> None:
    """Store the HPU per-layer KV-cache tuple unchanged.

    Upstream MLAAttention.bind_kv_cache executes `self.kv_cache = kv_cache.squeeze(1)`,
    assuming a single 4D CUDA tensor [B, 1, N, C]. On HPU, kv_cache is allocated as a
    tuple (key_cache, value_cache, key_scales, value_scales), which raises
    `AttributeError: 'tuple' object has no attribute 'squeeze'`.
    """
    if isinstance(kv_cache, (tuple, list)):
        self.kv_cache = tuple(kv_cache)
    else:
        self.kv_cache = kv_cache.squeeze(1)


def _patch_mla_bind_kv_cache() -> None:
    """Install the HPU-safe ``MLAAttention.bind_kv_cache`` replacement."""
    try:
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    except ImportError:
        return

    upstream = getattr(MLAAttention, "bind_kv_cache", None)
    if upstream is None or upstream is _hpu_mla_bind_kv_cache:
        return

    MLAAttention.bind_kv_cache = _hpu_mla_bind_kv_cache
    MLAAttention._patched_hpu_bind = True


def _hpu_get_num_layers_by_block_type(self, parallel_config, block_type="attention"):
    """HPU-safe replacement for ``ModelConfig.get_num_layers_by_block_type``.

    Identical to the upstream implementation, but normalizes both sides of the
    ``layers_block_type`` comparison so either naming convention matches. Under
    transformers>=5 ``layers_block_type`` aliases the remapped ``layer_types``
    ("attention" -> "full_attention", "mamba" -> "linear_attention"), while the
    HPU model runner queries with the legacy names ("attention", "mamba"). Left
    unpatched, the counts collapse to zero and the hybrid KV-cache setup breaks.

    Upstream ref: https://github.com/vllm-project/vllm/pull/47634
    """
    attn_block_type = block_type == "attention"
    is_transformer = not self.is_hybrid and not self.has_noops and not self.is_attention_free
    start, end = self.get_layers_start_end_indices(parallel_config)

    if is_transformer:
        return end - start if attn_block_type else 0
    elif self.is_attention_free:
        return 0 if attn_block_type else end - start
    elif self.has_noops:
        block_configs = self.hf_config.block_configs
        return sum(not bc.attention.no_op for bc in block_configs[start:end])
    else:
        # Hybrid model Jamba
        layers_block_type_value = getattr(self.hf_text_config, "layers_block_type", None)
        if layers_block_type_value is not None:
            if self.model_arch_config.text_model_type == "zamba2":
                if attn_block_type:
                    return sum(t == "hybrid" for t in layers_block_type_value[start:end])
                else:
                    return self.get_num_layers(parallel_config)
            aliases = _GRANITE_LEGACY_LAYER_ALIASES
            normalized_block_type = aliases.get(block_type, block_type)
            return sum(aliases.get(t, t) == normalized_block_type for t in layers_block_type_value[start:end])

        # Hybrid model Minimax
        attn_type_list = getattr(self.hf_config, "attn_type_list", None)
        if attn_type_list:
            return sum(t == 1 for t in attn_type_list[start:end])

        # Hybrid model Qwen3Next Qwen3.5 Series
        layer_types_value = getattr(self.hf_text_config, "layer_types", None)
        if layer_types_value is not None:
            if block_type == "attention":
                return sum(t == "full_attention" for t in layer_types_value[start:end])
            elif block_type == "linear_attention":
                return sum(t == "linear_attention" for t in layer_types_value[start:end])
            else:
                return sum(t == block_type for t in layer_types_value[start:end])

        # Hybrid model Kimi-K3 / Kimi Linear
        linear_attn_config = getattr(self.hf_text_config, "linear_attn_config", None) or getattr(
            self.hf_config, "linear_attn_config", None)
        if linear_attn_config is not None:
            if isinstance(linear_attn_config, dict):
                full_attn = set(linear_attn_config.get("full_attn_layers", []))
                kda = set(linear_attn_config.get("kda_layers", []))
            else:
                full_attn = set(getattr(linear_attn_config, "full_attn_layers", []))
                kda = set(getattr(linear_attn_config, "kda_layers", []))
            if block_type == "attention":
                return sum((idx + 1) in full_attn for idx in range(start, end))
            elif block_type == "linear_attention":
                return sum((idx + 1) in kda for idx in range(start, end))
            else:
                return 0

        if layers_block_type_value is None and attn_type_list is None and layer_types_value is None and linear_attn_config is None:
            raise ValueError("The model is an hybrid without a layers_block_type or an "
                             "attn_type_list, or a layer_types in the hf_config, "
                             f"cannot determine the num of {block_type} layers")
        raise AssertionError(f"Unsupported block type: {block_type}")


def _patch_get_num_layers_by_block_type() -> None:
    """Install the HPU-safe ``get_num_layers_by_block_type`` replacement.

    Guarded by ``inspect.getsource`` so this is a no-op once upstream vLLM
    normalizes the ``layers_block_type`` comparison itself (PR #47634).
    """
    import inspect

    from vllm.config import ModelConfig

    try:
        source = inspect.getsource(ModelConfig.get_num_layers_by_block_type)
    except (OSError, TypeError):
        return  # Source unavailable — leave upstream in place.

    if "get_num_layers_by_block_type" not in source or "layers_block_type" not in source:
        return  # Method shape changed — do not risk an incompatible override.
    if "linear_attn_config" in source and ("_GRANITE_LEGACY_LAYER_ALIASES" in source
                                           or "normalized_block_type" in source):
        return  # Already normalizes and supports linear_attn_config.

    ModelConfig.get_num_layers_by_block_type = _hpu_get_num_layers_by_block_type


def _patch_use_sequence_parallel_moe() -> None:
    """Restore the ``data_parallel_size > 1`` guard on ``use_sequence_parallel_moe``.

    vllm PR #48036 removed ``and self.data_parallel_size > 1`` from
    ``ParallelConfig.use_sequence_parallel_moe`` to enable SP-MoE for DSv3.2 +
    MTP on a single node. On HPU that flips SP-MoE on for plain EP + TP>1 +
    DP==1 setups (e.g. Kimi-K2.6 / DeepSeek MLA), whose reduce_scatter /
    sequence_parallel_chunk reshaping the HPU MLA rotary and RMSNorm ops cannot
    yet handle, crashing the forward pass. Until those ops support the SP-MoE
    layout, re-add the DP>1 guard so single-node EP behaves as before.

    Guarded by ``inspect.getsource`` so this becomes a no-op if upstream
    restores the guard or the property's shape changes.
    """
    import inspect

    from vllm.config.parallel import ParallelConfig

    prop = ParallelConfig.__dict__.get("use_sequence_parallel_moe")
    if not isinstance(prop, property) or prop.fget is None:
        return  # Not a property anymore — do not risk an incompatible override.

    try:
        source = inspect.getsource(prop.fget)
    except (OSError, TypeError):
        return  # Source unavailable — leave upstream in place.

    if "use_sequence_parallel_moe" not in source or "enable_expert_parallel" not in source:
        return  # Shape changed — do not risk an incompatible override.
    if "data_parallel_size" in source:
        return  # Guard already present (upstream restored it) — no-op.

    _original_fget = prop.fget

    def _hpu_use_sequence_parallel_moe(self) -> bool:
        return _original_fget(self) and self.data_parallel_size > 1

    ParallelConfig.use_sequence_parallel_moe = property(_hpu_use_sequence_parallel_moe)


def _patch_cleanup_dist_env_and_memory() -> None:
    """Install the HPU-safe ``cleanup_dist_env_and_memory`` replacement.

    Deferred to ``load_general_plugins`` time (rather than ``apply()`` at
    platform-registration time) so the ``vllm.distributed.parallel_state``
    import chain runs *after* the platform is initialised and *after*
    ``vllm.utils.torch_utils`` has finished importing (see ``apply()`` and
    the module-level NOTE — GAUDISW-249622).
    """
    from vllm.distributed import parallel_state
    import vllm.distributed as _vllm_distributed

    parallel_state.cleanup_dist_env_and_memory = _hpu_cleanup_dist_env_and_memory
    _vllm_distributed.cleanup_dist_env_and_memory = _hpu_cleanup_dist_env_and_memory


def _hpu_check_shm_free_space(*args, **kwargs) -> None:
    """No-op replacement for ``shm_broadcast.check_shm_free_space``.

    Upstream vLLM PR #48879 added an eager guard that rejects creating the shm
    ring buffer when its *nominal* size (default 240 MiB per ``MessageQueue``)
    exceeds the free space on ``/dev/shm``. HPU CI containers expose the Docker
    default 64 MiB ``/dev/shm``, on which the pre-#48879 path worked: the
    segment is backed by tmpfs and only the pages actually written are
    allocated, and the broadcast workers touch a small fraction of the ring.
    The guard aborts EngineCore start-up before any page is touched, so restore
    the prior behaviour by skipping the check on HPU.

    Accepts ``*args, **kwargs`` rather than mirroring the upstream signature so
    that a future signature change (an added positional/keyword parameter) does
    not raise ``TypeError`` at the (single, bare-name) call site in
    ``ShmRingBuffer.__init__``. All arguments are ignored.

    Upstream ref: https://github.com/vllm-project/vllm/pull/48879
    """
    return None


def _patch_check_shm_free_space() -> None:
    """Neutralize the eager ``/dev/shm`` size guard added by vLLM PR #48879.

    ``ShmRingBuffer.__init__`` calls ``check_shm_free_space`` by bare name, so
    replacing the module-level attribute intercepts every ring-buffer creation.

    We always override the symbol when it exists, rather than inspecting its
    source to confirm it still raises: the no-op is harmless even if upstream
    later softens the guard to size-to-fit (nothing would have raised anyway),
    whereas a source-string heuristic is brittle — upstream could keep failing
    fast while raising a different exception type, wrapping the logic in a
    helper, or rewording the message, any of which would silently stop the
    patch from applying and reintroduce the ``Insufficient space in /dev/shm``
    startup crash on HPU. The attribute-presence check below still self-retires
    the patch if upstream removes or renames the function (e.g. lands a proper
    fix).

    Deferred to ``load_general_plugins`` time so the
    ``vllm.distributed.device_communicators`` import chain runs after platform
    initialisation.

    Upstream ref: https://github.com/vllm-project/vllm/pull/48879
    """
    try:
        import vllm.distributed.device_communicators.shm_broadcast as _shm_mod
    except ImportError:
        return  # Import path changed/removed upstream — nothing to patch.

    upstream = getattr(_shm_mod, "check_shm_free_space", None)
    if upstream is None:
        return  # Guard absent (pre-#48879 or renamed) — nothing to patch.

    if upstream is _hpu_check_shm_free_space:
        return  # Already installed (idempotent within this process).

    _shm_mod.check_shm_free_space = _hpu_check_shm_free_space


def _hpu_free_blocks(self, ordered_blocks) -> None:
    """Single-pass ``BlockPool.free_blocks`` for ``enable_caching=False``.

    Upstream vLLM PR #42656 rewrote ``free_blocks`` to partition freed blocks
    into with-hash / without-hash lists and issue two queue ops
    (``prepend_n`` + ``append_n``) on every engine step.  When prefix caching
    is disabled, every block's hash is ``None``, so that split is a no-op that
    only adds per-step CPU work; on short decode steps (small model / low
    batch) it is a measurable fixed overhead (~3.5% output-token throughput on
    llama-3.1-8B FP8, 1 card, 4096/1024, mc=8 — see GAUDISW-250180).  Free in
    a single pass in that case; delegate to the original (upstream)
    implementation when prefix caching is enabled so #42656's LRU ordering is
    preserved.  Remove this patch once the fix lands upstream.
    """
    if self.enable_caching:
        assert _ORIGINAL_FREE_BLOCKS is not None  # set by _patch_free_blocks before install
        return _ORIGINAL_FREE_BLOCKS(self, ordered_blocks)

    freed_blocks = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            freed_blocks.append(block)
    self.free_block_queue.append_n(freed_blocks)


_ORIGINAL_FREE_BLOCKS: Optional[Callable] = None


def _patch_free_blocks() -> None:
    """Install the single-pass ``free_blocks`` fast path for APC-disabled runs.

    Deferred to ``load_general_plugins`` time (same as the other patches) so
    the ``vllm.v1.core`` import runs after platform initialisation.
    Idempotent: only wraps the original ``free_blocks`` once.
    """
    global _ORIGINAL_FREE_BLOCKS
    from vllm.v1.core.block_pool import BlockPool

    if _ORIGINAL_FREE_BLOCKS is not None:
        return  # already patched
    _ORIGINAL_FREE_BLOCKS = BlockPool.free_blocks
    BlockPool.free_blocks = _hpu_free_blocks


# Global cache for FusedSDPA operator to avoid recreation overhead
_CACHED_FSDPA_OP = None


def _ensure_fsdpa_cached() -> None:
    """Eagerly initialize the FusedSDPA operator cache.

    Must be called BEFORE torch.compile traces _hpu_sdpa_attention_forward,
    otherwise dynamo creates a guard on `_CACHED_FSDPA_OP is None` which
    fails at runtime and triggers recompilation.

    Called from hpu_model_runner.py during model initialization.
    """
    global _CACHED_FSDPA_OP
    if _CACHED_FSDPA_OP is not None:
        return
    try:
        from vllm_gaudi.extension.utils import ModuleFusedSDPA
        import vllm_gaudi.extension.kernels as kernels
        HPUFusedSDPA = kernels.fsdpa()
        _CACHED_FSDPA_OP = ModuleFusedSDPA(HPUFusedSDPA)
    except Exception:
        pass


def _hpu_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """HPU-optimized replacement for transformers' ``sdpa_attention_forward``.

    Based on ``transformers.integrations.sdpa_attention.sdpa_attention_forward``
    from transformers v5.14.1. Uses HPU FusedSDPA kernel when running on HPU
    with fsdpa_impl config, providing better performance for models that use
    transformers SDPA attention (e.g., vision encoders in multimodal models).

    The FusedSDPA operator is cached globally to avoid recreation overhead
    on each call, which would otherwise cause significant slowdown.
    """
    from transformers.integrations.sdpa_attention import (repeat_kv, use_gqa_in_sdpa, create_position_bias_mask,
                                                          _is_torch_npu_available, logger)

    if kwargs.get("output_attentions", False):
        logger.warning_once("`sdpa` attention does not support `output_attentions=True`."
                            " Please set your attention to `eager` if you want any of these features.")
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        if not use_gqa_in_sdpa(attention_mask, key, value):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    q_length = query.shape[2]
    kv_length = key.shape[2]

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = q_length > 1 and attention_mask is None and is_causal

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    if _is_torch_npu_available and attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    if is_causal and attention_mask is None and q_length > 1 and kv_length > q_length:
        key = key[:, :, :q_length, :]
        value = value[:, :, :q_length, :]
        if position_bias is not None:
            position_bias = position_bias[:, :, :, :q_length]

    if position_bias is not None:
        attention_mask = create_position_bias_mask(position_bias, attention_mask, is_causal, query, key)
        is_causal = False

    # Check if we should use HPU FusedSDPA
    _use_hpu_fsdpa = False
    _config = None
    if query.device.type == "hpu":
        try:
            from vllm_gaudi.extension.runtime import get_config
            _config = get_config()
            if _config.prompt_attn_impl == "fsdpa_impl":
                _use_hpu_fsdpa = True
        except Exception:
            pass

    if _use_hpu_fsdpa and _config is not None:
        # _CACHED_FSDPA_OP is initialized eagerly by _ensure_fsdpa_cached()
        # before torch.compile traces this function. This avoids the dynamo
        # guard on `_CACHED_FSDPA_OP is None` that would trigger recompilation.
        _ensure_fsdpa_cached()
        if _CACHED_FSDPA_OP is None:
            raise RuntimeError("FSDPA op failed to initialize")

        softmax_mode = "fp32" if _config.fp32_softmax else "fast"
        attn_output = _CACHED_FSDPA_OP(
            query,
            key,
            value,
            attention_mask,
            dropout_p=dropout,
            is_causal=is_causal,
            scale=scaling,
            softmax_mode=softmax_mode,
            recompute_mode=True,
            valid_sequence_lengths=None,
        )
    else:
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
            **sdpa_kwargs,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _patch_sdpa_attention_forward() -> None:
    """Replace transformers' ``sdpa_attention_forward`` with HPU-optimized version.

    This patch enables the HPU FusedSDPA kernel for models that use transformers
    SDPA attention, providing 1.4-2x performance improvement.

    The patch is only applied when running on HPU and when the fsdpa_impl
    attention implementation is configured.
    """
    try:
        import transformers.integrations.sdpa_attention as _sdpa_mod
        _sdpa_mod.sdpa_attention_forward = _hpu_sdpa_attention_forward

        # Also update ALL_ATTENTION_FUNCTIONS which caches function references
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if "sdpa" in ALL_ATTENTION_FUNCTIONS:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = _hpu_sdpa_attention_forward
    except ImportError:
        pass  # transformers version without this module

    # Eagerly initialize the operator cache at patch-registration time,
    # before torch.compile traces the function. Same pattern as hpu_attn.py
    # which calls kernels.fsdpa() in __init__.
    _ensure_fsdpa_cached()


_KIMI_PATCHES_INSTALLED: list[str] = []
_KIMI_PATCHES_FAILED: dict[str, str] = {}


def _kimi_patch_logger():
    """Logger under the 'vllm.' namespace (loggers outside it are silent), created lazily."""
    import logging
    return logging.getLogger("vllm.gaudi.kimi_patches")


def _kimi_patch_failed(name: str) -> None:
    """Call from an ``except`` block: remember the failed Kimi-K3 patch.

    The patches are installed in several attempts (plugin apply, then ``load_general_plugins``) and early attempts
    legitimately hit partially initialised vllm modules, so a failure is only logged at DEBUG (with traceback).
    ``_kimi_patches_log_failed`` warns about the patches that are still missing after the final attempt.
    """
    import sys
    exc = sys.exc_info()[1]
    _KIMI_PATCHES_FAILED[name] = f"{type(exc).__name__}: {exc}" if exc is not None else "unknown error"
    _kimi_patch_logger().debug("Kimi-K3 patch '%s' failed in this attempt", name, exc_info=True)


def _kimi_patch_ok(name: str) -> None:
    """Clear an earlier failure of a patch that is not listed in the 'installed' summary."""
    _KIMI_PATCHES_FAILED.pop(name, None)


def _kimi_patch_installed(name: str) -> None:
    _kimi_patch_ok(name)
    if name not in _KIMI_PATCHES_INSTALLED:
        _KIMI_PATCHES_INSTALLED.append(name)


def _kimi_patch_not_applicable(name: str) -> None:
    """The patch targets something this vllm build does not have: neither installed nor missing."""
    _KIMI_PATCHES_FAILED.pop(name, None)
    _kimi_patch_logger().debug("Kimi-K3 patch '%s' is not applicable here", name, exc_info=True)


def _kimi_patches_log_installed() -> None:
    _kimi_patch_logger().info("Kimi-K3 patches installed: %s", ", ".join(_KIMI_PATCHES_INSTALLED))


def _kimi_patches_log_failed(only=None) -> None:
    """One WARNING (no traceback) per patch that is still missing and was not reported yet.

    ``only`` restricts the report to the patches that matter to the calling process.
    """
    for name, reason in list(_KIMI_PATCHES_FAILED.items()):
        if name in _KIMI_PATCHES_WARNED or (only is not None and name not in only):
            continue
        _KIMI_PATCHES_WARNED.add(name)
        _kimi_patch_logger().warning("Kimi-K3 patch '%s' was not installed (%s)", name, reason)


_KIMI_PATCHES_REPORTED = False
_KIMI_CONFIG_REPORTED = False
_KIMI_PATCHES_WARNED: set[str] = set()
# vllm.models.kimi_k3 holds all K3 classes (KimiK3ForConditionalGeneration, KimiLinearForCausalLM, MTP, DSpark);
# Kimi-VL, Kimi-K2.5 and Kimi-Audio live in vllm.model_executor.models.kimi_* and are not K3.
_KIMI_K3_MODULE = "vllm.models.kimi_k3"
_KIMI_K3_ARCHITECTURES = ("KimiK3ForConditionalGeneration", "KimiK3MTPModel", "KimiLinearForCausalLM", "K3DSparkModel")
# Patches that act in the process which builds the config (APIServer, EngineCore), not only in model workers.
_KIMI_CONFIG_PATCHES = ("hybrid_config", "k3_arch")


def _is_kimi_k3_model(model) -> bool:
    return any(c.__module__.startswith(_KIMI_K3_MODULE) for c in type(model).__mro__)


def _is_kimi_k3_config(model_config) -> bool:
    archs = getattr(model_config, "architectures", None) or getattr(getattr(model_config, "hf_config", None),
                                                                    "architectures", None) or ()
    return any(a in _KIMI_K3_ARCHITECTURES for a in archs)


def _current_config_is_kimi_k3() -> bool:
    """True when the vllm config being built/used right now is for a Kimi-K3 model (False when there is none)."""
    try:
        from vllm.config import get_current_vllm_config_or_none
        vllm_config = get_current_vllm_config_or_none()
    except ImportError:
        return False
    return vllm_config is not None and _is_kimi_k3_config(getattr(vllm_config, "model_config", None))


def _optional_runner_absent(exc: BaseException, mod_name: str, cls_name: str) -> bool:
    """True when the ROCm-only latent MoE runner does not exist in this vllm build (not a failure on HPU)."""
    if cls_name != "ROCmLatentMoERunner":
        return False
    return (isinstance(exc, ModuleNotFoundError) and exc.name == mod_name) or isinstance(exc, AttributeError)


def report_kimi_patches(model) -> None:
    """Log the Kimi-K3 patch status once per process, after the model has been built.

    The patches are installed in several attempts, in every process (APIServer, EngineCore, workers), and an
    attempt cannot tell whether it is the last one, so nothing above DEBUG is logged while installing. The model
    runner calls this right after ``get_model``: only a process that really built a Kimi-K3 model reports, once,
    with the INFO summary and one WARNING per patch that is still missing.
    """
    global _KIMI_PATCHES_REPORTED
    if _KIMI_PATCHES_REPORTED or not _is_kimi_k3_model(model):
        return
    _KIMI_PATCHES_REPORTED = True
    _kimi_patches_log_installed()
    _kimi_patches_log_failed()


def report_kimi_patches_on_load_failure(model_config) -> None:
    """``get_model`` raised: no model exists, so say which patches are missing (the likely cause) before re-raising."""
    global _KIMI_PATCHES_REPORTED
    if _KIMI_PATCHES_REPORTED or not _is_kimi_k3_config(model_config):
        return
    _KIMI_PATCHES_REPORTED = True
    _kimi_patches_log_failed()


def _report_kimi_config_patches(model_config) -> None:
    """Warn once per process about missing config-side patches, right after a Kimi-K3 ModelConfig was built.

    APIServer and EngineCore never build the model, but they build the config, which needs ``k3_arch`` (architecture
    resolution) and ``hybrid_config`` (this hook). The patched ``ModelConfig.__post_init__`` is a definite late
    point: the patches are installed by then, so a name still in the failed list is really missing. Patches that
    only matter to model workers (kda_forward, attn_res_*, ...) are left to ``report_kimi_patches``.
    """
    global _KIMI_CONFIG_REPORTED
    if _KIMI_CONFIG_REPORTED or not _is_kimi_k3_config(model_config):
        return
    _KIMI_CONFIG_REPORTED = True
    _kimi_patches_log_failed(only=_KIMI_CONFIG_PATCHES)


def _patch_kimi_kda() -> None:
    """Pure PyTorch fallback for KDA on HPU (disabling GPU Triton kernels and dict metadata assertion)."""
    try:
        import vllm.models.kimi_k3.amd.kda as _amd_kda
        import vllm.models.kimi_k3.amd.ops.kda_chunk as _kda_chunk
        import vllm.models.kimi_k3.amd.ops.kda_decode as _kda_decode

        # The fused KDA checks query vllm.platforms.rocm (amdsmi), which fails on HPU. They are the only users of
        # that module in kimi_k3/amd, so answer False where they are defined and where amd.kda imported them by name.
        _kda_chunk.is_fused_kda_chunk_supported = lambda *args, **kwargs: False
        _kda_decode.is_fused_kda_decode_supported = lambda *args, **kwargs: False
        _amd_kda.is_fused_kda_chunk_supported = lambda *args, **kwargs: False
        _amd_kda.is_fused_kda_decode_supported = lambda *args, **kwargs: False

        def _kda_forward(self, hidden_states, positions=None):
            # Imported on first use: importing vllm_gaudi.models here (plugin-apply time) hits a circular
            # import with vllm.distributed in processes where it is still being initialised.
            from vllm_gaudi.models.kimi_k3_kda import hpu_kda_forward
            return hpu_kda_forward(self, hidden_states, positions)

        if hasattr(_amd_kda, "KimiK3DeltaAttention"):
            _amd_kda.KimiK3DeltaAttention.forward = _kda_forward
            _amd_kda.KimiK3DeltaAttention._forward = lambda *args, **kwargs: None
        if hasattr(_amd_kda, "GatedDeltaNetAttention"):
            _amd_kda.GatedDeltaNetAttention.forward = _kda_forward
            _amd_kda.GatedDeltaNetAttention._forward = lambda *args, **kwargs: None
        _kimi_patch_installed("kda_forward")

        try:
            import vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn as _gdn_mod
            _gdn_mod.KimiGatedDeltaNetAttention.forward = _kda_forward
            _gdn_mod.KimiGatedDeltaNetAttention._forward = lambda *args, **kwargs: None
            _kimi_patch_installed("kimi_gdn_forward")
        except Exception:
            _kimi_patch_failed("kimi_gdn_forward")
    except Exception:
        _kimi_patch_failed("kda_forward")


def _patch_mla_prefill_selector() -> None:
    """Return _DummyPrefillBackend for MLA prefill on HPU for Kimi-K3 configs; others keep the original selector."""
    try:
        import vllm.v1.attention.backends.mla.prefill.selector as sel_mod
        from vllm_gaudi.attention.oot_mla import _DummyPrefillBackend
        orig_fn = getattr(sel_mod, "get_mla_prefill_backend", None)

        def hpu_get_prefill(vllm_config):
            from vllm.platforms import current_platform
            if orig_fn and not _is_kimi_k3_config(getattr(vllm_config, "model_config", None)):
                return orig_fn(vllm_config)
            if getattr(current_platform, "device_type", None) == "hpu" or getattr(current_platform, "device_name",
                                                                                  None) == "hpu":
                return _DummyPrefillBackend
            if orig_fn:
                return orig_fn(vllm_config)
            return _DummyPrefillBackend

        sel_mod.get_mla_prefill_backend = hpu_get_prefill
        try:
            import vllm.model_executor.layers.attention.mla_attention as mla_mod
            mla_mod.get_mla_prefill_backend = hpu_get_prefill
            _kimi_patch_ok("mla_prefill_selector_attention_module")
        except Exception:
            _kimi_patch_failed("mla_prefill_selector_attention_module")
        _kimi_patch_installed("mla_prefill_selector")
    except Exception:
        _kimi_patch_failed("mla_prefill_selector")


def _patch_kimi_k3_hybrid_config() -> None:
    """Populate layer_types from linear_attn_config for hybrid models like Kimi-K3 and disable expert intermediate padding on EP."""
    try:
        from vllm.config import ModelConfig
        orig_post_init = getattr(ModelConfig, "__post_init__", None)
        if orig_post_init is not None and not getattr(orig_post_init, "_hpu_kimi_post_init", False):

            def _hpu_post_init(self, *args, **kwargs):
                orig_post_init(self, *args, **kwargs)
                _report_kimi_config_patches(self)
                # Ensure KimiMoE does not pad moe_intermediate_size from 3072 to 4096 under EP (saves ~30 GB memory per card)
                if _is_kimi_k3_config(self):
                    for cfg_obj in (getattr(self, "hf_text_config", None), getattr(self, "hf_config", None)):
                        if cfg_obj is not None:
                            setattr(cfg_obj, "min_moe_intermediate_per_partition", 1)
                lac = getattr(self.hf_text_config, "linear_attn_config", None) or getattr(
                    self.hf_config, "linear_attn_config", None)
                if lac is not None and not getattr(self.hf_text_config, "layer_types", None):
                    full_set = set(
                        lac.get("full_attn_layers", []) if isinstance(lac, dict
                                                                      ) else getattr(lac, "full_attn_layers", []))
                    num_layers = getattr(self.hf_text_config, "num_hidden_layers", 93)
                    l_types = [
                        "full_attention" if (i + 1) in full_set else "linear_attention" for i in range(num_layers)
                    ]
                    self.hf_text_config.layer_types = l_types
                    if hasattr(self.hf_config, "text_config"):
                        self.hf_config.text_config.layer_types = l_types

            _hpu_post_init._hpu_kimi_post_init = True
            ModelConfig.__post_init__ = _hpu_post_init
        _kimi_patch_ok("hybrid_config")
    except Exception:
        _kimi_patch_failed("hybrid_config")


def _patch_moe_runner_delegation() -> None:
    """Delegate missing attribute lookups on MoERunner to routed_experts, for Kimi-K3 runners only.

    The patch is installed on the classes for every model (load_general_plugins runs before any ModelConfig exists),
    so the decision is made per runner: ``hasattr(runner, name)`` must stay False for the attributes only
    ``routed_experts`` has, or any other model's ``register_parameter`` on the runner raises
    ``KeyError: attribute '...' already exists``. The flag is set when the runner is built, where the current vllm
    config is that of the model being built; runner classes defined in vllm.models.kimi_k3 always delegate.
    """
    for mod_name, cls_name in [
        ("vllm.model_executor.layers.fused_moe.runner.moe_runner", "MoERunner"),
        ("vllm.models.kimi_k3.amd.latent_moe_runner", "ROCmLatentMoERunner"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            if getattr(cls, "_patched_delegation", False):
                # Already patched, on this class or (ROCmLatentMoERunner subclasses MoERunner) on a base class that
                # the lookup reaches by inheritance: covered. Record it so that it clears an earlier failed attempt.
                _kimi_patch_installed(f"moe_runner_delegation[{cls_name}]")
                if "_patched_delegation" not in vars(cls):
                    _kimi_patch_logger().debug("Kimi-K3 patch 'moe_runner_delegation[%s]' is covered by a base class",
                                               cls_name)
                continue
            orig_getattr = getattr(cls, "__getattr__", None)
            orig_init = cls.__init__

            def _make_init(og_init):

                @functools.wraps(og_init)
                def _init(self, *args, **kwargs):
                    og_init(self, *args, **kwargs)
                    self.__dict__["_kimi_k3_delegation"] = _current_config_is_kimi_k3() or any(
                        c.__module__.startswith(_KIMI_K3_MODULE) for c in type(self).__mro__)

                return _init

            def _make_getattr(og_get, target_cls):

                def _getattr(self, name: str):
                    try:
                        if og_get:
                            return og_get(self, name)
                        return super(target_cls, self).__getattr__(name)
                    except AttributeError:
                        # __dict__ lookup: no recursion into __getattr__ while the runner is still being built
                        if not self.__dict__.get("_kimi_k3_delegation", False):
                            raise
                        routed = getattr(self, "routed_experts", None)
                        if routed is not None and hasattr(routed, name):
                            return getattr(routed, name)
                        raise

                return _getattr

            cls.__init__ = _make_init(orig_init)
            cls.__getattr__ = _make_getattr(orig_getattr, cls)
            cls._patched_delegation = True
            _kimi_patch_installed(f"moe_runner_delegation[{cls_name}]")
        except Exception as exc:
            name = f"moe_runner_delegation[{cls_name}]"
            # The ROCm runner is optional: a vllm build without it is not a failure.
            if _optional_runner_absent(exc, mod_name, cls_name):
                _kimi_patch_not_applicable(name)
            else:
                _kimi_patch_failed(name)


def _patch_latent_moe_runner_forward() -> None:
    """Run the Kimi-K3 latent-MoE runner through the inline HPU MoE path (no opaque custom op).

    ``ROCmLatentMoERunner._fused_forward`` calls ``torch.ops.vllm.moe_forward_shared`` which the HPU compile
    backend executes eagerly ("Fallback required"), breaking the graph at every MoE layer. Replace it with the
    dp_size==1 inline implementation used for the other MoE models. Disable with HPU_FUSED_MOE=0.
    """
    from vllm_gaudi.extension.logger import logger as init_logger

    logger = init_logger()
    try:
        from vllm.models.kimi_k3.amd.latent_moe_runner import ROCmLatentMoERunner
        from vllm_gaudi.ops.hpu_fused_moe import _MOE_COMPILE, patched_latent_moe_fused_forward
    except Exception as exc:
        logger.warning_once("Kimi-K3 latent MoE runner patch skipped, import failed: %r", exc)
        return
    if not _MOE_COMPILE:
        logger.warning_once("Kimi-K3 latent MoE runner patch skipped because HPU_FUSED_MOE=0")
        return
    if getattr(ROCmLatentMoERunner, "_patched_hpu_fused_forward", False):
        return

    upstream_fused_forward = ROCmLatentMoERunner._fused_forward

    def _hpu_fused_forward(self, hidden_states, router_logits, input_ids, shared_experts_input):
        if self.moe_config.dp_size != 1 or getattr(self.moe_config, "pcp_size", 1) > 1:
            return upstream_fused_forward(self, hidden_states, router_logits, input_ids, shared_experts_input)
        return patched_latent_moe_fused_forward(self, hidden_states, router_logits, input_ids, shared_experts_input)

    ROCmLatentMoERunner._fused_forward = _hpu_fused_forward
    ROCmLatentMoERunner._patched_hpu_fused_forward = True
    logger.debug("Patched %s._fused_forward with the inline HPU MoE path, pid %d", ROCmLatentMoERunner.__name__,
                 os.getpid())


def _configure_situ_aliases() -> str:
    """Validate HPU_KIMI_SITU_IMPL and return the effective implementation (debug note for "silu").

    SiTU (Kimi-K3 routed experts): the fused Habana op only knows silu/gelu. "exact" (default) computes the
    experts unfused (ops/hpu_situ_moe.py); "silu" maps "situ" to silu in the activation normalizers
    (situ_fused_alias, evaluated at use time: no module is imported here because the plugin load is circular). An
    invalid value is logged as an error and treated as "exact"; this runs outside of the broad try/except of
    the other Kimi patches so that it can neither be skipped by nor skip them.
    """
    import vllm_gaudi.envs as gaudi_envs
    from vllm_gaudi.extension.logger import logger as _ext_logger
    from vllm_gaudi.ops.hpu_situ_moe import situ_impl_or_exact

    situ_impl = situ_impl_or_exact(gaudi_envs.HPU_KIMI_SITU_IMPL, _ext_logger().error)
    if situ_impl == "silu":
        _ext_logger().debug("HPU_KIMI_SITU_IMPL=silu: the Kimi-K3 routed experts use silu instead of SiTU "
                            "(beta*tanh(gate/beta)*sigmoid(gate)*up); model outputs are APPROXIMATE")
    return situ_impl


def _patch_kimi_linear_weights() -> None:
    """Ensure KimiMoE safely accesses w13_weight and w2_weight without evaluating bool(Tensor)."""
    _configure_situ_aliases()
    try:
        import vllm.models.kimi_k3.amd.linear as k3_linear

        orig_moe_init = getattr(k3_linear.KimiMoE, "__init__", None)
        if orig_moe_init is not None and not getattr(k3_linear.KimiMoE, "_patched_weights", False):

            def _safe_moe_init(self, *args, **kwargs):
                try:
                    orig_moe_init(self, *args, **kwargs)
                except AttributeError as e:
                    if "w13_weight_packed" in str(e) or "w2_weight_packed" in str(e):
                        target = getattr(self.experts, "routed_experts", self.experts)
                        w13_candidates = (
                            getattr(target, "w13_weight", None),
                            getattr(target, "w13_weight_packed", None),
                            getattr(self.experts, "w13_weight", None),
                            getattr(self.experts, "w13_weight_packed", None),
                        )
                        w13_w = next((w for w in w13_candidates if w is not None), None)
                        w2_candidates = (
                            getattr(target, "w2_weight", None),
                            getattr(target, "w2_weight_packed", None),
                            getattr(self.experts, "w2_weight", None),
                            getattr(self.experts, "w2_weight_packed", None),
                        )
                        w2_w = next((w for w in w2_candidates if w is not None), None)
                        if w13_w is not None and hasattr(w13_w, "data"):
                            w13_w.data.zero_()
                        if w2_w is not None and hasattr(w2_w, "data"):
                            w2_w.data.zero_()
                        moe_cfg = getattr(self.experts, "moe_config", None) or getattr(target, "moe_config", None)
                        if moe_cfg is not None and hasattr(self, "padded_moe_intermediate_size"):
                            moe_intermediate_size = getattr(moe_cfg, "intermediate_size_per_partition", None)
                            if moe_intermediate_size:
                                moe_cfg.intermediate_size_per_partition_unpadded = (moe_intermediate_size //
                                                                                    self.tp_size)
                    else:
                        raise

            k3_linear.KimiMoE.__init__ = _safe_moe_init
            k3_linear.KimiMoE._patched_weights = True

        _kimi_patch_installed("linear_weights")
    except Exception:
        _kimi_patch_failed("linear_weights")


def _patch_kimi_attn_res() -> None:
    """Pure PyTorch fallback for Kimi-K3 attn_res when Triton is disabled on HPU."""
    try:
        from vllm_gaudi.ops.hpu_kda_pytorch import attn_res_torch as _torch_attn_res

        try:
            import vllm.models.kimi_k3.amd.ops.attn_res as _attn_res_mod
            _attn_res_mod.attn_res = _torch_attn_res
            _kimi_patch_installed("attn_res_op")
        except Exception:
            _kimi_patch_failed("attn_res_op")
        try:
            import vllm.models.kimi_k3.amd.linear as _linear_mod
            _linear_mod.attn_res = _torch_attn_res

            def _apply_attn_res_torch(prefix_sum,
                                      block_residual,
                                      proj,
                                      norm,
                                      num_valid_blocks,
                                      delta=None,
                                      output_norm=None,
                                      block_write_idx=-1,
                                      quant_key=None):
                # vLLM 28a5fba06+ passes quant_key to fuse an fp8 activation quant into attn_res. The HPU decoder
                # layer never asks for it (get_input_quant_key() is None), so a value here is a wiring bug.
                if quant_key is not None:
                    raise NotImplementedError(f"Kimi-K3 attn_res on HPU does not support quant_key={quant_key}")
                return _torch_attn_res(
                    prefix_sum,
                    delta,
                    block_residual,
                    norm.weight,
                    proj.weight.squeeze(0),
                    None if output_norm is None else output_norm.weight,
                    num_valid_blocks,
                    block_write_idx,
                    norm.variance_epsilon,
                    0.0 if output_norm is None else output_norm.variance_epsilon,
                )

            _linear_mod._apply_attn_res = _apply_attn_res_torch
            _kimi_patch_installed("attn_res_linear")
        except Exception:
            _kimi_patch_failed("attn_res_linear")
        _kimi_patch_ok("attn_res")
    except Exception:
        _kimi_patch_failed("attn_res")


def _patch_kimi_k3_arch() -> None:
    """Route Kimi-K3 models to the platform-independent AMD implementation on HPU."""
    try:
        _patch_kimi_kda()
        _patch_moe_runner_delegation()
        _patch_kimi_attn_res()
        _patch_kimi_linear_weights()
        _patch_mla_bind_kv_cache()
        import vllm.models.kimi_k3 as k3_pkg
        import vllm.models.kimi_k3.amd.linear as _amd_linear
        import vllm.models.kimi_k3.amd.mla as _amd_mla
        import vllm.models.kimi_k3.amd.model as _amd_model
        import vllm.models.kimi_k3.amd.mtp as _amd_mtp
        from vllm_gaudi.attention.oot_mla import HPUMultiHeadLatentAttentionWrapper

        _amd_linear.KimiK3MultiHeadLatentAttentionWrapper = HPUMultiHeadLatentAttentionWrapper
        _amd_mla.KimiK3MultiHeadLatentAttentionWrapper = HPUMultiHeadLatentAttentionWrapper

        k3_pkg.KimiLinearForCausalLM = _amd_linear.KimiLinearForCausalLM
        k3_pkg.KimiK3ForConditionalGeneration = _amd_model.KimiK3ForConditionalGeneration
        k3_pkg.KimiK3MTP = _amd_mtp.KimiK3MTP

        from vllm.model_executor.models.registry import ModelRegistry
        ModelRegistry.register_model("KimiLinearForCausalLM", _amd_linear.KimiLinearForCausalLM)
        ModelRegistry.register_model("KimiK3ForConditionalGeneration", _amd_model.KimiK3ForConditionalGeneration)
        ModelRegistry.register_model("KimiK3MTP", _amd_mtp.KimiK3MTP)
        _kimi_patch_installed("model_registration")

        # Monkey-patch KimiLinearModel.forward to ensure tokens/positions are 1D/2D
        # so block_residual is allocated with shape [num_tokens, attn_res_block_num, hidden_size].
        orig_linear_model_forward = getattr(_amd_linear.KimiLinearModel, "forward", None)
        if orig_linear_model_forward is not None and not getattr(_amd_linear.KimiLinearModel, "_patched_hpu_tokens",
                                                                 False):

            def _safe_kimi_linear_model_forward(
                self,
                input_ids=None,
                positions=None,
                intermediate_tensors=None,
                inputs_embeds=None,
                step=None,
                **kwargs,
            ):
                if input_ids is not None and input_ids.dim() == 2:
                    input_ids = input_ids.view(-1)
                if inputs_embeds is not None and inputs_embeds.dim() == 3:
                    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1))
                if positions is not None and positions.dim() == 2:
                    positions = positions.view(-1)

                # imported on first use (circular import with vllm.distributed at plugin-apply time)
                from vllm_gaudi.models.kimi_k3_kda import run_with_deferred_kda_state
                if step is not None:
                    # vLLM 28a5fba06+ passes `step` (5th positional argument) and reads step.layer(idx) per layer;
                    # ac7f3e11e has no such parameter, so it is forwarded only when given.
                    kwargs["step"] = step
                return run_with_deferred_kda_state(
                    self,
                    orig_linear_model_forward,
                    self,
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )

            _amd_linear.KimiLinearModel.forward = _safe_kimi_linear_model_forward
            _amd_linear.KimiLinearModel._patched_hpu_tokens = True
        _kimi_patch_installed("linear_model_tokens_1d")

        # out-of-place attn_res on the decoder layer (2D inputs, bank shape check): the in-place updates of the
        # upstream version do not survive the per-layer graph boundaries of torch.compile mode
        from vllm_gaudi.extension.logger import logger as _ext_logger
        from vllm_gaudi.ops.hpu_kda_pytorch import KIMI_DECODER_LAYER_CLASSES, install_functional_attn_res
        if install_functional_attn_res(_amd_linear) is None:
            _ext_logger().warning(
                "Kimi-K3: none of %s with forward_attn_residual in %s, the in-place upstream "
                "attn_res stays active", KIMI_DECODER_LAYER_CLASSES, _amd_linear.__name__)
        else:
            _kimi_patch_installed("functional_attn_res")

        # Monkey-patch KimiMoE.forward to ensure inputs are safely 2D
        orig_moe_forward = getattr(_amd_linear.KimiMoE, "forward", None)
        if orig_moe_forward is not None and not getattr(_amd_linear.KimiMoE, "_patched_moe_2d", False):

            def _safe_moe_forward(self, hidden_states):
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
                return orig_moe_forward(self, hidden_states)

            _amd_linear.KimiMoE.forward = _safe_moe_forward
            _amd_linear.KimiMoE._patched_moe_2d = True
        _kimi_patch_installed("kimi_moe_2d")
        _kimi_patch_ok("k3_arch")
    except Exception:
        _kimi_patch_failed("k3_arch")


def _hpu_mxfp4_routed_apply(layer, x, topk_weights, topk_ids, group_size):
    """Run the routed experts of an MXFP4 MoE layer on HPU (exact SiTU when needed, else the fused MXFP4 op)."""
    # imported at call time: these modules are not importable yet while the patches are applied
    import vllm_gaudi.envs as gaudi_envs
    from vllm_gaudi.ops.hpu_fused_moe import _normalize_moe_activation
    from vllm_gaudi.ops.hpu_situ_moe import maybe_situ_moe, situ_impl_or_exact

    input_shape = x.shape
    x = x.view(-1, x.shape[-1])

    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(x.dtype)

    if layer.moe_config.dp_size > 1:
        from vllm_gaudi.v1.worker.hpu_dp_utils import get_hpu_dp_metadata, dispatch_tensor
        dp_metadata = get_hpu_dp_metadata()
        hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
        x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

        topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
        topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

        topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
        topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)

    topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
    topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

    # no SiTU in the fused kernel: exact unfused experts over the same packed MXFP4 tensors
    output = maybe_situ_moe(layer, x, topk_ids, topk_weights, situ_impl_or_exact(gaudi_envs.HPU_KIMI_SITU_IMPL),
                            group_size, gaudi_envs.VLLM_MINIMAX_M3_MOE_TOKEN_TILE)
    if output is None:
        # the layer's own activation (was hard-coded "silu"); a name the fused op does not know fails
        # loudly. "situ" only gets here with HPU_KIMI_SITU_IMPL=silu, where the alias maps it to silu.
        output = layer.moe_op(
            x,
            topk_ids,
            topk_weights,
            permuted_weights=True,
            activation=_normalize_moe_activation(layer.activation),
        )

    if layer.moe_config.dp_size > 1:
        return output.view(*(output.size(0), *input_shape[1:]))
    else:
        return output.view(*input_shape)


def _patch_compressed_tensors_mxfp4() -> None:
    """Enable Gaudi native TPC kernel (VllmMixtureOfExpertsOpMXFP4) for CompressedTensorsW4A4Mxfp4MoEMethod."""
    try:
        import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 as mod
        from vllm.platforms import current_platform
        cls = mod.CompressedTensorsW4A4Mxfp4MoEMethod

        orig_init = cls.__init__

        def hpu_init(self, moe):
            orig_init(self, moe)
            if getattr(current_platform, "device_type", None) == "hpu":
                self.mxfp4_backend = None
                self.experts_cls = None
                self.use_cutlass_mxfp4 = False
                self.moe_quant_config = object()

        orig_process = getattr(cls, "process_weights_after_loading", None)

        def hpu_process_weights(self, layer):
            if getattr(current_platform, "device_type", None) != "hpu":
                if orig_process:
                    return orig_process(self, layer)
                return

            if hasattr(layer, "w13_weight_packed"):
                layer.w13_weight = torch.nn.Parameter(layer.w13_weight_packed.data, requires_grad=False)
                delattr(layer, "w13_weight_packed")
            if hasattr(layer, "w2_weight_packed"):
                layer.w2_weight = torch.nn.Parameter(layer.w2_weight_packed.data, requires_grad=False)
                delattr(layer, "w2_weight_packed")

            num_experts = layer.local_num_experts
            ep_shift = layer.moe_config.ep_rank * num_experts
            experts_min, experts_max = ep_shift, num_experts + ep_shift - 1

            try:
                from vllm_gaudi.extension.runtime import get_config
                use_dispatch_fn = get_config().use_dispatch_fn
            except Exception:
                use_dispatch_fn = False

            if layer.moe_config.dp_size > 1 and use_dispatch_fn:
                from functools import partial
                from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states
                dispatch_fn = partial(
                    dispatch_hidden_states,
                    is_sequence_parallel=layer.moe_config.is_sequence_parallel,
                )
            else:
                dispatch_fn = None

            has_bias = hasattr(layer, "w13_bias") and hasattr(layer, "w2_bias")

            from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4
            layer.moe_op = VllmMixtureOfExpertsOpMXFP4(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                block_size=self.group_size,
                has_bias=has_bias,
                dispatch_fn=dispatch_fn,
                in_graph_call=_current_config_is_kimi_k3(),
            )

            for expert_id in range(num_experts):
                layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
                layer.moe_op.w13_list[expert_id].set_scale(layer.w13_weight_scale.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_scale(layer.w2_weight_scale.data[expert_id])
                if has_bias:
                    layer.moe_op.w13_list[expert_id].set_bias(layer.w13_bias.data[expert_id])
                    layer.moe_op.w2_list[expert_id].set_bias(layer.w2_bias.data[expert_id])

            layer.moe_op._cache_weight_lists()
            self.moe_quant_config = object()

        orig_apply = getattr(cls, "apply", None)

        def hpu_apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
            if getattr(current_platform, "device_type", None) != "hpu":
                if orig_apply:
                    return orig_apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
                raise RuntimeError("Non-HPU platform without orig_apply")

            return _hpu_mxfp4_routed_apply(layer, x, topk_weights, topk_ids, self.group_size)

        cls.__init__ = hpu_init
        cls.process_weights_after_loading = hpu_process_weights
        cls.apply = hpu_apply

        try:
            import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as oracle_mod
            orig_oracle = getattr(oracle_mod, "select_deepseek_v4_mxfp4_moe_backend", None)

            def hpu_select_backend(*args, **kwargs):
                if getattr(current_platform, "device_type", None) == "hpu":
                    return (None, None)
                if orig_oracle:
                    return orig_oracle(*args, **kwargs)
                return (None, None)

            oracle_mod.select_deepseek_v4_mxfp4_moe_backend = hpu_select_backend
            _kimi_patch_ok("compressed_tensors_mxfp4_oracle")
        except Exception:
            _kimi_patch_failed("compressed_tensors_mxfp4_oracle")
        _kimi_patch_installed("compressed_tensors_mxfp4")
    except Exception:
        _kimi_patch_failed("compressed_tensors_mxfp4")


def _patch_inc_quantization_config() -> None:
    """Register the HPU ``_FakeINCConfig`` as the resolver for ``inc``.

    The Gaudi runtime-INC (Intel Neural Compressor) flow calibrates fp8 scales
    at runtime and ships no on-disk config, so resolving ``inc`` to vLLM's
    native ``INCConfig`` (whose ``get_config_filenames()`` expects
    ``quantization_config.json``) makes ``get_quant_config`` raise ``Cannot find
    the config file for inc`` — surfacing as a ``VllmConfig`` ValidationError
    during ``create_engine_config`` (exposed by upstream vllm#51695). The
    existing ``ops`` shim on ``get_quantization_config`` is order-sensitive;
    registering via the public ``register_quantization_config`` API is immune to
    import order because ``get_quantization_config`` merges the registry on
    every call. ``inc`` is already in ``QUANTIZATION_METHODS``, so this only
    overrides the config class and does not touch ``current_platform``.
    """
    from vllm.model_executor.layers.quantization import register_quantization_config

    from vllm_gaudi.extension.quant import _FakeINCConfig

    register_quantization_config("inc")(_FakeINCConfig)


def _patch_engine_core_sleep_host_guard() -> None:
    """Refuse ``EngineCore.sleep`` before it pauses scheduling when the model would not fit in host memory.

    HPU sleep moves the weights to CPU, so this is where the container can be OOM-killed.
    The check runs in the engine process on the reports of all workers: a worker raising
    inside the ``sleep`` RPC would leave the other workers' replies queued in
    MultiprocExecutor and shift every later collective RPC by one. All reports are
    collected before any worker moves its shard, so they share one headroom measurement.
    """
    from vllm.v1.engine.core import EngineCore

    from vllm_gaudi.v1.worker.host_headroom import collect_host_headroom_reports, guard_host_headroom

    original_sleep = EngineCore.sleep
    if getattr(original_sleep, "_hpu_host_guard", False):
        return

    @functools.wraps(original_sleep)
    def _hpu_sleep(self, level: int = 1, *args, **kwargs):
        if level >= 1:
            guard_host_headroom(lambda trim: collect_host_headroom_reports(self, trim),
                                action="Sleep",
                                outcome="the model stays on HPU")
        return original_sleep(self, level, *args, **kwargs)

    _hpu_sleep._hpu_host_guard = True  # type: ignore[attr-defined]
    EngineCore.sleep = _hpu_sleep


class _RemoteTokenizerNoticeFilter:
    """Drop the 'Calling super().encode with ...' notice of the Kimi remote tokenizer (tokenization_kimi.py).

    The remote module logs through a ``transformers_modules.*`` logger that has no handler, so the record reaches
    ``logging.lastResort`` (stderr) once per tokenization. Only this message of that namespace is dropped.
    """

    _PREFIX = "Calling super().encode with"

    def filter(self, record) -> bool:
        if record.name.startswith("transformers_modules"):
            return not str(record.getMessage()).startswith(self._PREFIX)
        return True


def _silence_remote_tokenizer_notice() -> None:
    import logging
    handler = logging.lastResort
    if handler is not None and not any(isinstance(f, _RemoteTokenizerNoticeFilter) for f in handler.filters):
        handler.addFilter(_RemoteTokenizerNoticeFilter())


def apply() -> None:
    """Install all HPU runtime monkey-patches."""
    _silence_remote_tokenizer_notice()
    # --- torch.accelerator.empty_cache ---
    torch.accelerator.empty_cache = _hpu_accelerator_empty_cache

    # --- torch.accelerator.empty_host_cache ---
    if hasattr(torch.accelerator, "empty_host_cache"):
        torch.accelerator.empty_host_cache = _hpu_accelerator_empty_host_cache

    # --- torch._C._host_emptyCache ---
    if not hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache = lambda: None

    _patch_hf3fs_mock_client_for_cpu_only()
    _patch_kimi_kda()
    _patch_kimi_attn_res()
    _patch_mla_prefill_selector()
    _patch_moe_runner_delegation()
    _patch_kimi_k3_hybrid_config()
    _patch_kimi_k3_arch()

    # --- Deferred patches (cleanup_dist_env_and_memory + sampler) ---
    # We cannot import ``vllm.distributed.parallel_state`` or the sampler
    # modules here, at platform-registration time.  Their import chain
    # re-enters ``vllm.utils.torch_utils`` while it is still initialising
    # (vllm PR #45424 made ``PIN_MEMORY`` resolve the current platform at
    # ``torch_utils`` import time, and that platform resolution is exactly
    # what triggers this plugin's registration).  The re-entry aborts HPU
    # platform detection ("Failed to infer device type").  Instead we hook
    # into ``load_general_plugins`` which runs in every process (parent +
    # EngineCore subprocess) after the platform is ready.
    import vllm.plugins as _plugins_mod

    _original_load_general = _plugins_mod.load_general_plugins

    def _load_general_with_hpu_patches():
        _original_load_general()
        _patch_cleanup_dist_env_and_memory()
        _patch_batched_count_greater_than()
        _patch_gather_logprobs()
        _patch_granite_hybrid_layer_types()
        _patch_get_num_layers_by_block_type()
        _patch_use_sequence_parallel_moe()
        _patch_check_shm_free_space()
        _patch_mamba_bind_kv_cache()
        _patch_mla_bind_kv_cache()
        _patch_free_blocks()
        _patch_sdpa_attention_forward()
        _patch_inc_quantization_config()
        _patch_engine_core_sleep_host_guard()
        _patch_mla_prefill_selector()
        _patch_moe_runner_delegation()
        _patch_latent_moe_runner_forward()
        _patch_kimi_k3_hybrid_config()
        _patch_kimi_k3_arch()
        _patch_compressed_tensors_mxfp4()

    _plugins_mod.load_general_plugins = _load_general_with_hpu_patches


def patch_hf3fs_mock_client():
    """Guard CUDA sync in the HF3FS mock client on non-CUDA platforms.

    The upstream mock client's ``batch_write`` unconditionally calls
    ``torch.cuda.current_stream().wait_event(event)``, which raises
    ``RuntimeError`` on platforms without CUDA (e.g. HPU). This helper
    installs the CPU-safe replacement for ``batch_write``.

    Called from ``register_utils()`` (general plugin) rather than
    ``apply()`` (platform plugin) to avoid circular imports — the mock
    client transitively imports ``vllm.config`` which is not yet fully
    initialized during platform registration.
    """
    _patch_hf3fs_mock_client_for_cpu_only()
