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

* ``vllm.v1.sample.sampler.Sampler.gather_logprobs`` — upstream PR #38933
  added two ``mark_unbacked()`` calls inside ``gather_logprobs`` to prevent
  0->1 batch-size specialization recompiles for ``batched_count_greater_than``
  on CUDA.  On HPU, ``torch._dynamo`` marks ``mark_unbacked`` as a forbidden
  callable (``is_forbidden=True``), so tracing a compiled ``Sampler`` raises
  ``AssertionError: Attempt to trace forbidden callable mark_unbacked`` for
  any request with ``logprobs=true``.  We replace ``gather_logprobs`` with an
  identical implementation that omits the two ``mark_unbacked`` calls.  HPU
  handles dynamic batch shapes without this hint.

* ``vllm.v1.core.block_pool.BlockPool.free_blocks`` — upstream PR #42656
  ("Apply LRU policy only to proper cache entries") made ``free_blocks``
  partition freed blocks into with-hash / without-hash lists and issue two
  queue ops (``prepend_n`` + ``append_n``) on every engine step.  When prefix
  caching is disabled every block's hash is ``None``, so the split is a no-op
  that only adds per-step CPU overhead — a measurable decode-throughput loss
  on small/low-batch models.  We restore a single-pass free for the
  ``enable_caching=False`` case and delegate to the original implementation
  when prefix caching is on.  Remove once fixed upstream.
"""

import gc
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

    _sampler_mod.Sampler.gather_logprobs = staticmethod(_hpu_gather_logprobs)


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

        if layers_block_type_value is None and attn_type_list is None and layer_types_value is None:
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
    if "_GRANITE_LEGACY_LAYER_ALIASES" in source or "normalized_block_type" in source:
        return  # Already normalizes (e.g. PR #47634 merged).

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


def apply() -> None:
    """Install all HPU runtime monkey-patches."""
    # --- torch.accelerator.empty_cache ---
    torch.accelerator.empty_cache = _hpu_accelerator_empty_cache

    # --- torch._C._host_emptyCache ---
    if not hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache = lambda: None

    _patch_hf3fs_mock_client_for_cpu_only()

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
        _patch_free_blocks()

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
