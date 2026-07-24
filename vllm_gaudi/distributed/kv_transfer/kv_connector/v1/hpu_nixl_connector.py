# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import inspect
import io
import textwrap
import tokenize

import msgspec
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlBaseConnectorWorker, NixlConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlHandshakePayload,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm_gaudi.platform import logger
import habana_frameworks.torch.utils.experimental as htexp

original_data_ptr = torch.Tensor.data_ptr
# NOTE(Chendi): Temp solution for HPU htexp._data_ptr
# If same tensor assigned with two Views, the htexp._data_ptr() fails on non-in-place view.
# So we record the mapping from original data_ptr to htexp._data_ptr
global_data_ptr_record = {}


def _hpu_data_ptr(tensor_self) -> int:
    """
    A temporary replacement for tensor.data_ptr().

    Checks if the tensor is on an HPU device and if host buffers are not
    in use, then calls the htexp._data_ptr utility. Otherwise, it falls
    back to the original method.
    """
    # The first `self` refers to the class instance (from the outer scope)
    # The `tensor_self` is the tensor instance on which .data_ptr() is called
    if tensor_self.device.type == "hpu":
        # return htexp._data_ptr(tensor_self)
        v_dataptr = original_data_ptr(tensor_self)
        if v_dataptr not in global_data_ptr_record:
            p_dataptr = htexp._data_ptr(tensor_self)
            global_data_ptr_record[v_dataptr] = p_dataptr
        else:
            p_dataptr = global_data_ptr_record[v_dataptr]
        return p_dataptr

    # Fallback to the original implementation for CPU tensors or host buffers
    return original_data_ptr(tensor_self)


def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """
    Initialize transfer buffer in CPU mem for accelerators
    NOT directly supported by NIXL (e.g., tpu)

    NOTE(Chendi): override to support HPU heterogeneousTP size.
    We intend to prepare host_buffer with HND layout as stride layout
    However, we want to keep shape as NHD
    """
    xfer_buffers: dict[str, torch.Tensor] = {}
    inv_order = [0, 1, 3, 2, 4]
    try:
        for layer_name, kv_cache in kv_caches.items():
            kv_shape = kv_cache.shape
            kv_dtype = kv_cache.dtype
            if not self.use_mla:
                kv_shape = tuple(kv_shape[i] for i in inv_order)
            xfer_buffers[layer_name] = torch.empty(kv_shape, dtype=kv_dtype, device="cpu")
            if not self.use_mla:
                xfer_buffers[layer_name] = xfer_buffers[layer_name].permute(inv_order)
    except MemoryError as e:
        logger.error("NIXLConnectorWorker gets %s.", e)
        raise

    self.host_xfer_buffers = xfer_buffers


torch.Tensor.data_ptr = _hpu_data_ptr
NixlConnectorWorker.initialize_host_xfer_buffer = initialize_host_xfer_buffer

# ── HPU TransferTopology.__post_init__ layout-contract override ─────────────── #
# Upstream vLLM PR #44455 ("Pack K/V into the content dim across attention
# backends") added two GPU-centric layout asserts to
# TransferTopology.__post_init__:
#
#   kv_cache_shape = attn_backend.get_kv_cache_shape(num_blocks=1, block_size=16,
#                                                    num_kv_heads=1, head_size=1)
#   assert kv_cache_shape[0] == 1, "KV cache layout must be blocks-first ..."
#   assert len(kv_cache_shape) == 4, "[num_blocks, num_kv_heads, block_size, ...]"
#
# HPU's get_kv_cache_shape() fuses blocks and tokens into the leading dim
# (num_blocks * block_size, num_kv_heads, head_size) — 3-D, block_size-first —
# so the mocked probe returns (16, 1, 1).  Both asserts therefore fire on HPU
# with "got shape (16, 1, 1)", crashing every NIXL PD-disaggregation run before
# any transfer topology is built.
#
# The HPU layout is intentional and never blocks-first, so the GPU-only asserts
# do not apply.  Reimplement __post_init__ for HPU: mirror upstream exactly but
# drop the two layout asserts, preserving every persistent side effect
# (local_physical_heads, _engines, _cross_layers_blocks + cross-layer stride
# reordering).  The final MLA guard keeps the pre-existing HPU fix: the
# _cross_layers_blocks dim-count heuristic (len(tensor_shape) == len(
# kv_cache_shape) + 1) can misfire for MLA host buffers, and MLA models never
# use cross-layer layout, so force it False.


def _hpu_transfer_topo_post_init(self):
    self.local_physical_heads = max(1, self.total_num_kv_heads // self.tp_size)

    self._engines = {}

    # HPU get_kv_cache_shape() is block_size-first, not blocks-first, so the
    # upstream blocks-first / 4-D layout asserts are intentionally skipped here.
    attn_backend = self.attn_backends[0]
    kv_cache_shape: tuple[int, ...] = ()
    if not self.is_mamba:
        _MOCK_BLOCK_SIZE = 16
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            num_blocks=1,
            block_size=_MOCK_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
        )
        logger.debug("[HPU] Test kv_cache_shape: %s", kv_cache_shape)

    self._cross_layers_blocks = False
    if self.tensor_shape is not None:
        self._cross_layers_blocks = len(self.tensor_shape) == len(kv_cache_shape) + 1

    if self._cross_layers_blocks:
        logger.debug("[HPU] Using cross-layer KV cache")
        _MOCK_NUM_LAYERS = 80
        kv_cache_shape = (_MOCK_NUM_LAYERS, ) + tuple(kv_cache_shape)
        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=self._cross_layers_blocks)
        except (AttributeError, NotImplementedError):
            assert self.tensor_shape is not None
            kv_cache_stride_order = tuple(range(len(self.tensor_shape)))
        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

    if self.is_mla and self._cross_layers_blocks:
        logger.warning("[HPU] TransferTopology: overriding false-positive _cross_layers_blocks=True "
                       "for MLA model. HPU get_kv_cache_shape() fuses blocks into the leading dim, "
                       "causing the dim-count heuristic to misfire.  Forcing _cross_layers_blocks=False.")
        self._cross_layers_blocks = False


TransferTopology.__post_init__ = _hpu_transfer_topo_post_init

# ── HPU K/V-split region registration override ─────────────────────────────── #
# Upstream vLLM PR #44456 ("[3/N][KV-Cache Layout Refactor] Standardize Mamba
# cache; drop get_transfer_cache_regions", commit 6700813f) *removed*
# TransferTopology.get_transfer_cache_regions entirely and inlined the region
# registration loop directly into NixlBaseConnectorWorker.register_kv_caches.
# The inlined loop iterates the per-layer caches and registers each as a single
# region, dropping the old K/V split (the cache_list split, the
# ``physical_page_size //= len(cache_list)`` adjustment, and the inner
# per-region loop) that PR #44455's predecessor had exposed via
# get_transfer_cache_regions.
#
# The earlier HPU fix (PR #1616) monkeypatched get_transfer_cache_regions to
# restore the split; after #44456 that method no longer exists, so the patch is
# dead code and the split is lost again.  HPU allocates K and V as two
# *separate* per-layer tensors, surfaced to NIXL as a TensorTuple((K, V)) (no
# host buffer) or a single 5-D tensor (host-buffer path) whose leading dim is
# the K/V split (2), not num_blocks.  With the upstream single-region path,
# register_kv_caches sees ``cache.shape == (2, num_blocks, ...)`` and its
# ``cache.shape[0] != num_blocks`` guard raises:
#
#   AssertionError: All kv cache tensors must have the same number of blocks;
#   ... cache_shape=(2, <N>, ...) ... kv_cache_layout=HND
#
# crashing every NIXL PD-disaggregation run right after the connector logs
# "Registering KV_Caches ... setting KV cache layout to HND".
#
# Restore the pre-#44456 behaviour for HPU by overriding register_kv_caches:
# mirror the upstream inlined loop verbatim, but for non-MLA attention layers
# whose per-layer cache exposes a leading K/V-split dim of 2, register K and V
# as two separate blocks-first regions (halving ``physical_page_size`` per
# region) so ``shape[0] == num_blocks`` holds again.  Mamba / hybrid-SSM / MLA
# layers keep the upstream single-region path.
#
# ── DRIFT HAZARD ─────────────────────────────────────────────────────────── #
# ``_hpu_register_kv_caches`` below is a VERBATIM COPY of
# ``NixlBaseConnectorWorker.register_kv_caches`` @ vllm 61c9ef98
# (base_worker.py:1024), with only three logic deviations (the ``cache_list``
# expansion, the ``physical_page_size //= len(cache_list)`` line, and the inner
# ``for cache in cache_list`` loop). Unlike the ``inspect.getsource`` guards in
# ``patches.py``, a raw copy has no way to notice when upstream reworks the
# descriptor math on the next vllm bump: the copy would silently keep computing
# the OLD region sizes / block lengths and hand NIXL wrong descriptors WITHOUT
# crashing. ``_warn_if_upstream_register_drifted`` (invoked at install time)
# compares a version-stable canonical token digest of the live upstream method
# against the baseline captured at the pin and logs a loud re-sync warning on
# mismatch. It deliberately WARNS rather than raises: this module is imported
# for every HPU worker at ``register_ops`` time (not just PD runs), so a hard
# failure would break unrelated attention/Mamba workloads on any upstream
# reflow. Re-sync procedure on warning: re-copy the upstream body, re-apply the
# three deviations, and refresh ``_UPSTREAM_REGISTER_KV_CACHES_BASELINE``.

# Canonical token digest of NixlBaseConnectorWorker.register_kv_caches @
# vllm 61c9ef98 (comments/docstring/whitespace stripped — see
# _canonical_source_digest). Refresh this whenever the copy below is re-synced
# to a newer vllm pin.
_UPSTREAM_REGISTER_KV_CACHES_BASELINE = "c28bbb0667f915b805af79c39178ffb7c312f8291c90c9318afb5e1a83625176"


def _canonical_source_digest(src: str) -> str:
    """Return a version-stable SHA-256 over the *semantic* tokens of ``src``.

    Drops comments, the leading docstring, and every layout token (newlines,
    indentation), keeping only the ordered token *strings*. This makes the
    digest insensitive to pure reformatting (black vs yapf, line wrapping) so
    only a genuine logic change in upstream trips the drift warning. Stable
    across CPython 3.10–3.12 for code without version-specific syntax.

    Args:
        src: Source text of a single function definition.

    Returns:
        Hex SHA-256 digest of the canonical token stream.
    """
    dedented = textwrap.dedent(src)
    values: list[str] = []
    depth = 0
    def_colon_seen = False
    doc_dropped = False
    for tok in tokenize.generate_tokens(io.StringIO(dedented).readline):
        tok_type, tok_str = tok.type, tok.string
        if tok_type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENCODING, tokenize.ENDMARKER):
            continue
        if tok_type == tokenize.OP and tok_str in "([{":
            depth += 1
        elif tok_type == tokenize.OP and tok_str in ")]}":
            depth -= 1
        elif tok_type == tokenize.OP and tok_str == ":" and depth == 0 and not def_colon_seen:
            # Colon that closes the ``def ...():`` header — the body starts next.
            def_colon_seen = True
            values.append(tok_str)
            continue
        if def_colon_seen and not doc_dropped and tok_type == tokenize.STRING:
            doc_dropped = True  # Drop the function docstring.
            continue
        values.append(tok_str)
    return hashlib.sha256(" ".join(values).encode()).hexdigest()


def _warn_if_upstream_register_drifted() -> None:
    """Warn (never raise) when upstream ``register_kv_caches`` has drifted.

    ``_hpu_register_kv_caches`` is a verbatim copy of the upstream method at the
    pinned vLLM SHA. If a later vllm bump reworks the upstream descriptor math,
    the copy silently keeps the old logic and would feed NIXL stale descriptors
    without crashing. Compare a canonical token digest of the live upstream
    method against the captured baseline and log a loud re-sync warning on
    mismatch. Best-effort: any failure to introspect upstream is swallowed so a
    diagnostic aid never breaks plugin registration.
    """
    try:
        upstream = inspect.unwrap(NixlBaseConnectorWorker.register_kv_caches)
        if upstream is _hpu_register_kv_caches:
            return  # Already patched in this process — nothing to compare.
        live_digest = _canonical_source_digest(inspect.getsource(upstream))
    except (OSError, TypeError, tokenize.TokenError, IndentationError, RecursionError):
        return  # Source unavailable (C/builtin) or unparseable — skip silently.
    if live_digest != _UPSTREAM_REGISTER_KV_CACHES_BASELINE:
        logger.warning(
            "[HPU] NixlBaseConnectorWorker.register_kv_caches has drifted from the vllm 61c9ef98 "
            "baseline that vllm_gaudi's _hpu_register_kv_caches was copied from "
            "(live token digest %s != baseline %s). The HPU K/V-split override may now compute "
            "stale NIXL descriptors. Re-sync _hpu_register_kv_caches to the current upstream body, "
            "re-apply the three HPU deviations, and refresh "
            "_UPSTREAM_REGISTER_KV_CACHES_BASELINE.", live_digest, _UPSTREAM_REGISTER_KV_CACHES_BASELINE)


def _hpu_transfer_cache_regions(transfer_topo, cache, layer_spec):
    """Return the NIXL memory regions for an HPU per-layer KV cache tensor.

    Reinstates the K/V split that vLLM PR #44456 dropped when it removed
    ``TransferTopology.get_transfer_cache_regions``. HPU surfaces K and V as
    two separate per-layer tensors (a K/V-split ``TensorTuple`` or a single
    5-D tensor with a leading K/V dim of 2), so each half is registered as its
    own blocks-first region; Mamba, hybrid-SSM and MLA layers keep the upstream
    single-region layout.

    Args:
        transfer_topo: The active :class:`TransferTopology` for this worker.
        cache: The per-layer KV cache tensor (``(conv, ssm)`` for Mamba, a
            K/V-split ``TensorTuple`` for regular attention, or a single
            tensor for MLA / already-blocks-first layouts).
        layer_spec: The ``KVCacheSpec`` for this layer.

    Returns:
        The list of tensor(s) to register as NIXL memory regions.
    """
    if isinstance(layer_spec, MambaSpec):
        # Register the whole shared conv/ssm tensor (mirror upstream).
        conv, _ssm = cache
        return [conv]

    # Mirror upstream's hybrid-SSM blocks-first transpose.
    if transfer_topo.is_mamba and cache.shape[0] == 2:
        return [cache.transpose(0, 1)]

    # HPU regular attention: K and V surface with a leading K/V-split dim of 2,
    # either as a TensorTuple((K, V)) (no host buffer) or as a single 5-D tensor
    # (host-buffer path).  Register each half as its own blocks-first region so
    # `shape[0] == num_blocks` holds again.  Indexing ([0]/[1]) yields the K and
    # V halves for both the tuple and the tensor form.
    if not transfer_topo.is_mla and len(cache) == 2:
        return [cache[0], cache[1]]

    # MLA (single tensor) and any already-blocks-first layout: single region.
    return [cache]


def _hpu_register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """Register HPU KV caches in NIXL, restoring the pre-#44456 K/V split.

    VERBATIM COPY of ``NixlBaseConnectorWorker.register_kv_caches`` at vllm
    61c9ef98 (base_worker.py:1024), with exactly three logic deviations: the
    per-layer cache is expanded via :func:`_hpu_transfer_cache_regions` into a
    ``cache_list`` so K and V register as two separate blocks-first regions
    instead of the single packed region the upstream inlined loop assumes;
    ``physical_page_size`` is divided by ``len(cache_list)``; and the region
    bookkeeping runs in an inner ``for cache in cache_list`` loop. Everything
    else mirrors upstream. See the DRIFT HAZARD note above — re-sync on the
    ``_warn_if_upstream_register_drifted`` warning.

    Args:
        kv_caches: Mapping of layer name to its device KV cache tensor.
    """
    # Detect packed allocation: all tensors are strided views into the same
    # backing storage (different data_ptr but same storage). DSv4-style packing.
    if len(kv_caches) > 1 and not self._has_mamba:
        storage_ptrs = {cache.untyped_storage().data_ptr() for cache in kv_caches.values()}
        data_ptrs = {cache.data_ptr() for cache in kv_caches.values()}
        if len(storage_ptrs) == 1 and len(data_ptrs) > 1:
            storage = next(iter(kv_caches.values())).untyped_storage()
            self._register_packed_kv_cache(storage)
            self.device_kv_caches = kv_caches
            return

    self.transfer_topo = TransferTopology(
        tp_rank=self.tp_rank,
        tp_size=self.world_size,
        block_size=self.block_size,
        engine_id=self.engine_id,
        is_mla=self.use_mla,
        total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
        attn_backends=self.attn_backends,
        # SSM States come in tuples (ssm, conv)
        tensor_shape=next(iter(kv_caches.values())).shape if not self._has_mamba else None,
        is_mamba=self._has_mamba,
    )
    self.compat_hash = compute_nixl_compatibility_hash(self.vllm_config, self.backend_name,
                                                       self.transfer_topo.cross_layers_blocks)

    if self.use_host_buffer:
        self.initialize_host_xfer_buffer(kv_caches=kv_caches)
        assert len(self.host_xfer_buffers) == len(kv_caches), \
            f"host_buffer: {len(self.host_xfer_buffers)}, kv_caches: {len(kv_caches)}"
        xfer_buffers = self.host_xfer_buffers
    else:
        xfer_buffers = kv_caches
        assert not self.host_xfer_buffers, \
            f"host_xfer_buffer should not be initialized when kv_buffer_device is {self.kv_buffer_device}"

    logger.info("Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, use_host_buffer: %s", self.use_mla,
                self.kv_buffer_device, self.use_host_buffer)

    caches_data = []
    # With hybrid allocator, layers can share a kv cache tensor.
    seen_base_addresses = []
    tensor_size_bytes = None

    for layer_name, cache_or_caches in xfer_buffers.items():
        layer_spec = self._layer_specs.get(layer_name)
        if layer_spec is None:
            logger.debug(
                "Skipping layer %s as no KVCache spec is present. "
                "This is likely because the layer is sharing its KV cache", layer_name)
            continue
        if isinstance(layer_spec, UniformTypeKVCacheSpecs):
            # MLA DSv32 Indexer case: UniformTypeKVCacheSpecs merges kv_cache_specs.
            layer_spec = layer_spec.kv_cache_specs[layer_name]

        # HPU deviation: expand K/V into separate blocks-first regions.
        cache_list = _hpu_transfer_cache_regions(self.transfer_topo, cache_or_caches, layer_spec)

        # `layer_spec.page_size_bytes` only accounts for logical page_size.
        physical_page_size = (layer_spec.page_size_bytes if isinstance(layer_spec, MambaSpec) else
                              layer_spec.page_size_bytes // self._physical_blocks_per_logical_kv_block)
        # For when registering multiple tensors e.g. K/V in separate regions.
        physical_page_size = physical_page_size // len(cache_list)
        if self.transfer_topo._cross_layers_blocks:
            # When cross-layers blocks are used, multiply by number of layers.
            physical_page_size = physical_page_size * len(self.kv_cache_config.kv_cache_tensors)
        num_blocks = (self._logical_num_blocks if isinstance(layer_spec, MambaSpec) else self.num_blocks)
        # `page_size` accounts for physical blocks, st KVCache is always
        # [`num_blocks` * `page_size`].
        curr_tensor_size_bytes = num_blocks * physical_page_size

        for cache in cache_list:
            base_addr = cache.data_ptr()
            if base_addr in seen_base_addresses:
                logger.debug("Skipping %s because it's already seen", layer_name)
                continue
            logger.debug("Registering layer %s with cache shape: %s", layer_name, cache.shape)
            seen_base_addresses.append(base_addr)
            # Only record non-Mamba page sizes.
            if isinstance(layer_spec, MambaSpec):
                self.block_len_per_layer.append(physical_page_size // self._physical_blocks_per_logical_kv_block)
            else:
                self.block_len_per_layer.append(physical_page_size)
            is_mla_region = isinstance(layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec))
            self._region_is_mla.append(is_mla_region)

            if not is_mla_region:
                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                assert tensor_size_bytes == curr_tensor_size_bytes, \
                    "All non-MLA kv cache tensors must have the same size"

            # When there's a mismatch between kbs<>bs, we rely on HMA to ensure
            # caches are either [NB, PS] or [NB*r, PS/r] where r is bs/kbs.
            if self._physical_blocks_per_logical_kv_block == 1 and cache.shape[0] != num_blocks:
                raise AssertionError("All kv cache tensors must have the same number of "
                                     f"blocks; layer={layer_name}, "
                                     f"expected_num_blocks={num_blocks}, "
                                     f"cache_shape={tuple(cache.shape)}, "
                                     f"cache_stride={tuple(cache.stride())}, "
                                     f"layer_spec={type(layer_spec).__name__}, "
                                     f"backend={self.backend_name}, "
                                     "all_backends="
                                     f"{[backend.get_name() for backend in self.attn_backends]}, "
                                     f"kv_cache_layout={self.kv_cache_layout}")

            # Need to make sure the device ID is non-negative for NIXL,
            # Torch uses -1 to indicate CPU tensors.
            self.device_id = max(cache.get_device(), 0)
            caches_data.append((base_addr, curr_tensor_size_bytes, self.device_id, ""))

    logger.debug("Different block lengths collected: %s", set(self.block_len_per_layer))
    assert len(self.block_len_per_layer) == len(seen_base_addresses) == len(self._region_is_mla)

    self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
    self.num_regions = len(caches_data)

    if self.pp_size > 1:
        start_layer, end_layer = self.model_config.get_layers_start_end_indices(self.vllm_config.parallel_config)
        num_local_layers = end_layer - start_layer
        assert num_local_layers > 0 and self.num_regions % num_local_layers == 0
        regions_per_layer = self.num_regions // num_local_layers
        self._remote_region_offset = regions_per_layer * start_layer

    # Total local FA descriptors (boundary between FA and mamba descs).
    self.num_descs = self.num_regions * self.num_blocks

    descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
    logger.debug("Registering descs: %s", caches_data)
    self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
    logger.debug("Done registering descs")
    self._registered_descs.append(descs)

    self.device_kv_caches = kv_caches
    self.dst_num_blocks[self.engine_id] = self.num_blocks

    if self._has_mamba:
        logger.info(
            "Hybrid SSM registration: num_blocks=%s, logical_num_blocks=%s, ratio=%s, "
            "num_regions=%s, num_descs=%s, mamba_ssm_size=%s, block_len_per_layer=%s", self.num_blocks,
            self._logical_num_blocks, self._physical_blocks_per_logical_kv_block, self.num_regions, self.num_descs,
            self._mamba_ssm_size, set(self.block_len_per_layer))

    # Register local/src descr for NIXL xfer.
    self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = \
        self.register_local_xfer_handler(self.block_size)

    # After KV Caches registered, listen for new connections.
    agent_metadata = NixlAgentMetadata(
        engine_id=self.engine_id,
        agent_metadata=self.nixl_wrapper.get_agent_metadata(),
        device_id=self.device_id,
        kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
        num_blocks=self.num_blocks,
        block_lens=self.block_len_per_layer,
        kv_cache_layout=self.kv_cache_layout if not self.use_host_buffer else self.host_buffer_kv_cache_layout,
        block_size=self.block_size,
        ssm_sizes=self._mamba_ssm_size,
        attn_backend_name=self.backend_name,
        physical_blocks_per_logical_kv_block=self._physical_blocks_per_logical_kv_block,
    )
    # Wrap metadata in payload with hash for defensive decoding.
    assert self.compat_hash is not None
    encoder = msgspec.msgpack.Encoder()
    self.xfer_handshake_metadata = NixlHandshakePayload(
        compatibility_hash=self.compat_hash,
        agent_metadata_bytes=encoder.encode(agent_metadata),
    )


# Warn if upstream's register_kv_caches drifted from the copied baseline BEFORE
# we overwrite it — afterwards the attribute points at our copy and the digest
# comparison is meaningless.
_warn_if_upstream_register_drifted()

# Patch the base worker so both the pull worker (NixlConnectorWorker, which
# inherits register_kv_caches unchanged) and the push worker (which calls
# super().register_kv_caches) pick up the HPU K/V split.
NixlBaseConnectorWorker.register_kv_caches = _hpu_register_kv_caches
