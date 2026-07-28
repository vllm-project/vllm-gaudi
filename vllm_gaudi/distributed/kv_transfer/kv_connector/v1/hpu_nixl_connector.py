# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.utils import (
    TransferTopology,
    kv_postprocess_blksize_and_layout_on_receive,
    kv_postprocess_blksize_on_receive,
    kv_postprocess_layout_on_receive,
)
from vllm.v1.kv_cache_interface import MambaSpec
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

# ── HPU TransferTopology.get_transfer_cache_regions K/V-split override ──────── #
# Upstream vLLM PR #44455 ("Pack K/V into the content dim across attention
# backends") rewrote get_transfer_cache_regions to always return a single
# packed region ([cache]) and dropped the old split_k_and_v / blocks-first
# handling.  This assumes the GPU layout where K and V are interleaved inside
# each block, so the registered tensor is blocks-first (shape[0] == num_blocks).
#
# HPU allocates K and V as two *separate* per-layer tensors, surfaced to NIXL
# as a TensorTuple((K, V)) whose leading dim is the K/V split (2), not
# num_blocks.  With the upstream single-region path, register_kv_caches sees
# cache.shape == (2, num_blocks, block_size, num_kv_heads, head_size) and its
# `cache.shape[0] != num_blocks` guard raises:
#
#   AssertionError: All kv cache tensors must have the same number of blocks;
#   ... cache_shape=(2, <N>, ...) ... kv_cache_layout=HND
#
# crashing every NIXL PD-disaggregation run right after the connector logs
# "setting KV cache layout to HND".
#
# Restore the pre-#44455 behaviour for HPU: register K and V as two separate
# blocks-first regions so each registered tensor is blocks-first
# (shape[0] == num_blocks) again.  Mirror upstream exactly for the Mamba and
# hybrid-SSM branches (they are backend-agnostic and already blocks-first), and
# only iterate the tuple for the regular attention path.  MLA layers surface a
# single tensor (not a TensorTuple), so they naturally fall through to the
# single-region return.


def _hpu_get_transfer_cache_regions(self, cache, layer_spec):
    """Return the NIXL memory regions for an HPU KV cache tensor.

    HPU packs K and V as two separate per-layer tensors (a K/V-split
    ``TensorTuple``) rather than a single blocks-first tensor, so this
    override registers each K/V half as its own blocks-first region. This
    reverses vLLM PR #44455's single-region packing, which would otherwise
    make ``register_kv_caches`` see a leading K/V dim of 2 and fail its
    ``cache.shape[0] == num_blocks`` contract.

    Args:
        cache: The per-layer KV cache tensor (or ``(conv, ssm)`` for Mamba,
            or a K/V-split ``TensorTuple`` for regular attention).
        layer_spec: The ``KVCacheSpec`` for this layer.

    Returns:
        The tensor(s) to register as NIXL memory regions.
    """
    if isinstance(layer_spec, MambaSpec):
        # Register the whole shared conv/ssm tensor (mirror upstream).
        conv, _ssm = cache
        return [conv]

    # Mirror upstream's hybrid-SSM blocks-first transpose.
    if self.is_mamba and cache.shape[0] == 2:
        return [cache.transpose(0, 1)]

    # HPU regular attention: K and V surface with a leading K/V-split dim of 2,
    # either as a TensorTuple((K, V)) (no host buffer) or as a single 5-D tensor
    # (host-buffer path).  Register each half as its own blocks-first region so
    # `shape[0] == num_blocks` holds again.  Indexing ([0]/[1]) yields the K and
    # V halves for both the tuple and the tensor form.
    if not self.is_mla and len(cache) == 2:
        return [cache[0], cache[1]]

    # MLA (single tensor) and any already-blocks-first layout: single region.
    return [cache]


TransferTopology.get_transfer_cache_regions = _hpu_get_transfer_cache_regions

# ── HPU TransferTopology layout-property restore ────────────────────────────── #
# Upstream #44455 deleted two layout properties along with the blocks-first
# handling above.  The hetero connector still reads both, so every hetero
# (VLLM_HPU_HETERO_KV_LAYOUT=true) run dies at startup with:
#
#   AttributeError: 'TransferTopology' object has no attribute 'split_k_and_v'
#   AttributeError: 'TransferTopology' object has no attribute
#                   'is_kv_layout_blocks_first'
#
# They describe two mutually exclusive layouts, so restore both together:
#
#   * is_kv_layout_blocks_first -- joint K/V packed inside each block, with
#     num_blocks leading (the FlashInfer/GPU 5-D layout).  Upstream computed
#     this as `is_mamba or (len(kv_cache_shape) == 5 and shape[0] == 1)`.
#     HPU's get_kv_cache_shape() returns the 3-D
#     (num_blocks * block_size, num_kv_heads, head_size) -- neither 5-D nor
#     blocks-first -- and the hetero path hardcodes is_mamba=False, so this is
#     always False here.  Keeping the upstream expression (rather than a bare
#     `False`) means a future HPU backend that does adopt a 5-D blocks-first
#     cache starts reporting True on its own.
#
#   * split_k_and_v -- K and V registered as two separate regions.  HPU
#     surfaces regular attention layers as a TensorTuple((K, V)) (see
#     HPUModelRunner.get_kv_caches_4D), so callers must iterate the pair
#     instead of treating the entry as one packed tensor.  True except for MLA
#     (single fused latent tensor) and cross-layer blocks (per-layer K/V
#     interleaving, where a flat pair-split does not separate K from V).
#
# Kept here rather than in the hetero module so both HPU connectors see them --
# hpu_nixl_connector is always imported first (vllm_gaudi/__init__.py).


def _hpu_is_kv_layout_blocks_first(self) -> bool:
    """Whether K/V are packed jointly per block with num_blocks leading."""
    if self.is_mamba:
        return True
    attn_backend = self.attn_backends[0]
    _MOCK_BLOCK_SIZE = 16
    kv_cache_shape = attn_backend.get_kv_cache_shape(
        num_blocks=1,
        block_size=_MOCK_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
    )
    return len(kv_cache_shape) == 5 and kv_cache_shape[0] == 1


def _hpu_split_k_and_v(self) -> bool:
    """Whether K and V are registered as two separate regions on HPU."""
    return not (self._cross_layers_blocks or self.is_mla or self.is_kv_layout_blocks_first)


TransferTopology.is_kv_layout_blocks_first = property(_hpu_is_kv_layout_blocks_first)
TransferTopology.split_k_and_v = property(_hpu_split_k_and_v)

# ── HPU post_process_device_kv_on_receive K/V-split restore ─────────────────── #
# Same root cause as the two properties above.  Before #44455, this method
# iterated the K/V pair:
#
#     cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
#     for cache in cache_list: ...
#
# #44455 assumed one packed tensor per layer and dropped the loop, calling the
# kv_postprocess_* helpers on the entry directly.  On HPU each entry is a
# TensorTuple((K, V)) (HPUModelRunner.get_kv_caches_4D), so the first helper
# call dies on the D side right after the transfer completes:
#
#   AttributeError: 'TensorTuple' object has no attribute 'index_select'
#
# Restore the pre-#44455 loop.  Everything else -- the ratio assert, which of
# the three kv_postprocess_* helpers runs, and the per-group indices -- mirrors
# current upstream, so packed-layout backends still take the single-cache path.


def _hpu_post_process_device_kv_on_receive(self, block_size_ratio: int, block_ids_list: list[list[int]]) -> None:
    """Transform received KV blocks into the local block size / layout on HPU.

    Args:
        block_size_ratio: local block size / remote block size (>= 1).
        block_ids_list: per-KV-cache-group lists of local (kernel) block ids
            that were just received into.
    """
    if len(self.device_kv_caches) == 0:
        return
    assert block_size_ratio >= 1, "Only nP < nD supported currently."
    assert self.transfer_topo is not None
    if self.enable_permute_local_kv and block_size_ratio > 1:
        logger.debug(
            "Post-processing device kv cache on receive by converting block_size with %sx bigger "
            "and permuting layout from HND to NHD.", block_size_ratio)
    elif self.enable_permute_local_kv:
        logger.debug("Post-processing device kv cache on receive by permuting layout from HND to NHD.")
    else:
        logger.debug("Post-processing device kv cache on receive by converting block_size with %sx bigger.",
                     block_size_ratio)

    split_k_and_v = self.transfer_topo.split_k_and_v

    for block_ids in block_ids_list:
        indices = torch.tensor(block_ids, device=self.device_type, dtype=torch.long)
        for cache_or_caches in self.device_kv_caches.values():
            # HPU surfaces regular attention as a (K, V) pair; iterate it so the
            # helpers below always receive a real 4-D blocks-first tensor.
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            for cache in cache_list:
                if self.enable_permute_local_kv and block_size_ratio > 1:
                    kv_postprocess_blksize_and_layout_on_receive(cache, indices, block_size_ratio)
                elif self.enable_permute_local_kv:
                    kv_postprocess_layout_on_receive(cache, indices)
                else:
                    kv_postprocess_blksize_on_receive(cache, indices, block_size_ratio)


NixlConnectorWorker.post_process_device_kv_on_receive = _hpu_post_process_device_kv_on_receive
