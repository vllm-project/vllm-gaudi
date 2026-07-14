# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
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
