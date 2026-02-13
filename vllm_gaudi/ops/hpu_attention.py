from __future__ import annotations

import torch

import vllm.attention.layer as layer
from vllm.forward_context import ForwardContext, get_forward_context

if not hasattr(layer.Attention, "_vllm_gaudi_original_forward"):
    layer.Attention._vllm_gaudi_original_forward = layer.Attention.forward


def patched_attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    # For some alternate attention backends like MLA the attention output
    # shape does not match the query shape, so we optionally let the model
    # definition specify the output tensor shape.
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
    """
    The KV cache is stored inside this class and is accessed via
    `self.kv_cache`.

    Attention metadata (`attn_metadata`) is set using a context manager in
    the model runner's `execute_model` method. It is accessed via forward
    context using
    `vllm.forward_context.get_forward_context().attn_metadata`.
    """
    if self.use_output or not self.use_direct_call:
        return layer.Attention._vllm_gaudi_original_forward(self, query, key, value, output_shape=output_shape)

    if self.calculate_kv_scales:
        torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)
    if self.query_quant is not None:
        # quantizing with a simple torch operation enables
        # torch.compile to fuse this into previous ops
        # which reduces overheads during decoding.
        # Otherwise queries are quantized using custom ops
        # which causes decoding overheads
        assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}

        # check if query quantization is supported
        if self.impl.supports_quant_query_input:
            query, _ = self.query_quant(query, self._q_scale)

    assert self.attn_backend.forward_includes_kv_cache_update, (
        "Split KV cache update not supported when output tensor not provided.")

    # direct call
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[self.layer_name]
    self_kv_cache = self.kv_cache[forward_context.virtual_engine]
    return self.impl.forward(self, query, key, value, self_kv_cache, attn_metadata)


if getattr(layer.Attention.forward, "__name__", "") != "patched_attention_forward":
    layer.Attention.forward = patched_attention_forward
