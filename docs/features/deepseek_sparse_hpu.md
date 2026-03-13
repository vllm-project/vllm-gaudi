---
title: DeepSeek V3 Sparse Attention on HPU
---
[](){ #deepseek_sparse_hpu }

## Overview

The vLLM Hardware Plugin for Intel® Gaudi® supports **DeepSeek V3 / V3.2** models that use sparse attention on top of MLA (Multi-head Latent Attention). Sparse token selection is performed in the model’s custom attention layer before the HPU attention backend; the backend then runs on the already-selected tokens.

This integration enables:

- **Chunked prefill** for long context (e.g. 16k and 32k token lengths) with correct cross-chunk and prefix-cache behaviour.
- **Long-context robustness**: int64 position/length indices and RoPE scaling support to avoid overflow and support DeepSeek-style positional encoding.

## Backend selection

Backend choice is determined in the HPU platform layer (`get_attn_backend_cls`):

- **Sparse + MLA**: If unified attention is enabled, `HPUUnifiedMLABackend` is used; otherwise (chunked prefill mode) `HPUMLAAttentionBackend` is used so that metadata matches `TrimmedAttentionMetadata` (attn_bias, block_list).
- **Sparse without MLA**: `HPUUnifiedAttentionBackend` is used.

See `vllm_gaudi/platform.py` for the exact conditions and log messages.

## Chunked prefill (16k / 32k)

For models with `attention_chunk_size` (e.g. DeepSeek V3.2):

- **Prefill**: When prefix cache is present (cross-chunk context), the MLA prefill path uses `naive_impl` for attention so that rectangular Q/KV and attention bias are handled correctly; the first chunk (no cached context) keeps the default implementation with causal masking and valid sequence lengths.
- **Decode**: Only the last-chunk KV blocks are used for decode, so the attention context window matches the chunked prefill view. The model runner sets `model_has_chunked_attention` and marks chunked attention layers so that decode metadata and prefill bias use the chunked block lists and `chunked_attn_bias`.

Relevant code: `vllm_gaudi/attention/backends/hpu_attn.py` (HPUMLAImpl prefill, HPUAttentionMetadata chunked_* fields), `vllm_gaudi/v1/worker/hpu_model_runner.py` (maybe_set_chunked_attention_layers, decode chunked block construction).

## Long-context and overflow

To support long context (e.g. 32k+ tokens) without overflow:

- **Position / length indices**: The model runner keeps sequence length and position ranges in int64 where needed (e.g. `arange_np` and related buffers).
- **RoPE**: The HPU rotary embedding layer supports offsets (long-context, LoRA) and scaling factors (e.g. DeepSeek-style RoPE) by recomputing cos/sin when `scaling_factors` or `scaling_factor` are present or when offsets require it.

See `vllm_gaudi/v1/worker/hpu_model_runner.py` (buffer comments) and `vllm_gaudi/ops/hpu_rotary_embedding.py` (prepare_cos_sin, forward_oot).

## Model registration

DeepSeek V3 is registered in `vllm_gaudi/models/__init__.py` as `DeepseekV3ForCausalLM` pointing to `HpuDeepseekV3ForCausalLM`, with support for chunked prefill and long context when using the HPU MLA backends.
