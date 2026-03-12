# KV Cache Initialization: HPU vs GPU Model Runner Comparison

This document compares how KV cache initialization is handled in the HPU model runner
(`vllm_gaudi/v1/worker/hpu_model_runner.py`) versus the upstream GPU model runner
(`vllm/v1/worker/gpu_model_runner.py`), with a focus on tensor allocation and hybrid
models that use Mamba2 layers.

## Overview

Both model runners follow the same high-level flow:

1. **Discover KV cache specs** (`get_kv_cache_spec`) — determine what kind of cache each layer needs
2. **Allocate memory** — create raw tensors for KV cache storage
3. **Reshape / assign** — reshape raw allocations into per-layer cache tensors
4. **Bind to model** (`bind_kv_cache`) — wire the tensors into the model's forward context

The key differences lie in *how* tensors are allocated (step 2–3) and how hybrid
Attention + Mamba models are handled.

---

## 1. KV Cache Spec Discovery (`get_kv_cache_spec`)

### GPU model runner (upstream)

Uses polymorphic dispatch via `AttentionLayerBase`:

```python
layer_type = cast(type[Any], AttentionLayerBase)
attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
for layer_name, attn_module in attn_layers.items():
    if isinstance(attn_module, Attention) and attn_module.kv_sharing_target_layer_name:
        self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
        continue
    if spec := attn_module.get_kv_cache_spec(self.vllm_config):
        kv_cache_spec[layer_name] = spec
```

Each layer module (`Attention`, `MLAAttention`, `MambaBase`, etc.) implements its own
`get_kv_cache_spec()` method that returns the appropriate `KVCacheSpec` subtype
(`FullAttentionSpec`, `SlidingWindowSpec`, `MLAAttentionSpec`, `MambaSpec`, etc.).

### HPU model runner (after this PR)

Now **aligned with upstream** — uses the same polymorphic dispatch pattern. Previously,
the HPU runner manually checked `isinstance` against each module type (`Attention`,
`MLAAttention`, `MambaBase`) and constructed specs inline, which missed spec subtypes
like `SlidingWindowSpec` and `CrossAttentionSpec`.

---

## 2. Tensor Allocation

### GPU model runner (upstream)

The GPU runner uses a two-phase approach:

**Phase 1 — `_allocate_kv_cache_tensors`**: Allocates one flat `int8` tensor per
`kv_cache_tensor` entry, sized exactly to the byte count from the KV cache planner:

```python
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

All layers that share a `kv_cache_tensor` get the *same* raw tensor object — this is
how Mamba layers share memory.

**Phase 2 — `_reshape_kv_cache_tensors`**: Reshapes each raw tensor into the format
expected by the layer:

- **Attention layers**: `raw_tensor.view(dtype).view(kv_cache_shape).permute(...)` —
  a single contiguous tensor holds both K and V interleaved.
- **Mamba layers**: `torch.as_strided(raw_tensor.view(dtype), ...)` — creates strided
  views over the raw buffer for `conv_state` and `ssm_state` tensors.

For hybrid models, an additional `_update_hybrid_attention_mamba_layout` step re-strides
attention KV caches from `(2, num_blocks, ...)` to `(num_blocks, 2, ...)` layout.

### HPU model runner

The HPU runner creates **separate** tensors for keys and values (HPU architecture
requires split K/V buffers for its attention kernels). It uses a single-pass approach
iterating through `kv_cache_config.kv_cache_tensors`:

```python
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    for layer_name in kv_cache_tensor.shared_by:
        if layer_name in kv_caches:
            continue  # already created by shared mamba layer
        # ... look up kv_cache_spec for this layer from kv_cache_groups ...
        if isinstance(kv_cache_spec, AttentionSpec):
            key_cache = torch.zeros(kv_cache_shape, dtype=dtype, device=self.device)
            value_cache = torch.zeros(v_cache_shape, dtype=dtype, device=self.device)
            kv_caches[layer_name] = (key_cache, value_cache, key_scales, value_scales)
        elif isinstance(kv_cache_spec, MambaSpec):
            state_tensors = []
            for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                tensor = torch.zeros((num_blocks + 1, *shape), dtype=dtype, device=self.device)
                state_tensors.append(tensor)
            for shared_layer in kv_cache_tensor.shared_by:
                kv_caches[shared_layer] = tuple(state_tensors)
```

Key differences from GPU:

| Aspect | GPU Runner | HPU Runner |
|--------|-----------|------------|
| Raw allocation | Single flat `int8` tensor per `kv_cache_tensor` | Direct per-type allocation (no raw buffer) |
| K/V layout | Single tensor, K and V interleaved | Separate `key_cache` and `value_cache` tensors |
| Cache tuple | `kv_cache[layer]` = single tensor | `kv_cache[layer]` = `(key, value, key_scales, value_scales)` |
| Mamba views | `torch.as_strided` over shared raw buffer | `torch.zeros` per state + shared via assignment |
| Dummy block | Not used | `num_blocks + 1` (extra block for padding) |
| FP8 scales | Not in KV cache init | Dynamic scale tensors created alongside K/V |

---

## 3. Hybrid Model Handling (Attention + Mamba2)

Hybrid models (e.g., Bamba, Jamba) have **both** attention layers and Mamba2 layers.
Each layer type has its own `KVCacheSpec`:

- **Attention layers**: `AttentionSpec` (or subtypes: `FullAttentionSpec`,
  `SlidingWindowSpec`) — need K/V cache blocks
- **Mamba2 layers**: `MambaSpec` — need `conv_state` and `ssm_state` cache blocks

The KV cache planner groups layers into `KVCacheGroupSpec` entries. Mamba layers that
share the same state specification are grouped together and share a single
`kv_cache_tensor`.

### GPU runner — Mamba tensor sharing

On GPU, sharing works via the raw buffer: all Mamba layers in the same group point to
the *same* `int8` tensor. `_reshape_kv_cache_tensors` then creates `as_strided` views
over this shared buffer:

```python
# Multiple layers get the same raw_tensor from _allocate_kv_cache_tensors
raw_tensor = kv_cache_raw_tensors[layer_name]  # shared object
state_tensors = []
for shape, dtype in zip(spec.shapes, spec.dtypes):
    tensor = torch.as_strided(raw_tensor.view(dtype), size=..., stride=..., storage_offset=...)
    state_tensors.append(tensor)
kv_caches[layer_name] = state_tensors
```

### HPU runner — Mamba tensor sharing

On HPU, sharing works by creating the state tensors once for the first Mamba layer
encountered in a `kv_cache_tensor` group, then assigning the same tuple to all layers
in that group:

```python
elif isinstance(kv_cache_spec, MambaSpec):
    state_tensors = []
    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
        tensor = torch.zeros((num_blocks + 1, *shape), dtype=dtype, device=self.device)
        state_tensors.append(tensor)
    # Share across all layers in this kv_cache_tensor group
    for shared_layer in kv_cache_tensor.shared_by:
        kv_caches[shared_layer] = tuple(state_tensors)
```

Subsequent layers in `kv_cache_tensor.shared_by` are skipped via the `if layer_name in
kv_caches: continue` guard.

---

## 4. `bind_kv_cache`

Both runners use the shared `bind_kv_cache` utility from `vllm.v1.worker.utils` to wire
tensors into the model's forward context and the runner's `kv_caches` list. Both now pass
`num_attn_module` to handle models with multiple attention modules per decoder layer
(e.g., encoder-decoder architectures like `longcat_flash`).

---

## Summary of Changes in This PR

| Change | Before | After |
|--------|--------|-------|
| `get_kv_cache_spec` | Manual `isinstance` checks against `Attention`, `MLAAttention`, `MambaBase` with inline spec construction | Polymorphic `attn_module.get_kv_cache_spec(vllm_config)` via `AttentionLayerBase` |
| `isinstance` checks | `FullAttentionSpec` (misses subtypes) | `AttentionSpec` (base class, catches all subtypes) |
| KV cache init paths | 3 env-var-gated paths: `VLLM_USE_HYBRID_CACHE`, `VLLM_USE_NAIVE_MAMBA_CACHE_SHARING`, non-hybrid | Single unified path handling both hybrid and non-hybrid |
| Mamba tensor sharing | Path 1: wasted memory (raw buffer unused by attention); Path 2: manual sharing loop; Path 3: no sharing | Automatic sharing via `kv_cache_tensor.shared_by` iteration |
| `bind_kv_cache` | Missing `num_attn_module` parameter | Passes `num_attn_module` matching upstream |
| Env vars removed | `VLLM_USE_HYBRID_CACHE`, `VLLM_USE_NAIVE_MAMBA_CACHE_SHARING` | N/A |
| Unused imports removed | `MLAAttention`, `MambaBase`, `FullAttentionSpec`, `MLAAttentionSpec`, `get_dtype_size` | N/A |
