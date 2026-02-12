---
title: Troubleshooting
---
[](){ #troubleshooting }

# Troubleshooting

This document contains troubleshooting instructions for common issues that you may encounter when using vLLM Hardware Plugin for Intel® Gaudi®.

## FP8 model fails when torch compile is enabled

If your Floating Point 8-bit (FP8) model is not working when torch compile is enabled and you receive the following error, the issue is likely caused by the Runtime Scale Patching feature.

```
AssertionError: Scaling method "ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW" is not supported for runtime scale patching (graph recompile reduction)
```

The default Runtime Scale Patching feature does not support the scaling method that your workload is using for the FP8 execution. To fix the issue, disable Runtime Scale Patching when running this model by exporting `RUNTIME_SCALE_PATCHING=0` in your environment.

## Server error occurs when setting max_concurrency

If setting `max_concurrency` causes the following error, the specified value is likely incorrect.

```
assert num_output_tokens == 0, \
(EngineCore_DP0 pid=545) ERROR 10-13 06:03:56 [core.py:710] AssertionError: req_id: cmpl-benchmark-serving39-0, 236
```

vLLM calculates the maximum available concurrency for current environment based on KV cache settings. To fix the issue, use the value printed in logs:

```
[kv_cache_utils.py:1091] Maximum concurrency for 4,352 tokens per request: 10.59x 
```

In this example, the correct `max_concurrency` value in this specific scenario is `10`.

## Out-of-memory errors occur during inference

Factors such as available HPU memory, model size, and input sequence length may prevent the standard inference command from running successfully for your model, potentially resulting in out-of-memory (OOM) errors. To address these errors, consider the following recommendations:

- Increase `gpu_memory_utilization` to a higher value than the default `0.9`. To address memory limitations, vLLM pre-allocates HPU cache using the percentage of memory defined by `gpu_memory_utilization`. Increasing this value allocates more space for the KV cache.

- Increase `tensor_parallel_size` to a higher value than the default `1`. This method distributes the model weights across HPUs, increasing the memory available for the KV cache on each HPU.

- Decrease `max_num_seqs` or `max_num_batched_tokens`: It may reduce the number of concurrent requests in a batch, leading to lower KV cache usage.

- Disable HPU Graphs completely by switching to any other execution mode to maximize KV cache space allocation.
