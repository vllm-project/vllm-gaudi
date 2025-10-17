---
title: Troubleshooting
---
[](){ #troubleshooting }

### 1. My FP8 model is not working when torch compile is enabled with following error:

```
AssertionError: Scaling method "ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW" is not supported for runtime scale patching (graph recompile reduction)
```

#### Solution:
The default feature, _Runtime Scale Patching_ does not support scaling method that your workload is using for fp8 execution. Please disable Runtime Scale Patching when running this model by exporting **RUNTIME_SCALE_PATCHING=0** in your environment.

### 2. Setting max_concurrency causes server to throw error:

```
assert num_output_tokens == 0, \
(EngineCore_DP0 pid=545) ERROR 10-13 06:03:56 [core.py:710] AssertionError: req_id: cmpl-benchmark-serving39-0, 236
```

#### Solution:
Vllm calculates maximum available concurrency for current environment based on kv cache settings. Please use value printed in logs:

```
[kv_cache_utils.py:1091] Maximum concurrency for 4,352 tokens per request: 10.59x 
```

So for this specific scenario correct value for --max_concurrency is 10
