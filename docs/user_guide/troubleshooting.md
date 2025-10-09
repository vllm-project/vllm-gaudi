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
