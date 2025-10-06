---
title: Troubleshooting
---
[](){ #troubleshooting }

1. My fp8 model is not working when torch compiled with following error:

```
AssertionError: Scaling method "ScaleMethodString.ACT_MAXABS_PCS_POW2_WEIGHT_MAXABS_PTS_POW2_HW" is not supported for runtime scale patching (graph recompile reduction)
```

- Currently set by default feature e.g. Runtime Scale Pataching does not support scaling method that your workload is used for fp8 execution. Please disable Runtime Scale Patching when running this model by exporting RUNTIME_SCALE_PATCHING=0 in your environment.
