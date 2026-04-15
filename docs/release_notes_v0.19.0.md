# vLLM Gaudi Plugin v0.19.0 Release Notes

## Overview

This release is based on [vLLM v0.19.0](https://github.com/vllm-project/vllm/releases/tag/v0.19.0) and supports [Intel® Gaudi® Software v1.24.0](https://docs.habana.ai/en/v1.24.0/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.10.

---

## Highlights

- Upgraded platform compatibility to **Intel® Gaudi® Software v1.24.0** and **PyTorch 2.10**, advancing from the v1.23.0 / PT 2.9 baseline used in previous releases.
- Added initial **Qwen 3.5** model support with compact mode for improved memory utilization (#1153, #1235, #1312).
- Introduced **Mamba prefix caching** support for hybrid SSM-Transformer models on v0.19.0 (#1330).
- Added **MxFP4 weight loading and dequantization** for next-generation quantization formats (#1156).
- Integrated **LMCache** support via monkey-patching for external cache backends (#1176).
- Introduced **custom depthwise conv1d TPC kernel** for MambaMixer2 to improve hybrid model performance (#1092).
- Adapted the **online defragmenter for torch.compile** mode, enabling memory defragmentation in compiled execution (#986).
- Improved warmup performance by **capping decode block bucket limits** (#1160).

---

## New Model Support

- Qwen 3.5 initial enablement ([#1153](https://github.com/vllm-project/vllm-gaudi/pull/1153))
- Qwen 3.5 compact mode ([#1235](https://github.com/vllm-project/vllm-gaudi/pull/1235))
- Qwen 3.5 additional changes and fixes ([#1312](https://github.com/vllm-project/vllm-gaudi/pull/1312))

---

## Performance

- Creating custom depthwise conv1d kernel for MambaMixer2 ([#1092](https://github.com/vllm-project/vllm-gaudi/pull/1092))
- Adapt Online defragmenter for torch compile ([#986](https://github.com/vllm-project/vllm-gaudi/pull/986))
- Set reserved memory for Torch compile ([#1093](https://github.com/vllm-project/vllm-gaudi/pull/1093))
- Cap decode block bucket limit to reduce warmup time ([#1160](https://github.com/vllm-project/vllm-gaudi/pull/1160))
- Optimize selective_state_update ([#1295](https://github.com/vllm-project/vllm-gaudi/pull/1295))
- Optimizing visible block number for Hybrid kv_cache ([#1319](https://github.com/vllm-project/vllm-gaudi/pull/1319))

---

## Attention & KV Cache

- Mamba prefix caching support for v0.19.0 ([#1330](https://github.com/vllm-project/vllm-gaudi/pull/1330))
- Fix for proper KV cache slot addressing for Hybrid models ([#1323](https://github.com/vllm-project/vllm-gaudi/pull/1323))
- Resolving kv_cache access in HPUMambaMixer2 and reintroducing Granite4.0 in CI ([#1287](https://github.com/vllm-project/vllm-gaudi/pull/1287))
- Remove dead Unified Attention (UA) code ([#1226](https://github.com/vllm-project/vllm-gaudi/pull/1226))
- Fix KV cache memory regression from unconditional RowParallelLinear OOT registration ([#1146](https://github.com/vllm-project/vllm-gaudi/pull/1146))
- Exclude dummy block from NIXL KV cache registration ([#1140](https://github.com/vllm-project/vllm-gaudi/pull/1140))

---

## Quantization

- Load and Dequant MxFP4 Weights ([#1156](https://github.com/vllm-project/vllm-gaudi/pull/1156))
- Fix INC FP8 dynamic quantization for MoE models on HPU ([#1183](https://github.com/vllm-project/vllm-gaudi/pull/1183))
- Fix FP8 block-to-channel conversion breaking MLA weight loading ([#1220](https://github.com/vllm-project/vllm-gaudi/pull/1220))
- Fix INC/MLA alias-path quantization failures ([#1222](https://github.com/vllm-project/vllm-gaudi/pull/1222))
- Granite-4.0-h Calibration config ([#1221](https://github.com/vllm-project/vllm-gaudi/pull/1221))
- Fix Synapse GC compile failure for FP8-quantized models ([#1334](https://github.com/vllm-project/vllm-gaudi/pull/1334))

---

## Plugin Core

- Patch for LMCache ([#1176](https://github.com/vllm-project/vllm-gaudi/pull/1176))
- Remove aggregate module HpuDeepseekOCRVisual ([#1102](https://github.com/vllm-project/vllm-gaudi/pull/1102))
- Remove deprecated virtual_engine from ForwardContext ([#1187](https://github.com/vllm-project/vllm-gaudi/pull/1187))
- Fix CPUOffloadingSpec import path and remove obsolete roberta patch ([#1229](https://github.com/vllm-project/vllm-gaudi/pull/1229))
- Separate conv1d for Granite 4.0 (v0.17.1-style) ([#1320](https://github.com/vllm-project/vllm-gaudi/pull/1320))
- Add num_spec field to MambaMixer2 for upstream compatibility ([#1141](https://github.com/vllm-project/vllm-gaudi/pull/1141))
- Fix include all sub-packages in setuptools package discovery ([#1212](https://github.com/vllm-project/vllm-gaudi/pull/1212))

---

## Serving & Infrastructure

- Parameterize EXTRA_INDEX_URL in Dockerfiles ([#1131](https://github.com/vllm-project/vllm-gaudi/pull/1131))
- Add VLLM_REPO and VLLM_GAUDI_REPO args to RHEL UBI Dockerfile ([#1225](https://github.com/vllm-project/vllm-gaudi/pull/1225))
- Add real context length to the high-level profile ([#1169](https://github.com/vllm-project/vllm-gaudi/pull/1169))
- Add more than 2 models to sleep mode model swapping test ([#1100](https://github.com/vllm-project/vllm-gaudi/pull/1100))
- Add AI agents config files ([#1123](https://github.com/vllm-project/vllm-gaudi/pull/1123))
- Update quickstart guide and supported model list ([#1173](https://github.com/vllm-project/vllm-gaudi/pull/1173))

---

## Fixes

- Fix OOM crashes during high-concurrency inference ([#1124](https://github.com/vllm-project/vllm-gaudi/pull/1124))
- Fix multimodal prefill batching for 2D padded inputs ([#1126](https://github.com/vllm-project/vllm-gaudi/pull/1126))
- Fix M-RoPE position tensor shape for batched multimodal prefill (BS>1) ([#1216](https://github.com/vllm-project/vllm-gaudi/pull/1216))
- Fix preempted prompts and prefill/decoding splitting ([#830](https://github.com/vllm-project/vllm-gaudi/pull/830))
- Fix grammar bitmask corruption in mixed structured-output batches ([#1200](https://github.com/vllm-project/vllm-gaudi/pull/1200))
- Fix Qwen Out of HOST memory (OOM) ([#1247](https://github.com/vllm-project/vllm-gaudi/pull/1247))
- Fix SharedFusedMoE attribute error for Llama4 MoE layers ([#1172](https://github.com/vllm-project/vllm-gaudi/pull/1172))
- Fix false-positive cross-layer block detection for MLA in NIXL ([#1205](https://github.com/vllm-project/vllm-gaudi/pull/1205))
- Fix block size setting for granite 4.0h ([#1318](https://github.com/vllm-project/vllm-gaudi/pull/1318))
- Granite4.0h fallback bucket padding fix ([#1207](https://github.com/vllm-project/vllm-gaudi/pull/1207))
- Fix wrong AI Lab names in validated_models.md ([#1282](https://github.com/vllm-project/vllm-gaudi/pull/1282))
- Fix -u flag requiring argument in calibrate_model.sh ([#1121](https://github.com/vllm-project/vllm-gaudi/pull/1121))

---

## Security

- Coverity fix including security, null-like values, duplicates and typos ([#1164](https://github.com/vllm-project/vllm-gaudi/pull/1164))
- SDL secure error handling fixes ([#1245](https://github.com/vllm-project/vllm-gaudi/pull/1245))

---

## Deprecation & Breaking Changes

- Upgraded to **Intel® Gaudi® Software v1.24.0** and **PyTorch 2.10** — users must update their Gaudi software stack from v1.23.0 to v1.24.0.
- Removed dead Unified Attention (UA) code ([#1226](https://github.com/vllm-project/vllm-gaudi/pull/1226)).
- Removed aggregate module HpuDeepseekOCRVisual ([#1102](https://github.com/vllm-project/vllm-gaudi/pull/1102)).
- Removed deprecated `virtual_engine` from `ForwardContext` ([#1187](https://github.com/vllm-project/vllm-gaudi/pull/1187)).

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1334](https://github.com/vllm-project/vllm-gaudi/pull/1334) | Fix Synapse GC compile failure for FP8-quantized models | @jiminha |
| [#1330](https://github.com/vllm-project/vllm-gaudi/pull/1330) | Mamba prefix caching support for v0.19.0 | @jbyczkow |
| [#1323](https://github.com/vllm-project/vllm-gaudi/pull/1323) | Fix for proper KV cache slot addressing for Hybrid models | @ksmusz |
| [#1320](https://github.com/vllm-project/vllm-gaudi/pull/1320) | Separate conv1d for Granite 4.0 (v0.17.1-style) | @jbyczkow |
| [#1319](https://github.com/vllm-project/vllm-gaudi/pull/1319) | Optimizing visible block number for Hybrid kv_cache | @ksmusz |
| [#1318](https://github.com/vllm-project/vllm-gaudi/pull/1318) | Fix block size setting for granite 4.0h | @jkaniecki |
| [#1315](https://github.com/vllm-project/vllm-gaudi/pull/1315) | Granite-4.0-h Calibration config | @jbyczkow |
| [#1312](https://github.com/vllm-project/vllm-gaudi/pull/1312) | Changes for qwen35 | @shepark |
| [#1295](https://github.com/vllm-project/vllm-gaudi/pull/1295) | Optimize selective_state_update | @jbyczkow |
| [#1287](https://github.com/vllm-project/vllm-gaudi/pull/1287) | Resolving kv_cache access in HPUMambaMixer2 and reintroducing Granite4.0 in CI | @ksmusz |
| [#1286](https://github.com/vllm-project/vllm-gaudi/pull/1286) | Temporarily removing granite-4-h-small from CI | @ksmusz |
| [#1282](https://github.com/vllm-project/vllm-gaudi/pull/1282) | Fix wrong AI Lab names in validated_models.md | @MaxAmende |
| [#1279](https://github.com/vllm-project/vllm-gaudi/pull/1279) | Upstream vLLM compatibility fix | @iboiko-habana |
| [#1262](https://github.com/vllm-project/vllm-gaudi/pull/1262) | ci: fix EOF error when PR title contains apostrophe | @adobrzyn |
| [#1247](https://github.com/vllm-project/vllm-gaudi/pull/1247) | Fix of Qwen Out of HOST memory (OOM) | @iboiko-habana |
| [#1245](https://github.com/vllm-project/vllm-gaudi/pull/1245) | SDL secure error handling fixes | @adobrzyn |
| [#1235](https://github.com/vllm-project/vllm-gaudi/pull/1235) | qwen35 compact mode | @libinta |
| [#1229](https://github.com/vllm-project/vllm-gaudi/pull/1229) | Fix CPUOffloadingSpec import path and remove obsolete roberta patch | @pawel-olejniczak |
| [#1226](https://github.com/vllm-project/vllm-gaudi/pull/1226) | Remove dead Unified Attention (UA) code | @adobrzyn |
| [#1225](https://github.com/vllm-project/vllm-gaudi/pull/1225) | Add VLLM_REPO and VLLM_GAUDI_REPO args to RHEL UBI Dockerfile | @aung-san-i |
| [#1222](https://github.com/vllm-project/vllm-gaudi/pull/1222) | Fix INC/MLA alias-path quantization failures | @pawel-olejniczak |
| [#1221](https://github.com/vllm-project/vllm-gaudi/pull/1221) | Granite-4.0-h Calibration config | @mfylcek |
| [#1220](https://github.com/vllm-project/vllm-gaudi/pull/1220) | Fix FP8 block-to-channel conversion breaking MLA weight loading | @afierka-intel |
| [#1216](https://github.com/vllm-project/vllm-gaudi/pull/1216) | Fix M-RoPE position tensor shape for batched multimodal prefill (BS>1) | @afierka-intel |
| [#1214](https://github.com/vllm-project/vllm-gaudi/pull/1214) | Fix param mismatch for compute_nixl_compatibility_hash() | @hsubramony |
| [#1212](https://github.com/vllm-project/vllm-gaudi/pull/1212) | Fix include all sub-packages in setuptools package discovery | @Xaenalt |
| [#1207](https://github.com/vllm-project/vllm-gaudi/pull/1207) | Granite4.0h fallback bucket padding fix | @mfylcek |
| [#1205](https://github.com/vllm-project/vllm-gaudi/pull/1205) | Fix false-positive cross-layer block detection for MLA in NIXL | @iboiko-habana |
| [#1200](https://github.com/vllm-project/vllm-gaudi/pull/1200) | Fix grammar bitmask corruption in mixed structured-output batches | @jbyczkow |
| [#1194](https://github.com/vllm-project/vllm-gaudi/pull/1194) | Port: Fix SharedFusedMoE attribute error for Llama4 MoE layers | @adobrzyn |
| [#1187](https://github.com/vllm-project/vllm-gaudi/pull/1187) | Remove deprecated virtual_engine from ForwardContext | @iboiko-habana |
| [#1183](https://github.com/vllm-project/vllm-gaudi/pull/1183) | Fix INC FP8 dynamic quantization for MoE models on HPU | @yeonsily |
| [#1181](https://github.com/vllm-project/vllm-gaudi/pull/1181) | Disable nixl CI tests | @iboiko-habana |
| [#1176](https://github.com/vllm-project/vllm-gaudi/pull/1176) | Monkey patch for LMCache | @hlin99 |
| [#1174](https://github.com/vllm-project/vllm-gaudi/pull/1174) | Upstream vLLM hourly fix | @tzielinski-habana |
| [#1173](https://github.com/vllm-project/vllm-gaudi/pull/1173) | Update quickstart guide and supported model list | @PatrykWo |
| [#1172](https://github.com/vllm-project/vllm-gaudi/pull/1172) | Fix SharedFusedMoE attribute error for Llama4 MoE layers | @adobrzyn |
| [#1169](https://github.com/vllm-project/vllm-gaudi/pull/1169) | Add real context length to the high-level profile | @yangulei |
| [#1165](https://github.com/vllm-project/vllm-gaudi/pull/1165) | Reintroduce ci test for granite-4-h-small | @microslaw |
| [#1164](https://github.com/vllm-project/vllm-gaudi/pull/1164) | Coverity fix including security, null-like values, duplicates and typos | @adobrzyn |
| [#1160](https://github.com/vllm-project/vllm-gaudi/pull/1160) | Cap decode block bucket limit to reduce warmup time | @adobrzyn |
| [#1156](https://github.com/vllm-project/vllm-gaudi/pull/1156) | Load and Dequant MxFP4 Weights | @SKRohit |
| [#1153](https://github.com/vllm-project/vllm-gaudi/pull/1153) | qwen35 initial enablement | @libinta |
| [#1146](https://github.com/vllm-project/vllm-gaudi/pull/1146) | Fix KV cache memory regression from unconditional RowParallelLinear OOT registration | @kamil-kaczor |
| [#1141](https://github.com/vllm-project/vllm-gaudi/pull/1141) | Add num_spec field to MambaMixer2 for upstream compatibility | @jbyczkow |
| [#1140](https://github.com/vllm-project/vllm-gaudi/pull/1140) | Exclude dummy block from NIXL KV cache registration | @yeonsily |
| [#1136](https://github.com/vllm-project/vllm-gaudi/pull/1136) | PR-1054 revert | @jczaja |
| [#1135](https://github.com/vllm-project/vllm-gaudi/pull/1135) | Temporary nixl test cases disablement | @iboiko-habana |
| [#1131](https://github.com/vllm-project/vllm-gaudi/pull/1131) | Parameterize EXTRA_INDEX_URL | @PatrykWo |
| [#1129](https://github.com/vllm-project/vllm-gaudi/pull/1129) | Upstream vLLM compatibility fixes | @iboiko-habana |
| [#1126](https://github.com/vllm-project/vllm-gaudi/pull/1126) | Fix multimodal prefill batching for 2D padded inputs | @afierka-intel |
| [#1124](https://github.com/vllm-project/vllm-gaudi/pull/1124) | Fix OOM crashes during high-concurrency inference | @afierka-intel |
| [#1123](https://github.com/vllm-project/vllm-gaudi/pull/1123) | Add AI agents config files | @kamil-kaczor |
| [#1121](https://github.com/vllm-project/vllm-gaudi/pull/1121) | Fix -u flag requiring argument in calibrate_model.sh | @afierka-intel |
| [#1102](https://github.com/vllm-project/vllm-gaudi/pull/1102) | Remove aggregate module HpuDeepseekOCRVisual | @jwieczorekhabana |
| [#1100](https://github.com/vllm-project/vllm-gaudi/pull/1100) | Add more than 2 models to sleep mode model swapping test | @12010486 |
| [#1093](https://github.com/vllm-project/vllm-gaudi/pull/1093) | Set reserved mem for Torch compile | @nngokhale |
| [#1092](https://github.com/vllm-project/vllm-gaudi/pull/1092) | Creating custom depthwise conv1d kernel for MambaMixer2 | @ksmusz |
| [#986](https://github.com/vllm-project/vllm-gaudi/pull/986) | Adapt Online defragmenter for torch compile | @jwieczorekhabana |
| [#830](https://github.com/vllm-project/vllm-gaudi/pull/830) | Fix preempted prompts and prefill/decoding splitting | @yangulei |

---

## New Contributors

Welcome to the following first-time contributors to vLLM Gaudi Plugin!

- **@aung-san-i** — Add VLLM_REPO and VLLM_GAUDI_REPO args to RHEL UBI Dockerfile ([#1225](https://github.com/vllm-project/vllm-gaudi/pull/1225))
- **@MaxAmende** — Fix wrong AI Lab names in validated_models.md ([#1282](https://github.com/vllm-project/vllm-gaudi/pull/1282))
- **@Xaenalt** — Fix setuptools package discovery to include sub-packages ([#1212](https://github.com/vllm-project/vllm-gaudi/pull/1212))
- **@12010486** — Add more than 2 models to sleep mode model swapping test ([#1100](https://github.com/vllm-project/vllm-gaudi/pull/1100))
- **@hlin99** — Monkey patch for LMCache ([#1176](https://github.com/vllm-project/vllm-gaudi/pull/1176))
