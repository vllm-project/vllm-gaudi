# vLLM Gaudi Plugin v0.19.0 Release Notes

## Overview

This release is based on [vLLM v0.19.0](https://github.com/vllm-project/vllm/releases/tag/v0.19.0) and supports [Intel® Gaudi® Software v1.24.0](https://docs.habana.ai/en/v1.24.0/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.10.

---

## Highlights

- Upgraded platform compatibility to **Intel® Gaudi® Software v1.24.0** and **PyTorch 2.10**.
- Introduced **Qwen 3.5** model support with compact mode for improved memory utilization.
- Introduced **Mamba prefix caching** support for hybrid SSM-Transformer models on v0.19.0.
- Added **MxFP4 weight loading and dequantization** for next-generation quantization formats.
- Added a **BF16 fallback path** for compressed-tensors INT8 (W8A8) weights on HPU (`HPUCompressedTensorsW8A8Int8_BF16Fallback`).
- Integrated **LMCache** support via monkey-patching for external cache backends.
- Introduced **custom depthwise conv1d TPC kernel** for MambaMixer2 to improve hybrid model performance.
- Adapted the **online defragmenter for torch.compile** mode, enabling memory defragmentation in compiled execution and automatically enabling it when contiguous PA is on.
- Added **single-process model swap** support, exposing the OpenAI-compatible `/v1/models/switch` endpoint for in-process model switching without server restarts.
- Stabilized **long-context decode** by bounding decode `block_list` growth and refining bucketing for non-power-of-two `block_size` (e.g. Granite hybrid models), significantly reducing recompilations and improving TPOT.
- Switched the **default `PT_HPU_LAZY_MODE`** in shipped Docker images to `0` (torch.compile) for both PyTorch upstream and fork builds.

---

## New Model Support

- Added initial Qwen 3.5 model support. ([#1153](https://github.com/vllm-project/vllm-gaudi/pull/1153))
- Added Qwen 3.5 compact mode. ([#1235](https://github.com/vllm-project/vllm-gaudi/pull/1235))
- Added Qwen 3.5 additional changes and fixes. ([#1312](https://github.com/vllm-project/vllm-gaudi/pull/1312))

---

## Performance

- Created a custom depthwise conv1d kernel for MambaMixer2. ([#1092](https://github.com/vllm-project/vllm-gaudi/pull/1092))
- Adapted the online defragmenter for torch.compile. ([#986](https://github.com/vllm-project/vllm-gaudi/pull/986))
- Enabled the online defragmenter automatically when contiguous PA is enabled. ([#1402](https://github.com/vllm-project/vllm-gaudi/pull/1402))
- Set reserved memory for torch.compile. ([#1093](https://github.com/vllm-project/vllm-gaudi/pull/1093))
- Optimized `selective_state_update`. ([#1295](https://github.com/vllm-project/vllm-gaudi/pull/1295))
- Optimized the visible block number for hybrid KV cache. ([#1319](https://github.com/vllm-project/vllm-gaudi/pull/1319))
- Stabilized decode `block_list` growth for long-context workloads to avoid HPU graph recompilation storms and OOM (significant TPOT/throughput gains on 200K-context runs). ([#1376](https://github.com/vllm-project/vllm-gaudi/pull/1376))
- Refined prompt bucket filtering, fallback bucket capping, and `mamba_decode_corrector` for non-power-of-two `block_size` long-context cases (e.g. Granite hybrid models). ([#1389](https://github.com/vllm-project/vllm-gaudi/pull/1389))
- Moved multimodal graph warmup under the `PT_COMPILE_ONLY_MODE` context (torch.compile path only) and reused the processor across resolution buckets to reduce warmup time. ([#1368](https://github.com/vllm-project/vllm-gaudi/pull/1368))

---

## Attention & KV Cache

- Added Mamba prefix caching support for v0.19.0. ([#1330](https://github.com/vllm-project/vllm-gaudi/pull/1330))
- Fixed proper KV cache slot addressing for hybrid models. ([#1323](https://github.com/vllm-project/vllm-gaudi/pull/1323))
- Resolved KV cache access in HPUMambaMixer2 and reintroduced Granite4.0 in CI. ([#1287](https://github.com/vllm-project/vllm-gaudi/pull/1287))
- Removed dead Unified Attention (UA) code. ([#1226](https://github.com/vllm-project/vllm-gaudi/pull/1226))
- Fixed a KV cache memory regression caused by unconditional RowParallelLinear OOT registration. ([#1146](https://github.com/vllm-project/vllm-gaudi/pull/1146))
- Excluded dummy block from NIXL KV cache registration. ([#1140](https://github.com/vllm-project/vllm-gaudi/pull/1140))

---

## Quantization

- Loaded and dequantized MxFP4 weights. ([#1156](https://github.com/vllm-project/vllm-gaudi/pull/1156))
- Added `HPUCompressedTensorsW8A8Int8_BF16Fallback` implementation for INT8 weights. ([#1394](https://github.com/vllm-project/vllm-gaudi/pull/1394))
- Fixed INC FP8 dynamic quantization for MoE models on HPU. ([#1183](https://github.com/vllm-project/vllm-gaudi/pull/1183))
- Fixed FP8 block-to-channel conversion breaking MLA weight loading. ([#1220](https://github.com/vllm-project/vllm-gaudi/pull/1220))
- Fixed INC/MLA alias-path quantization failures. ([#1222](https://github.com/vllm-project/vllm-gaudi/pull/1222))
- Added Granite-4.0-h calibration config. ([#1221](https://github.com/vllm-project/vllm-gaudi/pull/1221))
- Fixed Synapse GC compile failure for FP8-quantized models. ([#1334](https://github.com/vllm-project/vllm-gaudi/pull/1334))
- Renamed FP8 blockwise compressed-tensors scales to match HPU ops, fixing a regression in Mistral-Large-3-675B (cherry-pick of [#1304](https://github.com/vllm-project/vllm-gaudi/pull/1304)). ([#1374](https://github.com/vllm-project/vllm-gaudi/pull/1374))

---

## Plugin Core

- Added a patch for LMCache. ([#1176](https://github.com/vllm-project/vllm-gaudi/pull/1176))
- Added single-process model swap support exposing the OpenAI-compatible `/v1/models/switch` endpoint, allowing sequential serving of multiple small models within a single API server process without restarts (cherry-pick of [#1258](https://github.com/vllm-project/vllm-gaudi/pull/1258)). ([#1367](https://github.com/vllm-project/vllm-gaudi/pull/1367))
- Updated the HPU hetero NIXL connector class to track the new vLLM 0.19.0 NIXL interface. ([#1373](https://github.com/vllm-project/vllm-gaudi/pull/1373))
- Removed aggregate module HpuDeepseekOCRVisual. ([#1102](https://github.com/vllm-project/vllm-gaudi/pull/1102))
- Removed deprecated `virtual_engine` from `ForwardContext`. ([#1187](https://github.com/vllm-project/vllm-gaudi/pull/1187))
- Fixed CPUOffloadingSpec import path and removed obsolete roberta patch. ([#1229](https://github.com/vllm-project/vllm-gaudi/pull/1229))
- Separated conv1d for Granite 4.0 (v0.17.1-style). ([#1320](https://github.com/vllm-project/vllm-gaudi/pull/1320))
- Added the `num_spec` field to MambaMixer2 for upstream compatibility. ([#1141](https://github.com/vllm-project/vllm-gaudi/pull/1141))
- Fixed setuptools package discovery to include all sub-packages. ([#1212](https://github.com/vllm-project/vllm-gaudi/pull/1212))

---

## Serving & Infrastructure

- Parameterized EXTRA_INDEX_URL in Dockerfiles. ([#1131](https://github.com/vllm-project/vllm-gaudi/pull/1131))
- Added VLLM_REPO and VLLM_GAUDI_REPO arguments to RHEL UBI Dockerfile. ([#1225](https://github.com/vllm-project/vllm-gaudi/pull/1225))
- Set `PT_HPU_LAZY_MODE=0` (torch.compile) as the default in shipped Docker images for both PyTorch upstream and fork builds, with accompanying documentation updates. ([#1397](https://github.com/vllm-project/vllm-gaudi/pull/1397))
- Added real context length to the high-level profile. ([#1169](https://github.com/vllm-project/vllm-gaudi/pull/1169))
- Added more than 2 models to the sleep mode model swapping test. ([#1100](https://github.com/vllm-project/vllm-gaudi/pull/1100))
- Added AI agents config files. ([#1123](https://github.com/vllm-project/vllm-gaudi/pull/1123))
- Updated the quickstart guide and supported model list. ([#1173](https://github.com/vllm-project/vllm-gaudi/pull/1173))

---

## Fixes

- Fixed OOM crashes during high-concurrency inference. ([#1124](https://github.com/vllm-project/vllm-gaudi/pull/1124))
- Fixed multimodal prefill batching for 2D padded inputs. ([#1126](https://github.com/vllm-project/vllm-gaudi/pull/1126))
- Fixed M-RoPE position tensor shape for batched multimodal prefill (BS>1). ([#1216](https://github.com/vllm-project/vllm-gaudi/pull/1216))
- Fixed preempted prompts and prefill/decoding splitting. ([#830](https://github.com/vllm-project/vllm-gaudi/pull/830))
- Fixed grammar bitmask corruption in mixed structured-output batches. ([#1200](https://github.com/vllm-project/vllm-gaudi/pull/1200))
- Fixed Qwen out of host memory (OOM) errors. ([#1247](https://github.com/vllm-project/vllm-gaudi/pull/1247))
- Fixed a `SharedFusedMoE` attribute error for Llama4 MoE layers. ([#1172](https://github.com/vllm-project/vllm-gaudi/pull/1172))
- Fixed false-positive cross-layer block detection for MLA in NIXL. ([#1205](https://github.com/vllm-project/vllm-gaudi/pull/1205))
- Fixed block size setting for Granite 4.0h. ([#1318](https://github.com/vllm-project/vllm-gaudi/pull/1318))
- Fixed Granite4.0h fallback bucket padding. ([#1207](https://github.com/vllm-project/vllm-gaudi/pull/1207))
- Fixed wrong AI Lab names in validated_models.md. ([#1282](https://github.com/vllm-project/vllm-gaudi/pull/1282))
- Fixed the `-u` flag requiring an argument in calibrate_model.sh. ([#1121](https://github.com/vllm-project/vllm-gaudi/pull/1121))
- Flattened 3D `inputs_embeds` in `HpuModelAdapter.forward` to fix a shape mismatch on upstream VL models (e.g. Qwen3-VL-MoE deepstack) when using 2D padded prefill batches. ([#1380](https://github.com/vllm-project/vllm-gaudi/pull/1380))

---

## Security

- Fixed coverity issues, including security, null-like values, duplicates, and typos. ([#1164](https://github.com/vllm-project/vllm-gaudi/pull/1164))
- Fixed SDL secure error handling issues. ([#1245](https://github.com/vllm-project/vllm-gaudi/pull/1245))

---

## Deprecation & Breaking Changes

- Upgraded to **Intel® Gaudi® Software v1.24.0** and **PyTorch 2.10**, which requires users to update their Intel Gaudi software stack from v1.23.0 to v1.24.0.
- Changed the default `PT_HPU_LAZY_MODE` in shipped Docker images to `0` (torch.compile) for both PyTorch upstream and fork builds. Lazy mode users must opt back in explicitly. ([#1397](https://github.com/vllm-project/vllm-gaudi/pull/1397))
- Reverted the v0.19.0 backport of "Cap decode block bucket limit to reduce warmup time" ([#1160](https://github.com/vllm-project/vllm-gaudi/pull/1160)) on the `releases/v0.19.0` branch; long-context decode is now stabilized via [#1376](https://github.com/vllm-project/vllm-gaudi/pull/1376) and [#1389](https://github.com/vllm-project/vllm-gaudi/pull/1389) instead. ([#1388](https://github.com/vllm-project/vllm-gaudi/pull/1388))
- Removed unused Unified Attention (UA) code. ([#1226](https://github.com/vllm-project/vllm-gaudi/pull/1226)).
- Removed the aggregate module `HpuDeepseekOCRVisual`. ([#1102](https://github.com/vllm-project/vllm-gaudi/pull/1102)).
- Removed deprecated `virtual_engine` from `ForwardContext`. ([#1187](https://github.com/vllm-project/vllm-gaudi/pull/1187)).

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1402](https://github.com/vllm-project/vllm-gaudi/pull/1402) | To enable defrag if contig_pa is enabled | @iboiko-habana |
| [#1397](https://github.com/vllm-project/vllm-gaudi/pull/1397) | Cherry-pick to v0.19.0 Set Docker auto calc PT_HPU_LAZY_MODE=0 as default | @nngokhale |
| [#1394](https://github.com/vllm-project/vllm-gaudi/pull/1394) | HPUCompressedTensorsW8A8Int8_BF16Fallback impl | @jbyczkow |
| [#1389](https://github.com/vllm-project/vllm-gaudi/pull/1389) | Bucketing edge cases finetune for longer ctx (#1362) | @ksmusz |
| [#1388](https://github.com/vllm-project/vllm-gaudi/pull/1388) | [v0.19.0] Revert "Cap decode block bucket limit to reduce warmup time (#1160)" | @adobrzyn |
| [#1380](https://github.com/vllm-project/vllm-gaudi/pull/1380) | Flatten 3D inputs_embeds in HpuModelAdapter.forward | @shepark |
| [#1376](https://github.com/vllm-project/vllm-gaudi/pull/1376) | Stabilize decode block_list growth for long-context workloads (v0.19.0) | @adobrzyn |
| [#1374](https://github.com/vllm-project/vllm-gaudi/pull/1374) | Cherry-pick: Updated fix regression in Mistral-Large-3-675B (#1304) for v0.19.0 | @skavulya |
| [#1373](https://github.com/vllm-project/vllm-gaudi/pull/1373) | v0.19.0 interface fixes for hetero nixl connector | @sandeep-maddipatla |
| [#1368](https://github.com/vllm-project/vllm-gaudi/pull/1368) | Move mm graph warmup under pt compile only context | @shepark |
| [#1367](https://github.com/vllm-project/vllm-gaudi/pull/1367) | Cherry-pick from PR#1258 (single-process model swap) | @12010486 |
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
- **@sandeep-maddipatla** — v0.19.0 interface fixes for hetero nixl connector ([#1373](https://github.com/vllm-project/vllm-gaudi/pull/1373))
