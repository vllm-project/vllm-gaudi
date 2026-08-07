# vLLM Gaudi Plugin v0.26.0 Release Notes

## Overview

This release is based on [vLLM v0.26.0](https://github.com/vllm-project/vllm/releases/tag/v0.26.0) and supports [Intel® Gaudi® Software v1.24.1](https://docs.habana.ai/en/v1.24.1/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.11.

## Highlights

- Enabled the plugin on upstream [vLLM v0.26.0](https://github.com/vllm-project/vllm/releases/tag/v0.26.0), realigning the Intel® Gaudi® platform with extensive upstream API drift across MoE/MLA, quantization, serving, attention, KV-offload, the multi-model server, and the NIXL connector.
- Added new model support for MiniMax-M3 (vision-language MoE), the Gemma-4 family (E2B, 31B, 26B-A4B), the Kimi-K2.5 vision tower, and Qwen3-Coder-Next.
- Improved MXFP4 gpt-oss serving with SwiGLU-OAI and bias support, fixed INT8 W8A8 weight loading, and resolved multiple FP8/INC calibration OOM and graph-compile regressions.
- Reduced graph compilation and warmup overhead with configurable multimodal warmup resolutions, Gemma-4 compile-time fixes, and a fix for a decode throughput regression from upstream vLLM.
- Strengthened security by resolving Coverity-reported issues and adding a security guidance page that links to upstream vLLM.
- Restored the latest `transformers` version by removing the temporary `transformers==5.9.0` pin introduced in v0.24.0.

---

## New Model Support

- Added MiniMax-M3 vision-language MoE model support for Intel Gaudi (HPU). ([#1629](https://github.com/vllm-project/vllm-gaudi/pull/1629))
- Enabled the Gemma-4 models (E2B, 31B, 26B-A4B) on HPU. ([#1619](https://github.com/vllm-project/vllm-gaudi/pull/1619))
- Enabled the Kimi-K2.5 vision tower on HPU with real-valued 2D-RoPE and eager rope. ([#1617](https://github.com/vllm-project/vllm-gaudi/pull/1617))

---

## Performance

- Fixed a decode throughput regression introduced by upstream vLLM (#42656). ([#1631](https://github.com/vllm-project/vllm-gaudi/pull/1631))
- Improved MiniMax-M3 MoE decode performance after the upstream merge. ([#1656](https://github.com/vllm-project/vllm-gaudi/pull/1656))
- Added an item-count axis to multimodal warmup resolutions. ([#1665](https://github.com/vllm-project/vllm-gaudi/pull/1665))
- Enabled multimodal model warmup at specified resolutions on HPU. ([#1652](https://github.com/vllm-project/vllm-gaudi/pull/1652))
- Fixed Gemma-4 E-series PLE compile time on HPU with dynamic indexing. ([#1654](https://github.com/vllm-project/vllm-gaudi/pull/1654))
- Fixed Qwen3.5 MoE post-warmup graph recompilation. ([#1602](https://github.com/vllm-project/vllm-gaudi/pull/1602))
- Fixed decode bucket sparsity for long prompts. ([#1552](https://github.com/vllm-project/vllm-gaudi/pull/1552))
- Removed the custom conv1d update TPC kernel in favor of the native path. ([#1600](https://github.com/vllm-project/vllm-gaudi/pull/1600))

---

## Attention and KV Cache

- Fixed contiguous PA wrong-rows on prefill context (perf-preserving). ([#1546](https://github.com/vllm-project/vllm-gaudi/pull/1546))
- Optimized Gemma-4 sliding-window and Transformers SDPA attention. ([#1662](https://github.com/vllm-project/vllm-gaudi/pull/1662))
- Restored the HPU NIXL K/V split and heterogeneous connector attributes after upstream vLLM changes. ([#1653](https://github.com/vllm-project/vllm-gaudi/pull/1653))
- Skipped GPU-only blocks-first KV assertions in the HPU NIXL `TransferTopology`. ([#1616](https://github.com/vllm-project/vllm-gaudi/pull/1616))

---

## Quantization

- Added MXFP4 gpt-oss SwiGLU-OAI and bias serving support. ([#1680](https://github.com/vllm-project/vllm-gaudi/pull/1680))
- Fixed INT8 W8A8 weight loading by not wrapping the loader with the FP8-only `gaudi_weight_wrapper`. ([#1668](https://github.com/vllm-project/vllm-gaudi/pull/1668))
- Freed dead INC-quantized FP8 MoE weight copies to halve device memory after the upstream MoE refactor. ([#1580](https://github.com/vllm-project/vllm-gaudi/pull/1580))
- Fixed the Maverick FP8 INC calibration OOM after the upstream MoE refactor. ([#1594](https://github.com/vllm-project/vllm-gaudi/pull/1594))
- Fixed the DeepSeek-R1 FP8 grouped-topk MoE graph-compile failure. ([#1593](https://github.com/vllm-project/vllm-gaudi/pull/1593))
- Fixed a dense-model INC quantization OOM regression. ([#1626](https://github.com/vllm-project/vllm-gaudi/pull/1626))
- Fixed an FP8 dequant device mismatch. ([#1598](https://github.com/vllm-project/vllm-gaudi/pull/1598))
- Warned when `calibrate_model.sh -u` is used on TP>1 without `-r`. ([#1610](https://github.com/vllm-project/vllm-gaudi/pull/1610))

---

## Plugin Core

- Resolved 10 upstream vLLM API-drift breaks across MoE/MLA, quantization, server, and attention. ([#1611](https://github.com/vllm-project/vllm-gaudi/pull/1611))
- Realigned the HPU plugin with upstream vLLM API drift (KV-offload `OffloadingWorker`, `ServingRender`, GraniteMoeHybrid layer types, FusedMoE `shared_expert_weight`) and fixed a DP teardown hang. ([#1562](https://github.com/vllm-project/vllm-gaudi/pull/1562))
- Adapted the multi-model server, HPU LoRA/NIXL, and offloading-stats tests to upstream vLLM drift. ([#1558](https://github.com/vllm-project/vllm-gaudi/pull/1558))
- Adapted HPU KV-offload and MLA to upstream vLLM (#48150/#48251) API drift. ([#1621](https://github.com/vllm-project/vllm-gaudi/pull/1621))
- Fixed the `_mamba_block_aligned_split` signature mismatch with the pinned vLLM on v0.26.0. ([#1677](https://github.com/vllm-project/vllm-gaudi/pull/1677))
- Ported heterogeneous-deployment fixes for vLLM v0.24.0 to main. ([#1606](https://github.com/vllm-project/vllm-gaudi/pull/1606))
- Made granite-4.0-h-small compatible with single-process model swap. ([#1671](https://github.com/vllm-project/vllm-gaudi/pull/1671))
- Paired SP-MoE dispatch/combine at `dp_size==1` for Qwen3-30B EP (BF16 and FP8/compressed-tensors). ([#1618](https://github.com/vllm-project/vllm-gaudi/pull/1618))
- Detached the shared MoE gate when experts is the `MoERunner`. ([#1566](https://github.com/vllm-project/vllm-gaudi/pull/1566))
- Added `is_gemma4` to the heterogeneous layers check. ([#1663](https://github.com/vllm-project/vllm-gaudi/pull/1663))
- Kept the customization for GPTBigCode and Starcoder2 after upstream #30966. ([#1625](https://github.com/vllm-project/vllm-gaudi/pull/1625))
- Cherry-picked v0.24.0 fixes to main. ([#1615](https://github.com/vllm-project/vllm-gaudi/pull/1615))

---

## Serving and Infrastructure

- Added a pure-Python MiniMax-M3 tool-call parser for HPU. ([#1655](https://github.com/vllm-project/vllm-gaudi/pull/1655))
- Ported NIXL disaggregation accuracy test-script fixes to v0.26.0. ([#1647](https://github.com/vllm-project/vllm-gaudi/pull/1647))

---

## Fixes

- Fixed Kimi-K2.5/K2.6 multimodal warmup corruption and a dummy-input crash. ([#1651](https://github.com/vllm-project/vllm-gaudi/pull/1651))
- Fixed GraniteMoeHybrid GDN misdetection from the `transformers>=5` layer-type remap. ([#1624](https://github.com/vllm-project/vllm-gaudi/pull/1624))
- Fixed `sequence_parallel_chunk`. ([#1628](https://github.com/vllm-project/vllm-gaudi/pull/1628))
- Fixed granite-4.0-h-small tool-calling accuracy. ([#1553](https://github.com/vllm-project/vllm-gaudi/pull/1553))
- Removed the granite-4.0-h-small OOM workaround. ([#1575](https://github.com/vllm-project/vllm-gaudi/pull/1575))
- Enabled the HPU residual fix for `qwen3_next` (Qwen3-Coder-Next). ([#1676](https://github.com/vllm-project/vllm-gaudi/pull/1676))

---

## Security

- Fixed Coverity-reported dead code and null-residual issues. ([#1679](https://github.com/vllm-project/vllm-gaudi/pull/1679))
- Added a security guidance page that links to upstream vLLM. ([#1620](https://github.com/vllm-project/vllm-gaudi/pull/1620))

---

## Deprecation and Breaking Changes

- Restored the latest `transformers` version, removing the temporary `transformers==5.9.0` pin introduced in v0.24.0. ([#1560](https://github.com/vllm-project/vllm-gaudi/pull/1560))

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1680](https://github.com/vllm-project/vllm-gaudi/pull/1680) | Port of #1567: MXFP4 GPT-OSS SwiGLU-OAI + bias serving (releases/v0.26.0) | @adobrzyn |
| [#1676](https://github.com/vllm-project/vllm-gaudi/pull/1676) | Enable HPU residual fix for qwen3_next (Qwen3-Coder-Next) | @libinta |
| [#1679](https://github.com/vllm-project/vllm-gaudi/pull/1679) | Fix Coverity-reported dead code and null residual issues | @iboiko-habana |
| [#1677](https://github.com/vllm-project/vllm-gaudi/pull/1677) | Fix _mamba_block_aligned_split signature mismatch with pinned vLLM on v0.26.0 | @adobrzyn |
| [#1665](https://github.com/vllm-project/vllm-gaudi/pull/1665) | Add item-count axis to multimodal warmup resolutions | @libinta |
| [#1671](https://github.com/vllm-project/vllm-gaudi/pull/1671) | Port of: granite-4.0-h-small made compatible with single process models swap - #1595 | @iboiko-habana |
| [#1668](https://github.com/vllm-project/vllm-gaudi/pull/1668) | Fix #1612: don't wrap INT8 W8A8 loader with FP8-only gaudi_weight_wrapper | @rsmyrek |
| [#1664](https://github.com/vllm-project/vllm-gaudi/pull/1664) | Enable GLM-5.2 (DSA MoE) on HPU via dense-MLA fallback | @sureshnam |
| [#1655](https://github.com/vllm-project/vllm-gaudi/pull/1655) | Add pure-Python MiniMax-M3 tool-call parser for HPU | @mkrze |
| [#1656](https://github.com/vllm-project/vllm-gaudi/pull/1656) | Improve MiniMax-M3 MoE decode performance after upstream merge | @mkrze |
| [#1662](https://github.com/vllm-project/vllm-gaudi/pull/1662) | Gemma4:optimization for sliding window and transformers sdpa attn | @libinta |
| [#1663](https://github.com/vllm-project/vllm-gaudi/pull/1663) | Add is_gemma4 to heterogeneous layers check | @jiminha |
| [#1652](https://github.com/vllm-project/vllm-gaudi/pull/1652) | Enable mm model warmup at specified resolutions on HPU | @libinta |
| [#1651](https://github.com/vllm-project/vllm-gaudi/pull/1651) | Fix Kimi-K2.5/K2.6 multimodal warmup corruption and dummy-input crash | @yeonsily |
| [#1647](https://github.com/vllm-project/vllm-gaudi/pull/1647) | NIXL examples: port disagg accuracy test script fixes to v0.26.0 | @skaulintel |
| [#1654](https://github.com/vllm-project/vllm-gaudi/pull/1654) | Fix Gemma4 E-series PLE compile time on HPU with dynamic indexing (0.26) | @jiminha |
| [#1653](https://github.com/vllm-project/vllm-gaudi/pull/1653) | fix: restore HPU NIXL K/V split and hetero connector attrs after vLLM… | @skaulintel |
| [#1602](https://github.com/vllm-project/vllm-gaudi/pull/1602) | Qwen3.5 MOE: Fix post-warmup graph recompilation | @jiminha |
| [#1631](https://github.com/vllm-project/vllm-gaudi/pull/1631) | Port: [v0.24.0] Fix decode throughput regression from vLLM #42656 #1609 | @iboiko-habana |
| [#1628](https://github.com/vllm-project/vllm-gaudi/pull/1628) | Fix sequence_parallel_chunk | @Chris-Sigopt |
| [#1575](https://github.com/vllm-project/vllm-gaudi/pull/1575) | granite-4.0-h-small - remove OOM workaround | @rsmyrek |
| [#1629](https://github.com/vllm-project/vllm-gaudi/pull/1629) | Add MiniMax-M3 vision-language MoE model support for Intel Gaudi (HPU) | @mkrze |
| [#1620](https://github.com/vllm-project/vllm-gaudi/pull/1620) | docs: add security guidance page linking to upstream vLLM | @adobrzyn |
| [#1619](https://github.com/vllm-project/vllm-gaudi/pull/1619) | Enable Gemma-4 models (E2B, 31B, 26B-A4B) on HPU | @jiminha |
| [#1617](https://github.com/vllm-project/vllm-gaudi/pull/1617) | Enable Kimi-K2.5 vision tower on HPU (real-valued 2D-RoPE + eager rop… | @yeonsily |
| [#1625](https://github.com/vllm-project/vllm-gaudi/pull/1625) | Keep customization for GPTBigCode and Starcoder2, after #30966 | @iboiko-habana |
| [#1624](https://github.com/vllm-project/vllm-gaudi/pull/1624) | Fix GraniteMoeHybrid GDN misdetection from transformers>=5 layer-type remap | @rsmyrek |
| [#1626](https://github.com/vllm-project/vllm-gaudi/pull/1626) | Port of: [v0.24.0] Fix dense-model INC quant OOM regression from #1590 #1605 | @iboiko-habana |
| [#1621](https://github.com/vllm-project/vllm-gaudi/pull/1621) | fix: adapt HPU KV-offload + MLA to vLLM #48150/#48251 API drift | @pawel-olejniczak |
| [#1593](https://github.com/vllm-project/vllm-gaudi/pull/1593) | Port: Fix DeepSeek-R1 FP8 grouped-topk MoE graph compile failure #1586 | @iboiko-habana |
| [#1606](https://github.com/vllm-project/vllm-gaudi/pull/1606) | Port hetero fixes for vllm v0.24.0 to main | @hsubramony |
| [#1598](https://github.com/vllm-project/vllm-gaudi/pull/1598) | Fix FP8 dequant device mismatch | @mkrze |
| [#1610](https://github.com/vllm-project/vllm-gaudi/pull/1610) | Port of #1607: Warn when calibrate_model.sh -u is used on TP>1 without -r | @adobrzyn |
| [#1618](https://github.com/vllm-project/vllm-gaudi/pull/1618) | fix: pair SP-MoE dispatch/combine at dp_size==1 for Qwen3-30B EP (BF16 + FP8/compressed-tensors) | @pawel-olejniczak |
| [#1594](https://github.com/vllm-project/vllm-gaudi/pull/1594) | Port: Fix Maverick FP8 INC calibration OOM after #41184 MoE refactor #1590 | @iboiko-habana |
| [#1615](https://github.com/vllm-project/vllm-gaudi/pull/1615) | Cherry pick from v0.24.0 to main | @PatrykWo |
| [#1616](https://github.com/vllm-project/vllm-gaudi/pull/1616) | fix: skip GPU-only blocks-first KV asserts in HPU NIXL TransferTopology | @pawel-olejniczak |
| [#1611](https://github.com/vllm-project/vllm-gaudi/pull/1611) | Resolve 10 vLLM API-drift breaks (MoE/MLA/quant/server/attn) | @pawel-olejniczak |
| [#1600](https://github.com/vllm-project/vllm-gaudi/pull/1600) | remove conv1d update tpc kernel | @slokesha |
| [#1562](https://github.com/vllm-project/vllm-gaudi/pull/1562) | fix: realign HPU plugin with upstream vLLM API drift (kv-offload OffloadingWorker + ServingRender + GraniteMoeHybrid layer types + FusedMoE shared_expert_weight) and fix DP teardown hang | @pawel-olejniczak |
| [#1552](https://github.com/vllm-project/vllm-gaudi/pull/1552) | Fix decode bucket sparsity for long prompt | @shepark |
| [#1580](https://github.com/vllm-project/vllm-gaudi/pull/1580) | Free dead INC-quantized FP8 MoE weight copy to halve device memory, fix for MoE refactor #41184 | @iboiko-habana |
| [#1566](https://github.com/vllm-project/vllm-gaudi/pull/1566) | Detach shared MoE gate when experts is the MoERunner (vLLM #41184) | @iboiko-habana |
| [#1546](https://github.com/vllm-project/vllm-gaudi/pull/1546) | [HPU] Fix contiguous PA wrong-rows on prefill context (perf-preserving) | @adobrzyn |
| [#1553](https://github.com/vllm-project/vllm-gaudi/pull/1553) | granite-4.0-h-small - toolcalling accuracy fix | @rsmyrek |
| [#1560](https://github.com/vllm-project/vllm-gaudi/pull/1560) | Back the latest transformers version | @iboiko-habana |
| [#1558](https://github.com/vllm-project/vllm-gaudi/pull/1558) | Adapt multi-model server, HPU LoRA/NIXL, and offloading-stats tests to upstream vLLM drift | @pawel-olejniczak |

## New Contributors

Welcome to the following first-time contributors to vLLM Gaudi Plugin!

- **@Chris-Sigopt** — Fix `sequence_parallel_chunk` ([#1628](https://github.com/vllm-project/vllm-gaudi/pull/1628))
- **@sureshnam** — Enable GLM-5.2 (DSA MoE) on HPU via dense-MLA fallback ([#1664](https://github.com/vllm-project/vllm-gaudi/pull/1664))
