# vLLM Gaudi Plugin v0.24.0 Release Notes

## Overview

This release is based on [vLLM v0.24.0](https://github.com/vllm-project/vllm/releases/tag/v0.24.0) and supports [Intel® Gaudi® Software v1.24.1](https://docs.habana.ai/en/v1.24.1/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.11.

## Highlights

- Upgraded platform compatibility to [Intel® Gaudi® Software v1.24.1](https://docs.habana.ai/en/v1.24.1/Release_Notes/GAUDI_Release_Notes.html) and [PyTorch 2.11](https://github.com/pytorch/pytorch/releases#release-v2.11.0).
- Enabled upstream [vLLM v0.24.0](https://github.com/vllm-project/vllm/releases#release-v0.24.0) by adapting the HPU platform to the FusedMoE/MoERunner inversion ([vLLM #41184](https://github.com/vllm-project/vllm/pull/41184)), HPU scheduler and ngram-proposer updates, KV-connector and offloading-connector refactors, `ServingTokenization`, and the Mamba/GDN rewrite.
- Added [Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next) model support by registering `Qwen3NextForCausalLM` in the Mamba-like architecture set and adding `Qwen3.6-35B` CI coverage.
- Improved FP8/INC quantization memory efficiency by freeing dead INC-quantized FP8 MoE weight copies to roughly halve device memory and resolving multiple FP8/INC calibration OOM and graph-compile regressions.
- Switched hybrid/Mamba decode to TPC-native `causal_conv1d` operations, removing the custom conv1d update kernel.
- Extended single-card model swapping to hybrid SSM-Transformer models, including [granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small).
- Strengthened project security by adding a [SECURITY.md](https://github.com/vllm-project/vllm-gaudi/blob/main/SECURITY.md) vulnerability disclosure policy, removing the outdated and insecure `decord` dependency, and isolating `HF_TOKEN` CI jobs behind an approved-workflow environment.

---

## New Model Support

- Validated [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) and [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (BF16, single-card Gaudi 3) with GSM8K accuracy coverage. ([#1554](https://github.com/vllm-project/vllm-gaudi/pull/1554), [#1443](https://github.com/vllm-project/vllm-gaudi/pull/1443), [#1434](https://github.com/vllm-project/vllm-gaudi/pull/1434))
- Added `Qwen3-Next` architecture support via `Qwen3NextForCausalLM` in the Mamba-like architecture set. ([#1450](https://github.com/vllm-project/vllm-gaudi/pull/1450))
- Added hybrid-model support for single-card model swapping. ([#1415](https://github.com/vllm-project/vllm-gaudi/pull/1415))

---

## Performance

- Switched to TPC-native operations for the `causal_conv1d` update. ([#1527](https://github.com/vllm-project/vllm-gaudi/pull/1527))
- Removed the custom conv1d update TPC kernel in favor of the native path. ([#1585](https://github.com/vllm-project/vllm-gaudi/pull/1585))
- Fixed a decode throughput regression introduced by upstream vLLM. ([#1609](https://github.com/vllm-project/vllm-gaudi/pull/1609))
- Fixed decode bucket sparsity for long prompts. ([#1584](https://github.com/vllm-project/vllm-gaudi/pull/1584))
- Replaced `Sampler.gather_logprobs` to remove `mark_unbacked` calls. ([#1548](https://github.com/vllm-project/vllm-gaudi/pull/1548))
- Reduced graph (re)compilation warnings during single-process model swap. ([#1478](https://github.com/vllm-project/vllm-gaudi/pull/1478))
- Removed the `num_ctx_tokens_less_or_equal_batched_max_model_len` filter. ([#1454](https://github.com/vllm-project/vllm-gaudi/pull/1454))
- Fixed decode bucket filter issues. ([#1447](https://github.com/vllm-project/vllm-gaudi/pull/1447))

---

## Attention and KV Cache

- Fixed contiguous PA wrong-rows on prefill context (perf-preserving). ([#1546](https://github.com/vllm-project/vllm-gaudi/pull/1546))
- Fixed hetero HPU NIXL connector compatibility with upstream vLLM. ([#1534](https://github.com/vllm-project/vllm-gaudi/pull/1534))
- Fixed HPU-specific bugs in KV-offload with async spec-decode. ([#1401](https://github.com/vllm-project/vllm-gaudi/pull/1401))
- Fixed decode bucket generation for hybrid models with mismatched block sizes. ([#1485](https://github.com/vllm-project/vllm-gaudi/pull/1485))
- Fixed `conv_state` corruption in compact GDN batched decode. ([#1431](https://github.com/vllm-project/vllm-gaudi/pull/1431))
- Fixed `mamba_type` comparison for GDN hybrid cache allocation. ([#1449](https://github.com/vllm-project/vllm-gaudi/pull/1449))
- Reverted materialised causal `attn_bias` on FSDPA for non-GDN hybrid models. ([#1482](https://github.com/vllm-project/vllm-gaudi/pull/1482))

---

## Quantization

- Freed dead INC-quantized FP8 MoE weight copies to roughly halve device memory. ([#1583](https://github.com/vllm-project/vllm-gaudi/pull/1583))
- Fixed Maverick FP8 INC calibration OOM after the upstream MoE refactor. ([#1590](https://github.com/vllm-project/vllm-gaudi/pull/1590))
- Fixed DeepSeek-R1 FP8 grouped-topk MoE graph-compile failure. ([#1586](https://github.com/vllm-project/vllm-gaudi/pull/1586))
- Fixed a dense-model INC quantization OOM regression. ([#1605](https://github.com/vllm-project/vllm-gaudi/pull/1605))
- Fixed an FP8 dequant device mismatch. ([#1597](https://github.com/vllm-project/vllm-gaudi/pull/1597))
- Fixed gpt-oss MxFP4 bias handling. ([#1578](https://github.com/vllm-project/vllm-gaudi/pull/1578))
- Fixed gpt-oss MxFP4 TP partitioning and `quant_method` matching. ([#1499](https://github.com/vllm-project/vllm-gaudi/pull/1499))
- Skipped the WNA16 MoE backend oracle on Gaudi. ([#1513](https://github.com/vllm-project/vllm-gaudi/pull/1513))
- Bypassed `_forward_impl` for `dp_size==1` to fix a DeepSeek R1 FP8 crash. ([#1441](https://github.com/vllm-project/vllm-gaudi/pull/1441))
- Warned when `calibrate_model.sh -u` is used on TP>1 without `-r`. ([#1607](https://github.com/vllm-project/vllm-gaudi/pull/1607))

---

## Plugin Core

- Applied hetero fixes for upstream vLLM v0.24.0. ([#1592](https://github.com/vllm-project/vllm-gaudi/pull/1592))
- Adapted HPU MoE to the FusedMoE/MoERunner inversion and fixed KV-connector and sampler drift. ([#1536](https://github.com/vllm-project/vllm-gaudi/pull/1536))
- Detached the shared MoE gate when experts is the `MoERunner`. ([#1566](https://github.com/vllm-project/vllm-gaudi/pull/1566))
- Fixed a stale MoE refactor path. ([#1442](https://github.com/vllm-project/vllm-gaudi/pull/1442))
- Adapted HPU scheduler, ngram proposer, and offloading-connector tests to upstream API drift. ([#1556](https://github.com/vllm-project/vllm-gaudi/pull/1556))
- Adapted `multi_model_api_server` to the vLLM `ServingTokenization` refactor. ([#1558](https://github.com/vllm-project/vllm-gaudi/pull/1558))
- Overrode the HPU offloading handler `get_finished`/`shutdown`. ([#1525](https://github.com/vllm-project/vllm-gaudi/pull/1525))
- Fixed the `online_model_swap` generate import after upstream resettlement. ([#1522](https://github.com/vllm-project/vllm-gaudi/pull/1522))
- Fixed the `gdn_linear_attn` import path after the upstream Mamba refactor. ([#1496](https://github.com/vllm-project/vllm-gaudi/pull/1496))
- Fixed `DynamicNTKScalingRotaryEmbedding` and `HPUCompressedTensorsConfig`. ([#1479](https://github.com/vllm-project/vllm-gaudi/pull/1479))
- Fixed `MultiModelEngineClient`, Qwen3.5 compilation, and EPLB refactoring. ([#1436](https://github.com/vllm-project/vllm-gaudi/pull/1436))
- Made `granite-4.0-h-small` compatible with single-process model swap. ([#1595](https://github.com/vllm-project/vllm-gaudi/pull/1595))

---

## Serving and Infrastructure

- Capped `fastapi<0.137` to unbreak the Prometheus instrumentator. ([#1551](https://github.com/vllm-project/vllm-gaudi/pull/1551))
- Pinned `transformers==5.9.0` temporarily for compatibility. ([#1530](https://github.com/vllm-project/vllm-gaudi/pull/1530))
- Enhanced process management for the online model-swap example. ([#1414](https://github.com/vllm-project/vllm-gaudi/pull/1414))
- Documented `--distributed-executor-backend` and `VLLM_WORKER_MULTIPROC_METHOD` in the README. ([#1448](https://github.com/vllm-project/vllm-gaudi/pull/1448))
- Updated the LoRA tests. ([#1488](https://github.com/vllm-project/vllm-gaudi/pull/1488))
- Hardened the Qwen3.5 CI test to detect regressions. ([#1443](https://github.com/vllm-project/vllm-gaudi/pull/1443))
- Fixed the documented `torchaudio` version. ([#1425](https://github.com/vllm-project/vllm-gaudi/pull/1425))

---

## Fixes

- Fixed an eager-only env var leaking into lazy subprocesses; respect user input. ([#1524](https://github.com/vllm-project/vllm-gaudi/pull/1524))
- Fixed HPU `prompt_token_ids` device placement for penalty sampling. ([#1465](https://github.com/vllm-project/vllm-gaudi/pull/1465))
- Fixed a stale gate ref overriding caller `router_logits` in the `dp_size==1` MoE fast path. ([#1469](https://github.com/vllm-project/vllm-gaudi/pull/1469))
- Fixed `patch_hf3fs_mock_client_for_cpu_only`. ([#1439](https://github.com/vllm-project/vllm-gaudi/pull/1439))
- Fixed the kernel block size. ([#1453](https://github.com/vllm-project/vllm-gaudi/pull/1453))
- Replaced `batched_count_greater_than` to avoid a dynamic-shape `TypeError` on HPU. ([#1412](https://github.com/vllm-project/vllm-gaudi/pull/1412))
- Fixed a hybrid-model warmup block_size mismatch (Qwen3.5-35B-A3B). ([#1434](https://github.com/vllm-project/vllm-gaudi/pull/1434))
- Fixed M-RoPE accuracy for Qwen. ([#1437](https://github.com/vllm-project/vllm-gaudi/pull/1437))
- Fixed an accuracy issue in `minimax_m2` with TP>1. ([#1451](https://github.com/vllm-project/vllm-gaudi/pull/1451))
- Fixed `granite-4.0-h-small` tool-calling accuracy. ([#1561](https://github.com/vllm-project/vllm-gaudi/pull/1561))
- Removed the `granite-4.0-h-small` OOM workaround. ([#1579](https://github.com/vllm-project/vllm-gaudi/pull/1579))

---

## Security

- Added a [SECURITY.md](https://github.com/vllm-project/vllm-gaudi/blob/main/SECURITY.md) vulnerability disclosure policy. ([#1509](https://github.com/vllm-project/vllm-gaudi/pull/1509))
- Removed the outdated and insecure `decord` dependency. ([#1520](https://github.com/vllm-project/vllm-gaudi/pull/1520))
- Routed `HF_TOKEN`using CI jobs through an approved-workflow environment. ([#1473](https://github.com/vllm-project/vllm-gaudi/pull/1473))
- Added pre-merge approval for `execute_pre_merge`. ([#1471](https://github.com/vllm-project/vllm-gaudi/pull/1471))

---

## Deprecation and Breaking Changes

- Upgraded to Intel® Gaudi® Software v1.24.1 and PyTorch 2.11, which requires users to update their Intel® Gaudi® software stack to [v1.24.1](https://docs.habana.ai/en/v1.24.1/Release_Notes/GAUDI_Release_Notes.html).
- Removed `ray` and redundant `transformers` packages from the Gaudi requirements. ([#1445](https://github.com/vllm-project/vllm-gaudi/pull/1445))
- Removed the `decord` dependency; environments relying on it must migrate to a supported decoder. ([#1520](https://github.com/vllm-project/vllm-gaudi/pull/1520))
- Temporarily pinned `transformers==5.9.0`; expect this constraint to be relaxed in a later release. ([#1530](https://github.com/vllm-project/vllm-gaudi/pull/1530))

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1609](https://github.com/vllm-project/vllm-gaudi/pull/1609) | [v0.24.0] Fix decode throughput regression from vLLM #42656 | @adobrzyn |
| [#1605](https://github.com/vllm-project/vllm-gaudi/pull/1605) | [v0.24.0] Fix dense-model INC quant OOM regression from #1590 | @adobrzyn |
| [#1607](https://github.com/vllm-project/vllm-gaudi/pull/1607) | Warn when calibrate_model.sh -u is used on TP>1 without -r | @adobrzyn |
| [#1597](https://github.com/vllm-project/vllm-gaudi/pull/1597) | Fix FP8 dequant device mismatch | @mkrze |
| [#1595](https://github.com/vllm-project/vllm-gaudi/pull/1595) | granite-4.0-h-small made compatible with single process models swap | @12010486 |
| [#1592](https://github.com/vllm-project/vllm-gaudi/pull/1592) | hetero fixes for vllm v0.24.0 | @adobrzyn |
| [#1590](https://github.com/vllm-project/vllm-gaudi/pull/1590) | Fix Maverick FP8 INC calibration OOM after #41184 MoE refactor | @iboiko-habana |
| [#1586](https://github.com/vllm-project/vllm-gaudi/pull/1586) | Fix DeepSeek-R1 FP8 grouped-topk MoE graph compile failure | @iboiko-habana |
| [#1585](https://github.com/vllm-project/vllm-gaudi/pull/1585) | Remove conv1d update TPC kernel | @slokesha |
| [#1584](https://github.com/vllm-project/vllm-gaudi/pull/1584) | Fix decode bucket sparsity for longprompt | @shepark |
| [#1583](https://github.com/vllm-project/vllm-gaudi/pull/1583) | Free dead INC-quantized FP8 MoE weight copy to halve device memory | @iboiko-habana |
| [#1579](https://github.com/vllm-project/vllm-gaudi/pull/1579) | granite-4.0-h-small - remove OOM workaround | @rsmyrek |
| [#1578](https://github.com/vllm-project/vllm-gaudi/pull/1578) | Gptoss mxfp4 bias v0.24.0 | @SKRohit |
| [#1566](https://github.com/vllm-project/vllm-gaudi/pull/1566) | Detach shared MoE gate when experts is the MoERunner (vLLM #41184) | @iboiko-habana |
| [#1561](https://github.com/vllm-project/vllm-gaudi/pull/1561) | granite-4.0-h-small - toolcalling accuracy fix | @rsmyrek |
| [#1558](https://github.com/vllm-project/vllm-gaudi/pull/1558) | Adapt multi_model_api_server to vLLM ServingTokenization refactor | @pawel-olejniczak |
| [#1557](https://github.com/vllm-project/vllm-gaudi/pull/1557) | 3 hourly-CI fixes | @pawel-olejniczak |
| [#1556](https://github.com/vllm-project/vllm-gaudi/pull/1556) | Adapt HPU scheduler, ngram proposer and offloading connector tests to upstream API drift | @pawel-olejniczak |
| [#1554](https://github.com/vllm-project/vllm-gaudi/pull/1554) | Add Qwen3.6-35B CI test | @iboiko-habana |
| [#1548](https://github.com/vllm-project/vllm-gaudi/pull/1548) | Replace Sampler.gather_logprobs to remove mark_unbacked calls | @shepark |
| [#1546](https://github.com/vllm-project/vllm-gaudi/pull/1546) | [HPU] Fix contiguous PA wrong-rows on prefill context (perf-preserving) | @adobrzyn |
| [#1536](https://github.com/vllm-project/vllm-gaudi/pull/1536) | Adapt HPU MoE to the FusedMoE/MoERunner inversion and fix KV-connector and sampler drift | @pawel-olejniczak |
| [#1534](https://github.com/vllm-project/vllm-gaudi/pull/1534) | Fix hetero HPU NIXL connector compatibility with vLLM v0.22.1rc1 | @hsubramony |
| [#1530](https://github.com/vllm-project/vllm-gaudi/pull/1530) | Temporary fixed version transformers==5.9.0 | @iboiko-habana |
| [#1527](https://github.com/vllm-project/vllm-gaudi/pull/1527) | [HPU] Use TPC native ops for causal_conv1d update | @slokesha |
| [#1525](https://github.com/vllm-project/vllm-gaudi/pull/1525) | Override HPU offloading handler get_finished/shutdown | @pawel-olejniczak |
| [#1524](https://github.com/vllm-project/vllm-gaudi/pull/1524) | Fix eager-only env var leak into lazy subprocesses; respect user input | @adobrzyn |
| [#1522](https://github.com/vllm-project/vllm-gaudi/pull/1522) | Fix online_model_swap generate import after upstream resettlement | @pawel-olejniczak |
| [#1520](https://github.com/vllm-project/vllm-gaudi/pull/1520) | fix: remove outdated and insecure decord dependency | @adobrzyn |
| [#1513](https://github.com/vllm-project/vllm-gaudi/pull/1513) | Skip WNA16 MoE backend oracle on Gaudi | @pawel-olejniczak |
| [#1509](https://github.com/vllm-project/vllm-gaudi/pull/1509) | Create SECURITY.md | @sfblackl-intel |
| [#1508](https://github.com/vllm-project/vllm-gaudi/pull/1508) | Fix offloading connector prepare_store test | @pawel-olejniczak |
| [#1499](https://github.com/vllm-project/vllm-gaudi/pull/1499) | Fix gpt-oss mxfp4 TP partitioning and quant_method matching | @mkrze |
| [#1496](https://github.com/vllm-project/vllm-gaudi/pull/1496) | Fix gdn_linear_attn import path after upstream mamba refactor | @pawel-olejniczak |
| [#1488](https://github.com/vllm-project/vllm-gaudi/pull/1488) | Update lora tests | @iboiko-habana |
| [#1485](https://github.com/vllm-project/vllm-gaudi/pull/1485) | Fix decode bucket generation for hybrid models with mismatched block sizes | @yangulei |
| [#1482](https://github.com/vllm-project/vllm-gaudi/pull/1482) | Revert "Skip materialised causal attn_bias on FSDPA for non-GDN hybrid models" | @rsmyrek |
| [#1479](https://github.com/vllm-project/vllm-gaudi/pull/1479) | Fix DynamicNTKScalingRotaryEmbedding and HPUCompressedTensorsConfig | @pawel-olejniczak |
| [#1478](https://github.com/vllm-project/vllm-gaudi/pull/1478) | Fix graph (re)compilation warnings for single process models swap | @12010486 |
| [#1473](https://github.com/vllm-project/vllm-gaudi/pull/1473) | ci: route HF_TOKEN-using jobs through approved-workflow environment | @adobrzyn |
| [#1471](https://github.com/vllm-project/vllm-gaudi/pull/1471) | Add pre-merge-approval for execute_pre_merge | @bmyrcha |
| [#1469](https://github.com/vllm-project/vllm-gaudi/pull/1469) | Fix stale gate ref overriding caller router_logits in dp_size==1 MoE fast path | @iboiko-habana |
| [#1468](https://github.com/vllm-project/vllm-gaudi/pull/1468) | Fix offloading_connector test flush assertion for load transfers | @pawel-olejniczak |
| [#1465](https://github.com/vllm-project/vllm-gaudi/pull/1465) | Fix HPU prompt_token_ids device placement for penalty sampling | @yeonsily |
| [#1464](https://github.com/vllm-project/vllm-gaudi/pull/1464) | Increase timeout from default 6h to 12h | @bmyrcha |
| [#1454](https://github.com/vllm-project/vllm-gaudi/pull/1454) | Remove num_ctx_tokens_less_or_equal_batched_max_model_len filter | @yangulei |
| [#1453](https://github.com/vllm-project/vllm-gaudi/pull/1453) | fix kernel block size | @iboiko-habana |
| [#1451](https://github.com/vllm-project/vllm-gaudi/pull/1451) | Fix accuracy issue in minimax_m2 with TP > 1 | @skavulya |
| [#1450](https://github.com/vllm-project/vllm-gaudi/pull/1450) | Add Qwen3NextForCausalLM to mamba_like_arch | @rsmyrek |
| [#1449](https://github.com/vllm-project/vllm-gaudi/pull/1449) | Fix mamba_type comparison for GDN hybrid cache allocation | @shepark |
| [#1448](https://github.com/vllm-project/vllm-gaudi/pull/1448) | README update for --distributed-executor-backend and VLLM_WORKER_MULTIPROC_METHOD | @iboiko-habana |
| [#1447](https://github.com/vllm-project/vllm-gaudi/pull/1447) | Fix decode bucket filter issues | @yangulei |
| [#1445](https://github.com/vllm-project/vllm-gaudi/pull/1445) | Removal of ray and redundant transformers packages from gaudi requirements | @iboiko-habana |
| [#1443](https://github.com/vllm-project/vllm-gaudi/pull/1443) | Harden Qwen3.5 CI test to detect regressions | @shepark |
| [#1442](https://github.com/vllm-project/vllm-gaudi/pull/1442) | Fix for MoE refactor #35178 | @iboiko-habana |
| [#1441](https://github.com/vllm-project/vllm-gaudi/pull/1441) | fix: bypass _forward_impl for dp_size==1 to fix DeepSeek R1 FP8 crash | @kamil-kaczor |
| [#1439](https://github.com/vllm-project/vllm-gaudi/pull/1439) | Fix patch_hf3fs_mock_client_for_cpu_only | @hsubramony |
| [#1437](https://github.com/vllm-project/vllm-gaudi/pull/1437) | Mrope accuracy fix for qwen | @hsubramony |
| [#1436](https://github.com/vllm-project/vllm-gaudi/pull/1436) | Fix MultiModelEngineClient, Qwen3.5 compilation, and EPLB refactoring | @pawel-olejniczak |
| [#1434](https://github.com/vllm-project/vllm-gaudi/pull/1434) | fix: hybrid model warmup block_size mismatch (Qwen3.5-35B-A3B) | @adobrzyn |
| [#1431](https://github.com/vllm-project/vllm-gaudi/pull/1431) | Fix conv_state corruption in compact GDN batched decode | @osavchenkox |
| [#1415](https://github.com/vllm-project/vllm-gaudi/pull/1415) | Add hybrid models support for single card models swap | @12010486 |
| [#1414](https://github.com/vllm-project/vllm-gaudi/pull/1414) | Enhance process management for online model swap example | @12010486 |
| [#1412](https://github.com/vllm-project/vllm-gaudi/pull/1412) | fix: replace batched_count_greater_than to avoid dynamic shape TypeError on HPU | @kamil-kaczor |
| [#1401](https://github.com/vllm-project/vllm-gaudi/pull/1401) | fix: HPU-specific bug fixes for KV-offload + async spec-decode | @hsubramony |
| [#1425](https://github.com/vllm-project/vllm-gaudi/pull/1425) | [DOC] Fix torchaudio version | @yangulei |
| [#1551](https://github.com/vllm-project/vllm-gaudi/pull/1551) | fix: cap fastapi<0.137 to unbreak prometheus instrumentator | @pawel-olejniczak |

---

## New Contributors

Welcome to the following first-time contributor to vLLM Gaudi Plugin!

- **@sfblackl-intel** — Create SECURITY.md ([#1509](https://github.com/vllm-project/vllm-gaudi/pull/1509))
