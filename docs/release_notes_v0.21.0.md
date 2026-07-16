# vLLM Gaudi Plugin v0.21.0 Release Notes

## Overview

This release is based on [vLLM v0.21.0](https://github.com/vllm-project/vllm/releases/tag/v0.21.0) and supports [Intel® Gaudi® Software v1.24.0](https://docs.habana.ai/en/v1.24.0/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.10.

---

## Highlights

- **Removed lazy execution mode from CI** — eager mode is now the default in CI pipelines; lazy mode is still supported at runtime. ([#996](https://github.com/vllm-project/vllm-gaudi/pull/996))
- Introduced **new padding-aware bucketing strategy** for improved memory utilization and reduced padding overhead. ([#762](https://github.com/vllm-project/vllm-gaudi/pull/762))
- Added **W8A8 INT8 quantization with BF16 fallback** via `HPUCompressedTensorsW8A8Int8_BF16Fallback`. ([#1168](https://github.com/vllm-project/vllm-gaudi/pull/1168))
- Enabled **FusedSDPA slicing** for better attention performance. ([#1155](https://github.com/vllm-project/vllm-gaudi/pull/1155))
- Added **OpenAI-compatible `/v1/models/switch` entrypoint** and per-model tool-calling and FP8 configs for online model swap. ([#1258](https://github.com/vllm-project/vllm-gaudi/pull/1258))
- Introduced **HPU-specific KV-offload and async speculative decoding** fixes. ([#1264](https://github.com/vllm-project/vllm-gaudi/pull/1264))
- Fixed **NIXL connector heterogeneous and homogeneous** deployment scenarios. ([#1511](https://github.com/vllm-project/vllm-gaudi/pull/1511), [#1503](https://github.com/vllm-project/vllm-gaudi/pull/1503))

---

## Performance

- Introduced a new padding-aware bucketing strategy. ([#762](https://github.com/vllm-project/vllm-gaudi/pull/762))
- Enabled slicing for the FusedSDPA attention path. ([#1155](https://github.com/vllm-project/vllm-gaudi/pull/1155))
- Skipped HPU graphs for long (query + context) prefills. ([#1346](https://github.com/vllm-project/vllm-gaudi/pull/1346))
- Fixed guard breaks and improved warmup time for Qwen3 MoE. ([#1329](https://github.com/vllm-project/vllm-gaudi/pull/1329))
- Improved `selective_state_update` performance. ([#1291](https://github.com/vllm-project/vllm-gaudi/pull/1291))
- Removed splitting MoE decode layer compilation function. ([#1313](https://github.com/vllm-project/vllm-gaudi/pull/1313))
- Optimized visible block number for hybrid KV cache. ([#1317](https://github.com/vllm-project/vllm-gaudi/pull/1317))
- Fine-tuned bucketing edge cases for longer contexts. ([#1362](https://github.com/vllm-project/vllm-gaudi/pull/1362))
- Raised the default `max_cudagraph_capture_size` floor to 16384. ([#1507](https://github.com/vllm-project/vllm-gaudi/pull/1507))

---

## Attention & KV Cache

- Added prefix caching support for HPUMambaMixer2. ([#1366](https://github.com/vllm-project/vllm-gaudi/pull/1366))
- Fixed condition for materialised causal `attn_bias`. ([#1433](https://github.com/vllm-project/vllm-gaudi/pull/1433))
- Fixed proper KV cache slot addressing for hybrid models. ([#1327](https://github.com/vllm-project/vllm-gaudi/pull/1327))
- Fixed `mamba_type` comparison for GDN hybrid cache allocation. ([#1449](https://github.com/vllm-project/vllm-gaudi/pull/1449))
- Fixed extra masking for batched prefill in GDN layers. ([#1440](https://github.com/vllm-project/vllm-gaudi/pull/1440))
- Fixed HPU-specific bugs for KV-offload and async speculative decoding. ([#1264](https://github.com/vllm-project/vllm-gaudi/pull/1264))

---

## Quantization

- Implemented `HPUCompressedTensorsW8A8Int8_BF16Fallback` for W8A8 INT8 quantization with BF16 fallback. ([#1168](https://github.com/vllm-project/vllm-gaudi/pull/1168))
- Fixed Synapse GC compile failure for FP8-quantized models. ([#1324](https://github.com/vllm-project/vllm-gaudi/pull/1324))
- Enabled Llama4 Maverick FP8 torch.compile without breaking DeepSeek. ([#1396](https://github.com/vllm-project/vllm-gaudi/pull/1396))
- Fixed GPT-OSS MxFP4 TP partitioning and `quant_method` matching. ([#1498](https://github.com/vllm-project/vllm-gaudi/pull/1498))
- Added Granite-4.0-h calibration config. ([#1270](https://github.com/vllm-project/vllm-gaudi/pull/1270))
- Fixed load failure of MxFP4 GPT-OSS-120B with expert parallel. ([#1411](https://github.com/vllm-project/vllm-gaudi/pull/1411))
- Added `hf_config` parameter to HPU quantization config overrides. ([#1349](https://github.com/vllm-project/vllm-gaudi/pull/1349))

---

## Plugin Core

- Removed lazy execution mode from CI — eager is now the default in CI pipelines. ([#996](https://github.com/vllm-project/vllm-gaudi/pull/996))
- Accepted PEP 440 versions in build detection. ([#1351](https://github.com/vllm-project/vllm-gaudi/pull/1351))
- Patched `torch.accelerator.empty_cache` for HPU to fix import-order dependent cleanup failures. ([#1430](https://github.com/vllm-project/vllm-gaudi/pull/1430))
- Removed `matmul_qk` output-tensor compatibility gate after 1.24.0. ([#1409](https://github.com/vllm-project/vllm-gaudi/pull/1409))
- Removed `transformers` installation from vllm-gaudi. ([#1494](https://github.com/vllm-project/vllm-gaudi/pull/1494))
- Prevented eager-mode env vars from leaking to lazy-mode subprocesses. ([#1510](https://github.com/vllm-project/vllm-gaudi/pull/1510))
- Fixed multiple upstream regressions across MoE, MLA, NIXL, attention, FP8, offloading, and platform modules. ([#1279](https://github.com/vllm-project/vllm-gaudi/pull/1279), [#1311](https://github.com/vllm-project/vllm-gaudi/pull/1311), [#1338](https://github.com/vllm-project/vllm-gaudi/pull/1338), [#1342](https://github.com/vllm-project/vllm-gaudi/pull/1342), [#1354](https://github.com/vllm-project/vllm-gaudi/pull/1354), [#1375](https://github.com/vllm-project/vllm-gaudi/pull/1375), [#1377](https://github.com/vllm-project/vllm-gaudi/pull/1377), [#1403](https://github.com/vllm-project/vllm-gaudi/pull/1403), [#1421](https://github.com/vllm-project/vllm-gaudi/pull/1421), [#1428](https://github.com/vllm-project/vllm-gaudi/pull/1428), [#1442](https://github.com/vllm-project/vllm-gaudi/pull/1442))
- Ported fixes for MoE fast path, dynamic shape, kernel block size, and batched count operations. ([#1453](https://github.com/vllm-project/vllm-gaudi/pull/1453), [#1458](https://github.com/vllm-project/vllm-gaudi/pull/1458), [#1459](https://github.com/vllm-project/vllm-gaudi/pull/1459), [#1460](https://github.com/vllm-project/vllm-gaudi/pull/1460), [#1469](https://github.com/vllm-project/vllm-gaudi/pull/1469))

---

## Serving & Infrastructure

- Added torchaudio-free copies of CD Dockerfiles. ([#1446](https://github.com/vllm-project/vllm-gaudi/pull/1446))
- Set Docker `PT_HPU_LAZY_MODE=0` as the default auto-calculated value. ([#1378](https://github.com/vllm-project/vllm-gaudi/pull/1378))
- Added OpenAI-compatible `/v1/models/switch` entrypoint integration. ([#1258](https://github.com/vllm-project/vllm-gaudi/pull/1258))
- Added per-model tool-calling and FP8 configs. ([#1258](https://github.com/vllm-project/vllm-gaudi/pull/1258))
- Enhanced process management for online model swap example. ([#1414](https://github.com/vllm-project/vllm-gaudi/pull/1414))
- Fixed NIXL connector V1 API signature mismatches for heterogeneous HPU. ([#1503](https://github.com/vllm-project/vllm-gaudi/pull/1503))
- Fixed heterogeneous and homogeneous NIXL deployment issues for v0.21.0. ([#1511](https://github.com/vllm-project/vllm-gaudi/pull/1511))
- Enabled defragmentation when contiguous PA is enabled. ([#1400](https://github.com/vllm-project/vllm-gaudi/pull/1400))
- Clarified `VLLM_PROMPT_BS_BUCKET_MAX` runtime behavior in docs. ([#1410](https://github.com/vllm-project/vllm-gaudi/pull/1410))

---

## Fixes

- Fixed occasional Qwen3.5 segfault. ([#1500](https://github.com/vllm-project/vllm-gaudi/pull/1500))
- Fixed decode bucket generation for hybrid models with mismatched block sizes. ([#1486](https://github.com/vllm-project/vllm-gaudi/pull/1486))
- Fixed HPU `prompt_token_ids` device placement for penalty sampling. ([#1466](https://github.com/vllm-project/vllm-gaudi/pull/1466))
- Fixed decode bucket filter issues. ([#1447](https://github.com/vllm-project/vllm-gaudi/pull/1447))
- Fixed decode bucketing in non-contiguous PA scenario. ([#1122](https://github.com/vllm-project/vllm-gaudi/pull/1122))
- Fixed MRoPE accuracy for Qwen models. ([#1437](https://github.com/vllm-project/vllm-gaudi/pull/1437))
- Fixed warmup failure and multimodal graph warmup in PT compile-only mode. ([#1392](https://github.com/vllm-project/vllm-gaudi/pull/1392))
- Flattened 3D `inputs_embeds` in `HpuModelAdapter.forward`. ([#1381](https://github.com/vllm-project/vllm-gaudi/pull/1381))
- Fixed prompt logprobs gathering on Gaudi HPU. ([#1405](https://github.com/vllm-project/vllm-gaudi/pull/1405))
- Fixed regression in Mistral-Large-3-675B. ([#1304](https://github.com/vllm-project/vllm-gaudi/pull/1304))
- Fixed Granite 4h block size calculations. ([#1332](https://github.com/vllm-project/vllm-gaudi/pull/1332))
- Separated conv1d for Granite 4.0. ([#1322](https://github.com/vllm-project/vllm-gaudi/pull/1322))
- Corrected `get_supported_kernel_block_sizes`. ([#1384](https://github.com/vllm-project/vllm-gaudi/pull/1384))
- Fixed MoE graph breaks from ForwardContext lookups. ([#1357](https://github.com/vllm-project/vllm-gaudi/pull/1357))
- Reset hybrid `block_size` to 128 for tool calling. ([#1303](https://github.com/vllm-project/vllm-gaudi/pull/1303))
- Fixed hybrid model warmup `block_size` mismatch (Qwen3.5-35B-A3B). ([#1462](https://github.com/vllm-project/vllm-gaudi/pull/1462))
- Fixed stale gate ref overriding caller `router_logits` in dp_size==1 MoE fast path. ([#1469](https://github.com/vllm-project/vllm-gaudi/pull/1469))
- Fixed Ernie4.5-VL test. ([#1105](https://github.com/vllm-project/vllm-gaudi/pull/1105))
- Eliminated Llama4 torch.compile recompilations on HPU. ([#1360](https://github.com/vllm-project/vllm-gaudi/pull/1360))

---

## Deprecation & Breaking Changes

- Removed `transformers` installation from vllm-gaudi — it is now expected to be provided by the base environment. ([#1494](https://github.com/vllm-project/vllm-gaudi/pull/1494))

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1511](https://github.com/vllm-project/vllm-gaudi/pull/1511) | Heterogeneous and Homogeneous fixes for v0.21.0 | @hsubramony |
| [#1510](https://github.com/vllm-project/vllm-gaudi/pull/1510) | fix: prevent eager-mode env vars leaking to lazy-mode subprocesses | @adobrzyn |
| [#1507](https://github.com/vllm-project/vllm-gaudi/pull/1507) | fix: raise default max_cudagraph_capture_size floor to 16384 | @adobrzyn |
| [#1503](https://github.com/vllm-project/vllm-gaudi/pull/1503) | Fix NIXL connector V1 API signature mismatches for hetero HPU | @hsubramony |
| [#1500](https://github.com/vllm-project/vllm-gaudi/pull/1500) | QWN35: fix occasional segfault | @libinta |
| [#1498](https://github.com/vllm-project/vllm-gaudi/pull/1498) | Fix gpt-oss mxfp4 TP partitioning and quant_method matching | @mkrze |
| [#1494](https://github.com/vllm-project/vllm-gaudi/pull/1494) | Remove transformers installation from vllm-gaudi | @iboiko-habana |
| [#1491](https://github.com/vllm-project/vllm-gaudi/pull/1491) | ci: route HF_TOKEN-using jobs through approved-workflow environment | @adobrzyn |
| [#1489](https://github.com/vllm-project/vllm-gaudi/pull/1489) | Port of: Update lora tests | @iboiko-habana |
| [#1486](https://github.com/vllm-project/vllm-gaudi/pull/1486) | Fix decode bucket generation for hybrid models with mismatched block sizes | @yangulei |
| [#1471](https://github.com/vllm-project/vllm-gaudi/pull/1471) | Add pre-merge-approval for execute_pre_merge | @bmyrcha |
| [#1469](https://github.com/vllm-project/vllm-gaudi/pull/1469) | Port of: Fix stale gate ref overriding caller router_logits in dp_size==1 MoE fast path | @iboiko-habana |
| [#1466](https://github.com/vllm-project/vllm-gaudi/pull/1466) | Fix HPU prompt_token_ids device placement for penalty sampling | @yeonsily |
| [#1462](https://github.com/vllm-project/vllm-gaudi/pull/1462) | Port of: fix: hybrid model warmup block_size mismatch (Qwen3.5-35B-A3B) | @iboiko-habana |
| [#1460](https://github.com/vllm-project/vllm-gaudi/pull/1460) | Port of: Remove num_ctx_tokens_less_or_equal_batched_max_model_len filter | @iboiko-habana |
| [#1459](https://github.com/vllm-project/vllm-gaudi/pull/1459) | Port of: fix: bypass _forward_impl for dp_size==1 to fix DeepSeek R1 FP8 crash | @iboiko-habana |
| [#1458](https://github.com/vllm-project/vllm-gaudi/pull/1458) | Port of: fix: replace batched_count_greater_than to avoid dynamic shape TypeError on HPU | @iboiko-habana |
| [#1453](https://github.com/vllm-project/vllm-gaudi/pull/1453) | Port of: fix kernel block size | @iboiko-habana |
| [#1449](https://github.com/vllm-project/vllm-gaudi/pull/1449) | Fix mamba_type comparison for GDN hybrid cache allocation | @shepark |
| [#1447](https://github.com/vllm-project/vllm-gaudi/pull/1447) | Fix decode bucket filter issues | @yangulei |
| [#1446](https://github.com/vllm-project/vllm-gaudi/pull/1446) | Add torchaudio-free copies of CD Dockerfiles | @PatrykWo |
| [#1444](https://github.com/vllm-project/vllm-gaudi/pull/1444) | [MiniMax-M2] Remove reduce_results kwarg from FusedMoE init | @mounikamandava |
| [#1443](https://github.com/vllm-project/vllm-gaudi/pull/1443) | Harden Qwen3.5 CI test to detect regressions | @shepark |
| [#1442](https://github.com/vllm-project/vllm-gaudi/pull/1442) | Fix for MoE refactor | @iboiko-habana |
| [#1440](https://github.com/vllm-project/vllm-gaudi/pull/1440) | fix extra masking for batched prefill in GDN layers | @osavchenkox |
| [#1437](https://github.com/vllm-project/vllm-gaudi/pull/1437) | MRoPE accuracy fix for Qwen | @hsubramony |
| [#1435](https://github.com/vllm-project/vllm-gaudi/pull/1435) | fix: guard CUDA sync in hf3fs mock client for HPU compatibility | @kamil-kaczor |
| [#1433](https://github.com/vllm-project/vllm-gaudi/pull/1433) | Fixing condition for materialised causal attn_bias | @ksmusz |
| [#1430](https://github.com/vllm-project/vllm-gaudi/pull/1430) | fix: patch torch.accelerator.empty_cache for HPU to fix import-order dependent cleanup failures | @kamil-kaczor |
| [#1428](https://github.com/vllm-project/vllm-gaudi/pull/1428) | Fix moe_forward hidden_dim_unpadded parameter | @pawel-olejniczak |
| [#1425](https://github.com/vllm-project/vllm-gaudi/pull/1425) | [DOC] Fix torchaudio version | @yangulei |
| [#1424](https://github.com/vllm-project/vllm-gaudi/pull/1424) | fix: lower eagle3 spec_decode accuracy threshold to 0.60 | @kamil-kaczor |
| [#1422](https://github.com/vllm-project/vllm-gaudi/pull/1422) | CI cleanup v0.20.0 | @iboiko-habana |
| [#1421](https://github.com/vllm-project/vllm-gaudi/pull/1421) | Fix upstream regressions: MoE PluggableLayer recursion, MLA attention init crashes, KV offload module consolidation | @pawel-olejniczak |
| [#1414](https://github.com/vllm-project/vllm-gaudi/pull/1414) | Enhance process management for online model swap example | @12010486 |
| [#1411](https://github.com/vllm-project/vllm-gaudi/pull/1411) | Fix load failure of mxfp4 gpt-oss-120b with expert parallel | @malsbat |
| [#1410](https://github.com/vllm-project/vllm-gaudi/pull/1410) | Clarify VLLM_PROMPT_BS_BUCKET_MAX runtime behavior in docs | @iboiko-habana |
| [#1409](https://github.com/vllm-project/vllm-gaudi/pull/1409) | Remove matmul_qk output-tensor compatibility gate after 1.24.0 | @iboiko-habana |
| [#1405](https://github.com/vllm-project/vllm-gaudi/pull/1405) | Fix prompt logprobs gathering on Gaudi HPU | @iboiko-habana |
| [#1403](https://github.com/vllm-project/vllm-gaudi/pull/1403) | Fix upstream regressions: MoE refactor, DeepSeek V4 router, KV offload HMA | @pawel-olejniczak |
| [#1400](https://github.com/vllm-project/vllm-gaudi/pull/1400) | Enable defrag when contig_pa is enabled | @iboiko-habana |
| [#1399](https://github.com/vllm-project/vllm-gaudi/pull/1399) | QA test fixes for ValueError: No common block size for 16 | @hsubramony |
| [#1396](https://github.com/vllm-project/vllm-gaudi/pull/1396) | fix: enable Llama4 Maverick FP8 torch.compile without breaking DeepSeek | @adobrzyn |
| [#1392](https://github.com/vllm-project/vllm-gaudi/pull/1392) | Fix warmup failure and run mm graph warmup pt compile only mode | @shepark |
| [#1385](https://github.com/vllm-project/vllm-gaudi/pull/1385) | Update CODEOWNERS | @jbyczkow |
| [#1384](https://github.com/vllm-project/vllm-gaudi/pull/1384) | Correct get_supported_kernel_block_sizes | @jbyczkow |
| [#1383](https://github.com/vllm-project/vllm-gaudi/pull/1383) | fix: monkey-patch cleanup_dist_env_and_memory for HPU allocator | @kamil-kaczor |
| [#1381](https://github.com/vllm-project/vllm-gaudi/pull/1381) | flatten 3D inputs_embeds in HpuModelAdapter.forward | @shepark |
| [#1378](https://github.com/vllm-project/vllm-gaudi/pull/1378) | Set Docker auto calc PT_HPU_LAZY_MODE=0 as default | @nngokhale |
| [#1377](https://github.com/vllm-project/vllm-gaudi/pull/1377) | Fix upstream breakages: NIXL connector, TpKVTopology rename, MoE refactor, transformers v5 | @pawel-olejniczak |
| [#1375](https://github.com/vllm-project/vllm-gaudi/pull/1375) | Fix offloading test lambdas for upstream req_context parameter | @pawel-olejniczak |
| [#1366](https://github.com/vllm-project/vllm-gaudi/pull/1366) | Prefix caching support for HPUMambaMixer2 | @jbyczkow |
| [#1364](https://github.com/vllm-project/vllm-gaudi/pull/1364) | Logging omitted buckets when bucketing from file is enabled | @ksmusz |
| [#1363](https://github.com/vllm-project/vllm-gaudi/pull/1363) | Set GaudiSW version in CI to 1.24.0 | @afierka-intel |
| [#1362](https://github.com/vllm-project/vllm-gaudi/pull/1362) | Bucketing edge cases finetune for longer ctx | @ksmusz |
| [#1360](https://github.com/vllm-project/vllm-gaudi/pull/1360) | Eliminate Llama4 torch.compile recompilations on HPU | @adobrzyn |
| [#1357](https://github.com/vllm-project/vllm-gaudi/pull/1357) | Fix MoE graph breaks from ForwardContext lookups | @jbyczkow |
| [#1354](https://github.com/vllm-project/vllm-gaudi/pull/1354) | Fix upstream regressions in HPU worker, MoE router, and offloading tests | @pawel-olejniczak |
| [#1352](https://github.com/vllm-project/vllm-gaudi/pull/1352) | Add pre-merge-trigger.yaml | @bmyrcha |
| [#1351](https://github.com/vllm-project/vllm-gaudi/pull/1351) | Accept PEP 440 versions in build detection | @wjhrdy |
| [#1349](https://github.com/vllm-project/vllm-gaudi/pull/1349) | Add hf_config parameter to HPU quantization config overrides | @pawel-olejniczak |
| [#1342](https://github.com/vllm-project/vllm-gaudi/pull/1342) | Fix requirements paths and nixl_connector imports after upstream restructuring | @pawel-olejniczak |
| [#1339](https://github.com/vllm-project/vllm-gaudi/pull/1339) | Qwen3.5 changes cherry-picked from release 0.19.0 | @yeonsily |
| [#1338](https://github.com/vllm-project/vllm-gaudi/pull/1338) | Fix upstream regressions in attention, FP8, offloading and platform | @pawel-olejniczak |
| [#1332](https://github.com/vllm-project/vllm-gaudi/pull/1332) | Fix granite 4h block size calculations | @jkaniecki |
| [#1329](https://github.com/vllm-project/vllm-gaudi/pull/1329) | feat: fix guard breaks and improve warmup time for Qwen3 MoE | @kamil-kaczor |
| [#1327](https://github.com/vllm-project/vllm-gaudi/pull/1327) | Fix for proper KV cache slot addressing for Hybrid models | @ksmusz |
| [#1324](https://github.com/vllm-project/vllm-gaudi/pull/1324) | Fix Synapse GC compile failure for FP8-quantized models | @slokesha |
| [#1322](https://github.com/vllm-project/vllm-gaudi/pull/1322) | Separate conv1d for Granite 4.0 | @jbyczkow |
| [#1317](https://github.com/vllm-project/vllm-gaudi/pull/1317) | Optimizing visible block number for Hybrid kv_cache | @ksmusz |
| [#1313](https://github.com/vllm-project/vllm-gaudi/pull/1313) | Remove splitting moe decode layer compilation func | @shepark |
| [#1311](https://github.com/vllm-project/vllm-gaudi/pull/1311) | Fix Pixtral, MoE and Granite regressions | @pawel-olejniczak |
| [#1304](https://github.com/vllm-project/vllm-gaudi/pull/1304) | Fix regression in Mistral-Large-3-675B | @skavulya |
| [#1303](https://github.com/vllm-project/vllm-gaudi/pull/1303) | reset hybrid block_size to 128 for tool calling | @shepark |
| [#1291](https://github.com/vllm-project/vllm-gaudi/pull/1291) | improve selective_state_update | @osavchenkox |
| [#1279](https://github.com/vllm-project/vllm-gaudi/pull/1279) | Upstream vLLM compatibility fix | @iboiko-habana |
| [#1270](https://github.com/vllm-project/vllm-gaudi/pull/1270) | Granite-4.0-h Calibration config | @mfylcek |
| [#1264](https://github.com/vllm-project/vllm-gaudi/pull/1264) | fix: HPU-specific bug fixes for KV-offload + async spec-decode | @hsubramony |
| [#1258](https://github.com/vllm-project/vllm-gaudi/pull/1258) | Add per-model toolcalling and fp8 configs; OpenAI /v1/models/switch entrypoint | @12010486 |
| [#1242](https://github.com/vllm-project/vllm-gaudi/pull/1242) | avoid using pip show to get habana-torch-plugin version | @dtrifiro |
| [#1168](https://github.com/vllm-project/vllm-gaudi/pull/1168) | HPUCompressedTensorsW8A8Int8_BF16Fallback impl | @rsmyrek |
| [#1160](https://github.com/vllm-project/vllm-gaudi/pull/1160) | Revert "Cap decode block bucket limit to reduce warmup time" | @adobrzyn |
| [#1155](https://github.com/vllm-project/vllm-gaudi/pull/1155) | Enable slicing for the FusedSDPA | @yangulei |
| [#1122](https://github.com/vllm-project/vllm-gaudi/pull/1122) | Fixes for the decode bucketing in non-contiguous pa scenario | @yangulei |
| [#1105](https://github.com/vllm-project/vllm-gaudi/pull/1105) | fix ernie45-vl test | @jinyouzhi |
| [#996](https://github.com/vllm-project/vllm-gaudi/pull/996) | Remove all lazy execution mode from the codebase | @afierka-intel |
| [#762](https://github.com/vllm-project/vllm-gaudi/pull/762) | Add the padding-aware bucketing strategy | @yangulei |

---

## New Contributors

Welcome to the following first-time contributors to vLLM Gaudi Plugin!

- **@dtrifiro** — avoid using pip show to get habana-torch-plugin version ([#1242](https://github.com/vllm-project/vllm-gaudi/pull/1242))
- **@malsbat** — Fix load failure of mxfp4 gpt-oss-120b with expert parallel ([#1411](https://github.com/vllm-project/vllm-gaudi/pull/1411))
- **@mkrze** — Fix gpt-oss mxfp4 TP partitioning and quant_method matching ([#1498](https://github.com/vllm-project/vllm-gaudi/pull/1498))
- **@mounikamandava** — [MiniMax-M2] Remove reduce_results kwarg from FusedMoE init ([#1444](https://github.com/vllm-project/vllm-gaudi/pull/1444))
- **@osavchenkox** — fix extra masking for batched prefill in GDN layers ([#1440](https://github.com/vllm-project/vllm-gaudi/pull/1440))
- **@wjhrdy** — Accept PEP 440 versions in build detection ([#1351](https://github.com/vllm-project/vllm-gaudi/pull/1351))
