# vLLM Gaudi Plugin v0.17.1 Release Notes

## Overview

This release is based on [vLLM v0.17.1](https://github.com/vllm-project/vllm/releases/tag/v0.17.1) and supports [Intel® Gaudi® Software v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html) and [Intel® Gaudi® Software v1.24.0](https://docs.habana.ai/en/v1.24.0/Release_Notes/GAUDI_Release_Notes.html).

---

## Highlights

- Added validated support for **Ernie4.5-VL**, **GPT-OSS** (20B/120B), and **reranking models** (Bert-based, Roberta-based, and Qwen3-based).
- Introduced **MxFP4 weight loading and dequantization** support for Gaudi, enabling GPT-OSS model inference.
- Major **Mamba/Granite 4.0-h** improvements including prefix caching support, custom depthwise conv1d TPC kernels, and precision enhancements.
- Enhanced **RowParallel NIC chunking** for better distributed inference performance.
- Added **logprobs** output functionality and **Granite tool calling** accuracy improvements.
- Improved stability through grammar bitmask corruption fixes.

---

## New Model Support

- Add Support for **Ernie4.5-VL** ([#813](https://github.com/vllm-project/vllm-gaudi/pull/813))
- Ported reranking models: Bert-based, Roberta-based and Qwen3-based ([#1001](https://github.com/vllm-project/vllm-gaudi/pull/1001))
- Added support for **GPT-OSS** models (lmsys/gpt-oss-20b-bf16, lmsys/gpt-oss-120b-bf16, openai/gpt-oss-20b, openai/gpt-oss-120b) via MxFP4 weight dequantization ([#1251](https://github.com/vllm-project/vllm-gaudi/pull/1251))
- Enable caching for Qwen3 MoE op ([#1249](https://github.com/vllm-project/vllm-gaudi/pull/1249))

---

## Performance

- Optimization of `selective_state_update` ref in MambaMixer2 decode ([#1244](https://github.com/vllm-project/vllm-gaudi/pull/1244))
- Replacing fancy indexing with select and copy for Granite4 state update ([#1210](https://github.com/vllm-project/vllm-gaudi/pull/1210))
- Creating custom depthwise conv1d kernel for MambaMixer2 ([#1175](https://github.com/vllm-project/vllm-gaudi/pull/1175))
- Improving precision of `_depthwise_conv1d_tpc` for bf16 ([#1203](https://github.com/vllm-project/vllm-gaudi/pull/1203))
- hpu_mamba_chunk_scan_combined_varlen improvements ([#997](https://github.com/vllm-project/vllm-gaudi/pull/997))
- RowParallel NIC chunking ([#896](https://github.com/vllm-project/vllm-gaudi/pull/896))
- Adding `compute_logits` to `_compile_methods` ([#1081](https://github.com/vllm-project/vllm-gaudi/pull/1081))
- Blocking B2BMatmul in dynamic quantization ([#1002](https://github.com/vllm-project/vllm-gaudi/pull/1002))

---

## Attention & KV Cache

- Instead of changing KV cache shape, transpose state in conv1d ([#1025](https://github.com/vllm-project/vllm-gaudi/pull/1025))
- Fix KV cache memory regression from unconditional RowParallelLinear OOT registration ([#1215](https://github.com/vllm-project/vllm-gaudi/pull/1215))
- Prefix caching support for HPUMambaMixer2 ([#1198](https://github.com/vllm-project/vllm-gaudi/pull/1198))

---

## Quantization

- Load and Dequant MxFP4 Weights ([#1251](https://github.com/vllm-project/vllm-gaudi/pull/1251))
- Granite-4.0-h Calibration config ([#1221](https://github.com/vllm-project/vllm-gaudi/pull/1221))
- Force CPU loading for INC quantization to prevent OOM during weight loading ([#1006](https://github.com/vllm-project/vllm-gaudi/pull/1006))
- Fix type mismatch in DeepSeek with fp8_fused_sdpa for mla prefill ([#978](https://github.com/vllm-project/vllm-gaudi/pull/978))

---

## Plugin Core

- Add `num_spec` field to MambaMixer2 for upstream compatibility ([#1142](https://github.com/vllm-project/vllm-gaudi/pull/1142))
- Fix SharedFusedMoE attribute error for Llama4 MoE layers ([#1172](https://github.com/vllm-project/vllm-gaudi/pull/1172))
- Fixing redundant transpose in HPUMambaMixer2 ([#999](https://github.com/vllm-project/vllm-gaudi/pull/999))
- Fix HPUMambaMixer2 inheritance dependency ([#1017](https://github.com/vllm-project/vllm-gaudi/pull/1017))
- Fix mamba cumsum padded calculations ([#1009](https://github.com/vllm-project/vllm-gaudi/pull/1009))
- `use_qk_norm` parameter sourced directly from config ([#972](https://github.com/vllm-project/vllm-gaudi/pull/972))
- Replace mm dummy options ([#1085](https://github.com/vllm-project/vllm-gaudi/pull/1085))
- Logprobs functionality ([#1101](https://github.com/vllm-project/vllm-gaudi/pull/1101))
- Granite accuracy for tool calling ([#1018](https://github.com/vllm-project/vllm-gaudi/pull/1018))
- Add torch inference decorator back to warmup ([#1104](https://github.com/vllm-project/vllm-gaudi/pull/1104))
- Added mechanism for adding events to tlparse ([#1054](https://github.com/vllm-project/vllm-gaudi/pull/1054))
- Fix to gemma3 UT — replaced tuple operation by TC friendly equivalent ([#1083](https://github.com/vllm-project/vllm-gaudi/pull/1083))

---

## Serving & Infrastructure

- Set docker auto calc rules for reserved memory in Torch compile mode ([#1170](https://github.com/vllm-project/vllm-gaudi/pull/1170))
- Improve docker autocalc linear recipe for long contexts ([#959](https://github.com/vllm-project/vllm-gaudi/pull/959))
- Fix Dockerfile for RHEL 9.6 build by updating package installation order ([#1008](https://github.com/vllm-project/vllm-gaudi/pull/1008))
- Install torchaudio from CPU wheel to match PyTorch version in Dockerfile ([#1110](https://github.com/vllm-project/vllm-gaudi/pull/1110))
- Moved inline Dockerfile to a separate file and added torchaudio ([#1050](https://github.com/vllm-project/vllm-gaudi/pull/1050))
- Install torchaudio in CD Dockerfiles ([#1051](https://github.com/vllm-project/vllm-gaudi/pull/1051))
- Add PT_VERSION argument and install torchaudio in Dockerfile ([#1043](https://github.com/vllm-project/vllm-gaudi/pull/1043))
- UBI image: remove pt_fork and duplicated package ([#1066](https://github.com/vllm-project/vllm-gaudi/pull/1066))
- Restore default `temperature=0` for the server after #32723 ([#1039](https://github.com/vllm-project/vllm-gaudi/pull/1039))
- Fix setuptools package discovery to include sub-packages ([#1219](https://github.com/vllm-project/vllm-gaudi/pull/1219))
- Fix `-u` flag requiring argument in `calibrate_model.sh` ([#1167](https://github.com/vllm-project/vllm-gaudi/pull/1167))

---

## Fixes

- Fix OOM crashes during high-concurrency inference ([#1252](https://github.com/vllm-project/vllm-gaudi/pull/1252))
- Fix of Qwen Out of HOST memory (OOM) ([#1256](https://github.com/vllm-project/vllm-gaudi/pull/1256))
- Fix grammar bitmask corruption in mixed structured-output batches ([#1199](https://github.com/vllm-project/vllm-gaudi/pull/1199))
- Granite4.0h fallback bucket padding fix ([#1207](https://github.com/vllm-project/vllm-gaudi/pull/1207))
- Fix prefill bucket mismatch when prefills with no context are padded ([#1064](https://github.com/vllm-project/vllm-gaudi/pull/1064))
- Fix for default max decode blocks in exponential ([#1091](https://github.com/vllm-project/vllm-gaudi/pull/1091))
- Fix import error for MultiModalBudget ([#1062](https://github.com/vllm-project/vllm-gaudi/pull/1062))
- Qwen3-VL WarmUp Fix ([#994](https://github.com/vllm-project/vllm-gaudi/pull/994))
- Server doesn't crash when request is canceled ([#990](https://github.com/vllm-project/vllm-gaudi/pull/990))
- Fix param mismatch for `compute_nixl_compatibility_hash()` ([#1224](https://github.com/vllm-project/vllm-gaudi/pull/1224))

---

## Security

- SDL secure error handling fixes ([#1246](https://github.com/vllm-project/vllm-gaudi/pull/1246))
- Coverity fix including security, null-like values, duplicates and typos ([#1163](https://github.com/vllm-project/vllm-gaudi/pull/1163))

---

## Full Changelog

| PR | Title | Author |
|----|-------|--------|
| [#813](https://github.com/vllm-project/vllm-gaudi/pull/813) | Add Support for Ernie4.5-VL | @jinyouzhi |
| [#896](https://github.com/vllm-project/vllm-gaudi/pull/896) | RowParallel NIC chunking | @kamil-kaczor |
| [#921](https://github.com/vllm-project/vllm-gaudi/pull/921) | Add fp8 calibration tests to CI | @afierka-intel |
| [#959](https://github.com/vllm-project/vllm-gaudi/pull/959) | Improve docker autocalc linear recipe for long contexts | @nngokhale |
| [#967](https://github.com/vllm-project/vllm-gaudi/pull/967) | Add ci test for granite-4-h-small | @microslaw |
| [#972](https://github.com/vllm-project/vllm-gaudi/pull/972) | use_qk_norm parameter sourced directly from config | @rsmyrek |
| [#978](https://github.com/vllm-project/vllm-gaudi/pull/978) | Fix type mismatch in DeepSeek with fp8_fused_sdpa for mla prefill | @skavulya |
| [#990](https://github.com/vllm-project/vllm-gaudi/pull/990) | Server doesn't crash when request is canceled | @tzielinski-habana |
| [#994](https://github.com/vllm-project/vllm-gaudi/pull/994) | Qwen3-VL WarmUp Fix | @slokesha |
| [#995](https://github.com/vllm-project/vllm-gaudi/pull/995) | Add sleep mode model swapping test for Gaudi | @PatrykWo |
| [#997](https://github.com/vllm-project/vllm-gaudi/pull/997) | hpu_mamba_chunk_scan_combined_varlen improvements | @PatrykWilczewski |
| [#998](https://github.com/vllm-project/vllm-gaudi/pull/998) | Hourly fixes – batch no. 2 | @pawel-olejniczak |
| [#999](https://github.com/vllm-project/vllm-gaudi/pull/999) | Fixing redundant transpose in HPUMambaMixer2 | @ksmusz |
| [#1001](https://github.com/vllm-project/vllm-gaudi/pull/1001) | Ported the reranking models: Bert-based, Roberta-based and Qwen3-based | @gyou2021 |
| [#1002](https://github.com/vllm-project/vllm-gaudi/pull/1002) | Blocking B2BMatmul in dynamic quantization | @HolyFalafel |
| [#1006](https://github.com/vllm-project/vllm-gaudi/pull/1006) | Force CPU loading for INC quantization to prevent OOM during weight loading | @agrabow |
| [#1008](https://github.com/vllm-project/vllm-gaudi/pull/1008) | Fix Dockerfile for RHEL 9.6 build by updating package installation order | @PatrykWo |
| [#1009](https://github.com/vllm-project/vllm-gaudi/pull/1009) | Fix mamba cumsum padded calculations | @jkaniecki |
| [#1017](https://github.com/vllm-project/vllm-gaudi/pull/1017) | Fix HPUMambaMixer2 inheritance dependency | @jbyczkow |
| [#1018](https://github.com/vllm-project/vllm-gaudi/pull/1018) | Granite accuracy for tool calling | @adobrzyn |
| [#1025](https://github.com/vllm-project/vllm-gaudi/pull/1025) | Instead of changing kv cache shape, transpose state in conv1d | @jmamzax |
| [#1031](https://github.com/vllm-project/vllm-gaudi/pull/1031) | Change Qwen3VL to use HPUMMEncoderAttention | @jiminha |
| [#1039](https://github.com/vllm-project/vllm-gaudi/pull/1039) | Back temperature=0 for server as default after #32723 | @iboiko-habana |
| [#1043](https://github.com/vllm-project/vllm-gaudi/pull/1043) | Add PT_VERSION argument and install torchaudio in Dockerfile | @PatrykWo |
| [#1050](https://github.com/vllm-project/vllm-gaudi/pull/1050) | Moved inline Dockerfile to a separate file and added torchaudio | @tzielinski-habana |
| [#1051](https://github.com/vllm-project/vllm-gaudi/pull/1051) | Install torchaudio in CD Dockerfiles | @tzielinski-habana |
| [#1053](https://github.com/vllm-project/vllm-gaudi/pull/1053) | Hourly fixes – batch no. 3 | @pawel-olejniczak |
| [#1054](https://github.com/vllm-project/vllm-gaudi/pull/1054) | Added mechanism for adding events to tlparse | @jczaja |
| [#1062](https://github.com/vllm-project/vllm-gaudi/pull/1062) | Fix import error for MultiModalBudget | @tvoas |
| [#1064](https://github.com/vllm-project/vllm-gaudi/pull/1064) | Fix prefill bucket mismatch when prefills with no context are padded | @mfylcek |
| [#1066](https://github.com/vllm-project/vllm-gaudi/pull/1066) | UBI image: remove pt_fork and duplicated package | @ghandoura |
| [#1067](https://github.com/vllm-project/vllm-gaudi/pull/1067) | Add workflow to update VLLM_COMMUNITY_COMMIT via GitHub Actions | @PatrykWo |
| [#1081](https://github.com/vllm-project/vllm-gaudi/pull/1081) | Adding compute_logits to _compile_methods | @ksmusz |
| [#1083](https://github.com/vllm-project/vllm-gaudi/pull/1083) | Fix to gemma3 UT — replaced tuple operation by TC friendly equivalent | @jczaja |
| [#1085](https://github.com/vllm-project/vllm-gaudi/pull/1085) | Replace mm dummy options | @skaulintel |
| [#1090](https://github.com/vllm-project/vllm-gaudi/pull/1090) | Fix for MoE refactor #32344 | @iboiko-habana |
| [#1091](https://github.com/vllm-project/vllm-gaudi/pull/1091) | Fix for default max decode blocks in exponential | @adobrzyn |
| [#1101](https://github.com/vllm-project/vllm-gaudi/pull/1101) | Logprobs functionality | @adobrzyn |
| [#1104](https://github.com/vllm-project/vllm-gaudi/pull/1104) | Add torch inference decorator back to warmup | @skaulintel |
| [#1108](https://github.com/vllm-project/vllm-gaudi/pull/1108) | Hourly fixes, part 3 | @iboiko-habana |
| [#1110](https://github.com/vllm-project/vllm-gaudi/pull/1110) | Install torchaudio from CPU wheel to match PyTorch version in Dockerfile | @PatrykWo |
| [#1114](https://github.com/vllm-project/vllm-gaudi/pull/1114) | Hourly fixes, part 4 | @iboiko-habana |
| [#1115](https://github.com/vllm-project/vllm-gaudi/pull/1115) | Fix for vLLM #35503 | @iboiko-habana |
| [#1116](https://github.com/vllm-project/vllm-gaudi/pull/1116) | Fix for vLLM #35503 | @iboiko-habana |
| [#1125](https://github.com/vllm-project/vllm-gaudi/pull/1125) | Cherry from 0.16.0 release | @PatrykWo |
| [#1142](https://github.com/vllm-project/vllm-gaudi/pull/1142) | Add num_spec field to MambaMixer2 for upstream compatibility | @jbyczkow |
| [#1163](https://github.com/vllm-project/vllm-gaudi/pull/1163) | Coverity fix including security, null-like values, duplicates and typos | @adobrzyn |
| [#1167](https://github.com/vllm-project/vllm-gaudi/pull/1167) | Fix -u flag requiring argument in calibrate_model.sh | @adobrzyn |
| [#1170](https://github.com/vllm-project/vllm-gaudi/pull/1170) | Set docker auto calc rules for reserved memory in Torch compile mode | @nngokhale |
| [#1172](https://github.com/vllm-project/vllm-gaudi/pull/1172) | Fix SharedFusedMoE attribute error for Llama4 MoE layers | @adobrzyn |
| [#1175](https://github.com/vllm-project/vllm-gaudi/pull/1175) | Creating custom depthwise conv1d kernel for MambaMixer2 | @ksmusz |
| [#1178](https://github.com/vllm-project/vllm-gaudi/pull/1178) | Update quickstart guide and supported model list | @PatrykWo |
| [#1198](https://github.com/vllm-project/vllm-gaudi/pull/1198) | Prefix caching support for HPUMambaMixer2 | @jbyczkow |
| [#1199](https://github.com/vllm-project/vllm-gaudi/pull/1199) | Fix grammar bitmask corruption in mixed structured-output batches | @jbyczkow |
| [#1203](https://github.com/vllm-project/vllm-gaudi/pull/1203) | Improving precision of _depthwise_conv1d_tpc for bf16 | @ksmusz |
| [#1207](https://github.com/vllm-project/vllm-gaudi/pull/1207) | Granite4.0h fallback bucket padding fix | @mfylcek |
| [#1210](https://github.com/vllm-project/vllm-gaudi/pull/1210) | Replacing fancy indexing with select and copy for Granite4 state update | @ksmusz |
| [#1215](https://github.com/vllm-project/vllm-gaudi/pull/1215) | Fix KV cache memory regression from unconditional RowParallelLinear OOT registration | @kamil-kaczor |
| [#1219](https://github.com/vllm-project/vllm-gaudi/pull/1219) | Fix setuptools package discovery to include sub-packages | @app/copilot-swe-agent |
| [#1221](https://github.com/vllm-project/vllm-gaudi/pull/1221) | Granite-4.0-h Calibration config | @mfylcek |
| [#1224](https://github.com/vllm-project/vllm-gaudi/pull/1224) | Fix param mismatch for compute_nixl_compatibility_hash | @hsubramony |
| [#1228](https://github.com/vllm-project/vllm-gaudi/pull/1228) | Add ci test for granite-4-h-small to 0.17.1 | @microslaw |
| [#1244](https://github.com/vllm-project/vllm-gaudi/pull/1244) | Optimization of selective_state_update ref in MambaMixer2 decode | @ksmusz |
| [#1246](https://github.com/vllm-project/vllm-gaudi/pull/1246) | SDL secure error handling fixes | @adobrzyn |
| [#1249](https://github.com/vllm-project/vllm-gaudi/pull/1249) | Enable caching for qwen3 moe op | @shepark |
| [#1251](https://github.com/vllm-project/vllm-gaudi/pull/1251) | Load and Dequant MxFP4 Weights | @SKRohit |
| [#1252](https://github.com/vllm-project/vllm-gaudi/pull/1252) | Fix OOM crashes during high-concurrency inference | @afierka-intel |
| [#1256](https://github.com/vllm-project/vllm-gaudi/pull/1256) | Fix of Qwen Out of HOST memory (OOM) | @iboiko-habana |

---

## New Contributors

Welcome to the following first-time contributors to vLLM Gaudi Plugin!

- **@gyou2021** — Ported reranking models: Bert-based, Roberta-based and Qwen3-based ([#1001](https://github.com/vllm-project/vllm-gaudi/pull/1001))
- **@jczaja** — Added mechanism for adding events to tlparse ([#1054](https://github.com/vllm-project/vllm-gaudi/pull/1054))
- **@jinyouzhi** — Add Support for Ernie4.5-VL ([#813](https://github.com/vllm-project/vllm-gaudi/pull/813))
- **@mfylcek** — Granite4.0h fallback bucket padding fix ([#1207](https://github.com/vllm-project/vllm-gaudi/pull/1207))
- **@pawel-olejniczak** — Hourly upstream compatibility fixes ([#998](https://github.com/vllm-project/vllm-gaudi/pull/998))
- **@skaulintel** — Replace mm dummy options and warmup improvements ([#1085](https://github.com/vllm-project/vllm-gaudi/pull/1085))
