# vLLM Gaudi Plugin v0.16.0 Release Notes

## Overview

This release is based on [vLLM v0.16.0](https://github.com/vllm-project/vllm/releases/tag/v0.16.0) and supports [Intel® Gaudi® Software v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html).

## Highlights

- Added validated support for the following models: Qwen3-VL, DeepSeek OCR, MiniMax-M2, Ovis, Mistral-Large-3, and Hunyuan V1.
- Improved performance by introducing backported bug fixes, mamba improvements, and model weight loading speeds.
- Enhanced quantization to force CPU loading for INC quantization to prevent OOM.
- Introduced various improvements to UBI/RHEL Docker images, server defaults, and Coverity fixes.

## New Model Support and Updates

- Change Qwen3-VL to use HPUMMEncoderAttention ([#1060](https://github.com/vllm-project/vllm-gaudi/pull/1060))
- Enable caching for Qwen3 MoE op ([#1068](https://github.com/vllm-project/vllm-gaudi/pull/1068))
- Fix Qwen3-VL MoE execution failure ([#1028](https://github.com/vllm-project/vllm-gaudi/pull/1028))
- Enable DeepSeek OCR model ([#954](https://github.com/vllm-project/vllm-gaudi/pull/954))
- Add dotsocr and seedoss ([#977](https://github.com/vllm-project/vllm-gaudi/pull/977))
- Add MiniMax-M2 support ([#964](https://github.com/vllm-project/vllm-gaudi/pull/964))
- Add Ovis model support with default buckets ([#846](https://github.com/vllm-project/vllm-gaudi/pull/846))
- Enable Mistral-Large-3-675B-Instruct-2512 model ([#871](https://github.com/vllm-project/vllm-gaudi/pull/871))
- Add Hunyuan V1 model support (Dense & MoE bf16/FP8) ([#875](https://github.com/vllm-project/vllm-gaudi/pull/875))

## Performance

- [GAUDISW-246429] hpu_mamba_chunk_scan_combined_varlen improvements ([#1074](https://github.com/vllm-project/vllm-gaudi/pull/1074))
- Improve model weight loading speed ([#807](https://github.com/vllm-project/vllm-gaudi/pull/807))
- Fix warmup regression ([#962](https://github.com/vllm-project/vllm-gaudi/pull/962))

## Attention and KV Cache

- Instead of changing KV cache shape, transpose state in conv1d ([#1065](https://github.com/vllm-project/vllm-gaudi/pull/1065))
- [GAUDISW-245713] Remove bucket densification for long ctx; Edge buckets only for long ctx ([#915](https://github.com/vllm-project/vllm-gaudi/pull/915))
- Temporarily disable chunked attention ([#981](https://github.com/vllm-project/vllm-gaudi/pull/981))
- Multimodal model embedding fixes ([#759](https://github.com/vllm-project/vllm-gaudi/pull/759))
- [CT] Add FP8 GQA Support ([#874](https://github.com/vllm-project/vllm-gaudi/pull/874))
- [CT] Fix CT Config to honor `fp8_inc` KV cache dtype ([#929](https://github.com/vllm-project/vllm-gaudi/pull/929))

## Quantization

- Force CPU loading for INC quantization to prevent OOM during weight loading ([#1055](https://github.com/vllm-project/vllm-gaudi/pull/1055))
- Fix INC patching `_gate` twice ([#955](https://github.com/vllm-project/vllm-gaudi/pull/955))
- [GAUDISW-246337] Added config with scale method: maxabs_pcs_pow2 for dynamic quant ([#949](https://github.com/vllm-project/vllm-gaudi/pull/949))

## Plugin Core

- Source `use_qk_norm` parameter directly from config ([#1084](https://github.com/vllm-project/vllm-gaudi/pull/1084))
- Fix last_chunk_indices calculations ([#1023](https://github.com/vllm-project/vllm-gaudi/pull/1023))
- Fix mamba cumsum padded calculations ([#1021](https://github.com/vllm-project/vllm-gaudi/pull/1021))
- Fix redundant transpose in HPUMambaMixer2 ([#1015](https://github.com/vllm-project/vllm-gaudi/pull/1015))
- Fix HPUMambaMixer2 inheritance dependency ([#1016](https://github.com/vllm-project/vllm-gaudi/pull/1016))
- Add _MAMBA_PAD_BLOCK_ID ([#951](https://github.com/vllm-project/vllm-gaudi/pull/951))
- Enable OffloadingConnector on HPU. ([#827](https://github.com/vllm-project/vllm-gaudi/pull/827))
- GPT OSS Integration Code ([#887](https://github.com/vllm-project/vllm-gaudi/pull/887))
- Fix async scheduler + unified attention failure on Qwen2.5-VL ([#931](https://github.com/vllm-project/vllm-gaudi/pull/931))
- Fix undefined behavior in copy_blocks when source and destination blocks overlap ([#329](https://github.com/vllm-project/vllm-gaudi/pull/329))

## Serving and Infrastructure

- Fix RHEL Dockerfile build order and remove obsolete TencentOS Dockerfile ([#1056](https://github.com/vllm-project/vllm-gaudi/pull/1056))
- Improve Docker autocalc linear recipe for long contexts (cherry-pick to 0.16.0) ([#1041](https://github.com/vllm-project/vllm-gaudi/pull/1041))
- Add `libfdt-devel` to UBI Dockerfile ([#974](https://github.com/vllm-project/vllm-gaudi/pull/974))
- Fix device detection when ENABLE_CONSOLE=true ([#963](https://github.com/vllm-project/vllm-gaudi/pull/963))

## Fixes

- Don't destroy server with logprobs ([#1098](https://github.com/vllm-project/vllm-gaudi/pull/1098))
- Coverity fix including security, null-like values, duplicates and typos ([#1094](https://github.com/vllm-project/vllm-gaudi/pull/1094))
- Fix param mismatch for `compute_nixl_compatibility_hash()` ([#1087](https://github.com/vllm-project/vllm-gaudi/pull/1087))
- Fix Topk Calculation in GPTOSS ([#970](https://github.com/vllm-project/vllm-gaudi/pull/970))
- Fix reported version of vLLM ([#811](https://github.com/vllm-project/vllm-gaudi/pull/811))
- Fixing _compile_region for nested attributes ([#956](https://github.com/vllm-project/vllm-gaudi/pull/956))
- Fix sampler & TP>1 recompilations ([#935](https://github.com/vllm-project/vllm-gaudi/pull/935))
- Restore default `temperature=0` for the server after #32723 ([#1037](https://github.com/vllm-project/vllm-gaudi/pull/1037))

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1098](https://github.com/vllm-project/vllm-gaudi/pull/1098) | Don't destroy server with logprobs | @adobrzyn |
| [#1094](https://github.com/vllm-project/vllm-gaudi/pull/1094) | Coverity fix including security, null-like values, duplicates and typos | @adobrzyn |
| [#1087](https://github.com/vllm-project/vllm-gaudi/pull/1087) | fix param mismatch for compute_nixl_compatibility_hash() | @hsubramony |
| [#1060](https://github.com/vllm-project/vllm-gaudi/pull/1060) | Change Qwen3VL to use HPUMMEncoderAttention | @jiminha |
| [#1068](https://github.com/vllm-project/vllm-gaudi/pull/1068) | Enable caching for qwen3 moe op | @shepark |
| [#1084](https://github.com/vllm-project/vllm-gaudi/pull/1084) | use_qk_norm parameter sourced directly from config | @rsmyrek |
| [#1056](https://github.com/vllm-project/vllm-gaudi/pull/1056) | Fix RHEL Dockerfile build order and remove obsolete TencentOS Dockerfile | @PatrykWo |
| [#1037](https://github.com/vllm-project/vllm-gaudi/pull/1037) | Back temperature=0 for server as default after #32723 | @iboiko-habana |
| [#1089](https://github.com/vllm-project/vllm-gaudi/pull/1089) | Change upstream last_good_commit 89a77b10846fd96273cce78d86d2556ea582d26e | @iboiko-habana |
| [#1041](https://github.com/vllm-project/vllm-gaudi/pull/1041) | Improve docker autocalc linear recipe for long contexts (cherry-pick to 0.16.0) | @nngokhale |
| [#1080](https://github.com/vllm-project/vllm-gaudi/pull/1080) | Port of #1050 for CI unblocking | @iboiko-habana |
| [#1074](https://github.com/vllm-project/vllm-gaudi/pull/1074) | hpu_mamba_chunk_scan_combined_varlen improvements | @PatrykWilczewski |
| [#1057](https://github.com/vllm-project/vllm-gaudi/pull/1057) | Add ci test for granite-4-h-small to v0.16.0 | @microslaw |
| [#1065](https://github.com/vllm-project/vllm-gaudi/pull/1065) | Instead of changing kv cache shape, transpose state in conv1d | @jmamzax |
| [#1023](https://github.com/vllm-project/vllm-gaudi/pull/1023) | Fix last_chunk_indices calculations | @jbyczkow |
| [#1021](https://github.com/vllm-project/vllm-gaudi/pull/1021) | Fix mamba cumsum padded calculations | @jkaniecki |
| [#999](https://github.com/vllm-project/vllm-gaudi/pull/999) | Fix redundant transpose in HPUMambaMixer2  (#1015) | @ksmusz |
| [#1019](https://github.com/vllm-project/vllm-gaudi/pull/1019) | Fixes for #33559 and #34103 | @iboiko-habana |
| [#1055](https://github.com/vllm-project/vllm-gaudi/pull/1055) | Force CPU loading for INC quantization to prevent OOM during weight loading | @agrabow |
| [#1016](https://github.com/vllm-project/vllm-gaudi/pull/1016) | Fix HPUMambaMixer2 inheritance dependency | @jbyczkow |
| [#1028](https://github.com/vllm-project/vllm-gaudi/pull/1028) | Fix qwen3 vl moe execution failure | @shepark |
| [#1042](https://github.com/vllm-project/vllm-gaudi/pull/1042) | Adding ci_calibration_smoke_tests.sh into v0.16.0 | @iboiko-habana |
| [#971](https://github.com/vllm-project/vllm-gaudi/pull/971) | UBI images improvements | @ghandoura |
| [#954](https://github.com/vllm-project/vllm-gaudi/pull/954) | Enable deepseek ocr model | @HeJunyan |
| [#977](https://github.com/vllm-project/vllm-gaudi/pull/977) | Add dotsocr and seedoss | @tianyuan211 |
| [#975](https://github.com/vllm-project/vllm-gaudi/pull/975) | Monkey-patch of Attention.forward | @tzielinski-habana |
| [#824](https://github.com/vllm-project/vllm-gaudi/pull/824) | Adjust pre-merge workflow to support merge queue trigger event | @bmyrcha |
| [#970](https://github.com/vllm-project/vllm-gaudi/pull/970) | Fix Topk Calculation in GPTOSS | @SKRohit |
| [#981](https://github.com/vllm-project/vllm-gaudi/pull/981) | Temporarily disable chunked attention | @adobrzyn |
| [#982](https://github.com/vllm-project/vllm-gaudi/pull/982) | adding FIX_FOR_VLLM_CUSTOM to CI | @iboiko-habana |
| [#974](https://github.com/vllm-project/vllm-gaudi/pull/974) | Add libfdt-devel (new habanalabs-thunk dependency) to ubi dockerfile | @mmuszynskihabana |
| [#930](https://github.com/vllm-project/vllm-gaudi/pull/930) | Fix for individual unit tests | @tzielinski-habana |
| [#969](https://github.com/vllm-project/vllm-gaudi/pull/969) | CI cleanup 2 | @microslaw |
| [#964](https://github.com/vllm-project/vllm-gaudi/pull/964) | Add MiniMax-M2 support | @testdig |
| [#846](https://github.com/vllm-project/vllm-gaudi/pull/846) | Add ovis models support with default buckets | @testdig |
| [#713](https://github.com/vllm-project/vllm-gaudi/pull/713) | Create UBI based vLLM docker build instructions | @ghandoura |
| [#811](https://github.com/vllm-project/vllm-gaudi/pull/811) | Fix reported version of vllm | @ghandoura |
| [#960](https://github.com/vllm-project/vllm-gaudi/pull/960) | Add docker image cleanup at the end of workflows | @bmyrcha |
| [#962](https://github.com/vllm-project/vllm-gaudi/pull/962) | Fix warmup regression | @kamil-kaczor |
| [#965](https://github.com/vllm-project/vllm-gaudi/pull/965) | Add hf_transfer to test requirements | @bmyrcha |
| [#955](https://github.com/vllm-project/vllm-gaudi/pull/955) | Fix INC patching _gate twice | @kamil-kaczor |
| [#933](https://github.com/vllm-project/vllm-gaudi/pull/933) | CI cleanup | @microslaw |
| [#963](https://github.com/vllm-project/vllm-gaudi/pull/963) | Fix device detection when ENABLE_CONSOLE=true | @afierka-intel |
| [#956](https://github.com/vllm-project/vllm-gaudi/pull/956) | Fixing _compile_region for nested attributes | @ksmusz |
| [#871](https://github.com/vllm-project/vllm-gaudi/pull/871) | Enable Mistral-Large-3-675B-Instruct-2512 model | @skavulya |
| [#915](https://github.com/vllm-project/vllm-gaudi/pull/915) | Remove bucket densification for long ctx; Edge buckets only for long ctx | @kfojcik-intel |
| [#723](https://github.com/vllm-project/vllm-gaudi/pull/723) | Dryrun implementation for generating command line file | @rajanintel24 |
| [#759](https://github.com/vllm-project/vllm-gaudi/pull/759) | Multimodal model embedding fixes | @libinta |
| [#329](https://github.com/vllm-project/vllm-gaudi/pull/329) | Fix undefined behavior in copy_blocks when source and destination blocks overlap | @yafshar |
| [#949](https://github.com/vllm-project/vllm-gaudi/pull/949) | Added config with scale method: maxabs_pcs_pow2 for dynamic quant | @HolyFalafel |
| [#951](https://github.com/vllm-project/vllm-gaudi/pull/951) | Add _MAMBA_PAD_BLOCK_ID | @jbyczkow |
| [#875](https://github.com/vllm-project/vllm-gaudi/pull/875) | Add Hunyuan V1 model support (Dense & MoE bf16/FP8) | @jjmiao1 |
| [#887](https://github.com/vllm-project/vllm-gaudi/pull/887) | GPT OSS Integration Code | @hlahkar |
| [#916](https://github.com/vllm-project/vllm-gaudi/pull/916) | Port: Initialization profiling noop  (#932) | @michalkuligowski |
| [#941](https://github.com/vllm-project/vllm-gaudi/pull/941) | Port profile run off #916 | @adobrzyn |
| [#931](https://github.com/vllm-project/vllm-gaudi/pull/931) | Fix async scheduler + unified attention failure on Qwen2.5-VL | @tvoas |
| [#662](https://github.com/vllm-project/vllm-gaudi/pull/662) | Add local path option for hf_cache | @PatrykWo |
| [#940](https://github.com/vllm-project/vllm-gaudi/pull/940) | Missing updates for Llama4 on main | @Luca-Calabria |
| [#902](https://github.com/vllm-project/vllm-gaudi/pull/902) | Add unit tests for multimodal inputs classes | @microslaw |
| [#827](https://github.com/vllm-project/vllm-gaudi/pull/827) | Enable OffloadingConnector on HPU. | @yeonsily |
| [#788](https://github.com/vllm-project/vllm-gaudi/pull/788) | Set device according to local rank | @yangulei |
| [#889](https://github.com/vllm-project/vllm-gaudi/pull/889) | Adapt OnlineDefragmenter and CacheSwapUtils for torc… | @jwieczorekhabana |
| [#923](https://github.com/vllm-project/vllm-gaudi/pull/923) | Modify ubi docker to support both internal and external builds | @mmuszynskihabana |
| [#944](https://github.com/vllm-project/vllm-gaudi/pull/944) | New testowners | @adobrzyn |
| [#893](https://github.com/vllm-project/vllm-gaudi/pull/893) | Fix torch.compile crash in sampler by removing NumPy dependency in tensor padding | @tvoas |
| [#874](https://github.com/vllm-project/vllm-gaudi/pull/874) | [CT] Add FP8 GQA Support | @yiliu30 |
| [#807](https://github.com/vllm-project/vllm-gaudi/pull/807) | Improve model weight loading speed | @yupengzh-intel |
| [#935](https://github.com/vllm-project/vllm-gaudi/pull/935) | Fix sampler & TP>1 recompilations | @kamil-kaczor |
| [#929](https://github.com/vllm-project/vllm-gaudi/pull/929) | [CT] Fix CT Config to honor `fp8_inc` KV cache dtype | @yiliu30 |

## New Contributors

Welcome to the first-time contributors to the vllm-gaudi plugin!

- @agrabowski [#1055](https://github.com/vllm-project/vllm-gaudi/pull/1055) 'Force CPU loading for INC quantization to prevent OOM during weight loading'
