# vLLM Gaudi Plugin v0.33.0 Release Notes

## Overview

This release is based on [vLLM v0.33.0](https://github.com/vllm-project/vllm/releases/tag/v0.33.0) and supports [Intel® Gaudi® Software v1.24.3](https://docs.habana.ai/en/v1.24.3/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.14.1.

## Highlights

- Enabled the plugin on upstream [vLLM v0.33.0](https://github.com/vllm-project/vllm/releases/tag/v0.33.0) and upgraded the platform to Intel® Gaudi® Software v1.24.3 with PyTorch 2.14.1.
- Added an opt-in native int4 WNA16 MoE path for compressed-tensors models, including non-gated experts for Nemotron-H.
- Reduced the DeepSeek Sparse Attention decode indexer cost with per-request top-k selection.
- Validated Phi-4, Phi-4-mini, Nemotron-3-Ultra W4A16, and Granite 4.1 8B, with a LongRoPE rotary embedding fix for Phi-3 and Phi-4-mini.
- Fixed output correctness for Qwen2.5-VL image inputs, Gemma 4 vision with transformers 5.19 and later, Granite 4.0-H, FP8 KV cache decode on Gaudi 2, sliced FusedSDPA, GDN prefill, structured output sampling, and sliding-window prefill with cached context.
- Released host memory on worker shutdown and guarded sleep mode against host OOM in multi-model swap flows.
- Added Ubuntu 22.04 with Python 3.11 Dockerfiles and restored the tcmalloc preload in the RHEL UBI Dockerfiles.

---

## New Model Support

- Enabled Nemotron-H WNA16 on HPU with non-gated MoE experts on the native int4 kernel. ([#1808](https://github.com/vllm-project/vllm-gaudi/pull/1808), [#1874](https://github.com/vllm-project/vllm-gaudi/pull/1874))
- Fixed the HPU LongRoPE rotary embedding for Phi-3 and Phi-4-mini, and validated Phi-4 and Phi-4-mini. ([#1820](https://github.com/vllm-project/vllm-gaudi/pull/1820), [#1821](https://github.com/vllm-project/vllm-gaudi/pull/1821))
- Validated Nemotron-3-Ultra W4A16 (`RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-quantized.w4a16`) on 4 and 8 cards, and Granite 4.1 8B in BF16 on one card.

---

## Performance

- Added the opt-in `VLLM_WNA16_NATIVE_INT4_MOE` environment variable to run compressed-tensors WNA16 MoE experts on the native int4 fused-weights kernel instead of dequantizing on every forward pass. ([#1808](https://github.com/vllm-project/vllm-gaudi/pull/1808), [#1874](https://github.com/vllm-project/vllm-gaudi/pull/1874))
- Ranked each request's own KV blocks in the DeepSeek Sparse Attention decode indexer, so its top-k cost no longer grows with the square of the batch size. ([#1870](https://github.com/vllm-project/vllm-gaudi/pull/1870))
- Built the selective sampling metadata without synchronous host-to-device copies on every decode step. ([#1911](https://github.com/vllm-project/vllm-gaudi/pull/1911))
- Removed two `torch.compile` specializations in the Nemotron-H Mamba path, allowing `VLLM_T_COMPILE_FULLGRAPH=true`, and avoided an eager fallback in the prefix-caching `conv1d` gather. ([#1872](https://github.com/vllm-project/vllm-gaudi/pull/1872), [#1873](https://github.com/vllm-project/vllm-gaudi/pull/1873))
- Compiled pooling (embedding and reranker) model buckets during warmup instead of on the first requests. ([#1883](https://github.com/vllm-project/vllm-gaudi/pull/1883))
- Restored the HPU block size of 128 after the upstream multi-backend default block size selection. ([#1871](https://github.com/vllm-project/vllm-gaudi/pull/1871))

---

## Attention and KV Cache

- Added the `VLLM_PREFER_128_KERNEL_BLOCKS` environment variable to keep 128-token attention kernel blocks for Nemotron-H whenever they fit, and kept the prompt context when the attention kernel block is larger than the bucketing unit. ([#1987](https://github.com/vllm-project/vllm-gaudi/pull/1987), [#1986](https://github.com/vllm-project/vllm-gaudi/pull/1986))
- Fixed pad block and slot IDs pointing into a real KV block for Mamba-like models. ([#1962](https://github.com/vllm-project/vllm-gaudi/pull/1962))
- Fixed corrupted batched decode with an FP8 KV cache (`kv_cache_dtype=fp8_inc`) on Gaudi 2. ([#1931](https://github.com/vllm-project/vllm-gaudi/pull/1931))
- Fixed NaN outputs in sliced FusedSDPA (`VLLM_HPU_FSDPA_SLICE_ENABLED=true`) for fully masked chunks. ([#1906](https://github.com/vllm-project/vllm-gaudi/pull/1906))
- Fixed sliding-window prefill attention with cached context for non-interleaved sliding-window models such as Phi-4-mini. ([#1850](https://github.com/vllm-project/vllm-gaudi/pull/1850))
- Fixed non-finite initial-state masking in GDN prefill, which could contaminate outputs and cached state. ([#1895](https://github.com/vllm-project/vllm-gaudi/pull/1895))
- Fixed an engine crash when a Qwen3.5/GDN prefill resumes off the Mamba chunk grid with prefix caching. ([#1890](https://github.com/vllm-project/vllm-gaudi/pull/1890))
- Fixed a non-reproducible RoPE table built on the CPU. ([#1954](https://github.com/vllm-project/vllm-gaudi/pull/1954))

---

## Quantization

- Fixed INC FP8 skipping MoE routers after upstream vLLM moved them to `GateLinear`. ([#1926](https://github.com/vllm-project/vllm-gaudi/pull/1926))
- Rebound the speculative drafter's WNA16 MoE expert views after moving it to HPU. ([#1963](https://github.com/vllm-project/vllm-gaudi/pull/1963))

---

## Plugin Core

- Tracked upstream vLLM changes: MoE router factory `skip_padding` arguments, offloading-connector recency tracking, read-only executor `is_sleeping`, `apply_lora_full_linear` on `PunicaWrapperHPU`, multimodal input support detection, relocated chat and CLI helpers, KV cache planning imports, rejection sampler arguments, block-table kernel block splitting, and `get_kv_connector_output` for asynchronous runner output. ([#1806](https://github.com/vllm-project/vllm-gaudi/pull/1806), [#1810](https://github.com/vllm-project/vllm-gaudi/pull/1810), [#1880](https://github.com/vllm-project/vllm-gaudi/pull/1880), [#1900](https://github.com/vllm-project/vllm-gaudi/pull/1900), [#1910](https://github.com/vllm-project/vllm-gaudi/pull/1910), [#1914](https://github.com/vllm-project/vllm-gaudi/pull/1914), [#1927](https://github.com/vllm-project/vllm-gaudi/pull/1927))
- Fixed the multimodal warmup fallback after upstream moved `get_dummy_mm_inputs` to the multimodal processor. ([#1878](https://github.com/vllm-project/vllm-gaudi/pull/1878))
- Matched the upstream GemmaRMSNorm precision on HPU by preserving FP32 intermediates. ([#1949](https://github.com/vllm-project/vllm-gaudi/pull/1949))
- Restored the split Mamba prefill graph for Granite 4.0-H through the `VLLM_MAMBA_SPLIT_PREFILL_GRAPH` environment variable, enabled by default for Granite 4.0-H. ([#1947](https://github.com/vllm-project/vllm-gaudi/pull/1947))

---

## Serving and Infrastructure

- Added Dockerfile support for Ubuntu 22.04 with Python 3.11. ([#1979](https://github.com/vllm-project/vllm-gaudi/pull/1979))
- Restored the tcmalloc preload in the RHEL UBI Dockerfiles, fixing a low-batch decode throughput regression. ([#1909](https://github.com/vllm-project/vllm-gaudi/pull/1909))
- Released host memory on worker shutdown and added a host-memory guard for sleep mode (`VLLM_GAUDI_SLEEP_HOST_RESERVE_MB`, `VLLM_GAUDI_SKIP_SLEEP_HOST_GUARD`), preventing host OOM in multi-model swap flows. ([#1844](https://github.com/vllm-project/vllm-gaudi/pull/1844))
- Built NIXL 1.5.0 in `install_nixl.py`, as required by upstream vLLM for KV transfer. ([#1900](https://github.com/vllm-project/vllm-gaudi/pull/1900))
- Fixed the text serving benchmarks after the upstream removal of the `sonnet` dataset. ([#1901](https://github.com/vllm-project/vllm-gaudi/pull/1901))
- Updated dependency requirements: `transformers<5.21`, `fastapi<0.163`, `numba>=0.68.1`, and `kaldi-native-fbank>=1.23.0`. ([#1799](https://github.com/vllm-project/vllm-gaudi/pull/1799), [#1858](https://github.com/vllm-project/vllm-gaudi/pull/1858), [#1887](https://github.com/vllm-project/vllm-gaudi/pull/1887), [#1935](https://github.com/vllm-project/vllm-gaudi/pull/1935), [#1856](https://github.com/vllm-project/vllm-gaudi/pull/1856), [#1904](https://github.com/vllm-project/vllm-gaudi/pull/1904), [#1933](https://github.com/vllm-project/vllm-gaudi/pull/1933), [#1857](https://github.com/vllm-project/vllm-gaudi/pull/1857), [#1983](https://github.com/vllm-project/vllm-gaudi/pull/1983), [#1934](https://github.com/vllm-project/vllm-gaudi/pull/1934), [#1980](https://github.com/vllm-project/vllm-gaudi/pull/1980))

---

## Fixes

- Fixed Qwen2.5-VL ignoring image content and an `mmmu_val` crash on HPU. ([#1852](https://github.com/vllm-project/vllm-gaudi/pull/1852))
- Matched processor-side BF16 rounding in HPU device image normalization (`--mm-device-do-normalize`) for `uint8` image inputs, which could otherwise change vision embeddings and generated text. ([#1994](https://github.com/vllm-project/vllm-gaudi/pull/1994))
- Restored the FP32 matmul in the Gemma 4 vision rotary embedding on HPU, fixing an OCR accuracy regression (OmniDocBench) with transformers 5.19 and later. ([#2000](https://github.com/vllm-project/vllm-gaudi/pull/2000))
- Fixed a zeroed convolution state in the HPU graph for Granite 4.0-H, which broke multi-turn tool calling. ([#1915](https://github.com/vllm-project/vllm-gaudi/pull/1915))
- Sampled structured output rows with their own request parameters, and re-keyed `bad_words` to the rows of the selective sampling metadata. ([#1916](https://github.com/vllm-project/vllm-gaudi/pull/1916), [#1917](https://github.com/vllm-project/vllm-gaudi/pull/1917))
- Restored the batch dimension of prefill hidden states for prompt logprobs, preventing an engine core crash. ([#1839](https://github.com/vllm-project/vllm-gaudi/pull/1839))
- Corrected the `VLLM_GDN_COMPUTE_FP32` documentation: the default is FP32. ([#1859](https://github.com/vllm-project/vllm-gaudi/pull/1859))

---

## Deprecation and Breaking Changes

- The platform now requires Intel® Gaudi® Software v1.24.3 with PyTorch 2.14.1.
- The HPU Seed-OSS model override was removed after upstream vLLM removed the module it extended. ([#1880](https://github.com/vllm-project/vllm-gaudi/pull/1880))
- NIXL-based KV transfer requires NIXL 1.5.0 or later. The libfabric (OFI) prefill/decode disaggregation path is not validated in this release until an OFI-enabled NIXL 1.5 build is available. ([#1900](https://github.com/vllm-project/vllm-gaudi/pull/1900))

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#2000](https://github.com/vllm-project/vllm-gaudi/pull/2000) | Restore FP32 matmul Gemma4 vision RoPE on HPU (fix transformers 5.19 OmniDocBench regression) | @slokesha |
| [#1994](https://github.com/vllm-project/vllm-gaudi/pull/1994) | Fix BF16 rounding in HPU device image normalization | @shepark |
| [#1986](https://github.com/vllm-project/vllm-gaudi/pull/1986) | Keep the prompt context when the attention kernel block is larger than the bucketing unit | @rsmyrek |
| [#1987](https://github.com/vllm-project/vllm-gaudi/pull/1987) | Keep 128-token attention kernel blocks for Nemotron-H when they fit | @rsmyrek |
| [#1983](https://github.com/vllm-project/vllm-gaudi/pull/1983) | Port of: Update numba requirement from >=0.68.0 to >=0.68.1-#1951 | @iboiko-habana |
| [#1980](https://github.com/vllm-project/vllm-gaudi/pull/1980) | Port of: Update kaldi-native-fbank requirement to >=1.23.0-#1978 | @iboiko-habana |
| [#1979](https://github.com/vllm-project/vllm-gaudi/pull/1979) | Port of: dockerfiles: add support for Ubuntu 22.04 with Python 3.11-#1956 | @iboiko-habana |
| [#1963](https://github.com/vllm-project/vllm-gaudi/pull/1963) | Rebind the drafter's WNA16 MoE expert views after moving it to HPU | @rsmyrek |
| [#1947](https://github.com/vllm-project/vllm-gaudi/pull/1947) | Restore the split mamba prefill graph for Granite 4.0-H | @rsmyrek |
| [#1962](https://github.com/vllm-project/vllm-gaudi/pull/1962) | Fix pad block and slot ids pointing into a real KV block | @rsmyrek |
| [#1954](https://github.com/vllm-project/vllm-gaudi/pull/1954) | Fix non-reproducible RoPE table built on CPU (port of #1948) | @pawel-olejniczak |
| [#1949](https://github.com/vllm-project/vllm-gaudi/pull/1949) | [Bugfix] Match upstream GemmaRMSNorm precision on HPU (port of #1930) | @shepark |
| [#1874](https://github.com/vllm-project/vllm-gaudi/pull/1874) | Enable Nemotron-H WNA16 on HPU: Non-gated MoE on WNA16 and use the native int4 kernel - is_gated solution | @rsmyrek |
| [#1859](https://github.com/vllm-project/vllm-gaudi/pull/1859) | Fix VLLM_GDN_COMPUTE_FP32 documentation: the code default is fp32, and the documented bf16 default silently corrupts output | @biondogs |
| [#1916](https://github.com/vllm-project/vllm-gaudi/pull/1916) | Sample structured output rows with their own request params | @thediymaker |
| [#1917](https://github.com/vllm-project/vllm-gaudi/pull/1917) | Re-key bad_words to the rows of selective sampling metadata | @thediymaker |
| [#1926](https://github.com/vllm-project/vllm-gaudi/pull/1926) | Fix INC FP8 skipping GateLinear MoE routers | Dariusz Kreft |
| [#1935](https://github.com/vllm-project/vllm-gaudi/pull/1935) | Update transformers requirement from <5.20 to <5.21 | @dependabot[bot] |
| [#1934](https://github.com/vllm-project/vllm-gaudi/pull/1934) | Update kaldi-native-fbank requirement from >=1.22.3 to >=1.22.4 | @dependabot[bot] |
| [#1933](https://github.com/vllm-project/vllm-gaudi/pull/1933) | Update fastapi requirement from <0.147 to <0.163 | @dependabot[bot] |
| [#1931](https://github.com/vllm-project/vllm-gaudi/pull/1931) | Fix FP8 KV decode softmax sums on Gaudi2 | @slokesha |
| [#1911](https://github.com/vllm-project/vllm-gaudi/pull/1911) | Build the selective sampling metadata without synchronous copies on HPU | @rsmyrek |
| [#1927](https://github.com/vllm-project/vllm-gaudi/pull/1927) | [FIX_FOR_VLLM_CUSTOM=1fd6337e22243f40bf4b1cd307c68faf150aaf7d] Fix KV cache tests for upstream global cache groups API | @iboiko-habana |
| [#1915](https://github.com/vllm-project/vllm-gaudi/pull/1915) | Fix zeroed conv state in HPU graph for Granite 4.0-H | @rsmyrek |
| [#1857](https://github.com/vllm-project/vllm-gaudi/pull/1857) | Update numba requirement from >=0.67.0 to >=0.68.0 | @dependabot[bot] |
| [#1904](https://github.com/vllm-project/vllm-gaudi/pull/1904) | Update fastapi requirement from <0.143 to <0.147 | @dependabot[bot] |
| [#1914](https://github.com/vllm-project/vllm-gaudi/pull/1914) | [FIX_FOR_VLLM_CUSTOM=2de0531d81525b3fd652ca2f89810b2ff038395e] Implement get_kv_connector_output in AsyncHPUModelRunnerOutput | @pawel-olejniczak |
| [#1906](https://github.com/vllm-project/vllm-gaudi/pull/1906) | Fix NaN in sliced FusedSDPA for fully masked chunks | @slokesha |
| [#1887](https://github.com/vllm-project/vllm-gaudi/pull/1887) | Update transformers requirement from <5.19 to <5.20 | @dependabot[bot] |
| [#1895](https://github.com/vllm-project/vllm-gaudi/pull/1895) | Fix non-finite initial-state masking in GDN prefill | @shepark |
| [#1901](https://github.com/vllm-project/vllm-gaudi/pull/1901) | Fix text serving benchmarks after sonnet dataset removal | Dariusz Kreft |
| [#1910](https://github.com/vllm-project/vllm-gaudi/pull/1910) | [FIX_FOR_VLLM_CUSTOM=876d5da8316a81180b8c7f10dea5ffab5b5a849f] Import cli_args and cli_env_setup from vllm.entrypoints.common | @pawel-olejniczak |
| [#1913](https://github.com/vllm-project/vllm-gaudi/pull/1913) | Set Phi-4-mini 16k GSM8K card to its typical CI score | @pawel-olejniczak |
| [#1870](https://github.com/vllm-project/vllm-gaudi/pull/1870) | [DSA] Per-request top-k selection in the decode indexer | @jkaniecki |
| [#1909](https://github.com/vllm-project/vllm-gaudi/pull/1909) | Restore tcmalloc preload in RHEL UBI Dockerfiles | @iboiko-habana |
| [#1872](https://github.com/vllm-project/vllm-gaudi/pull/1872) | [Nemotron-H] Remove two torch.compile specializations in the mamba path | @rsmyrek |
| [#1890](https://github.com/vllm-project/vllm-gaudi/pull/1890) | [Qwen3.5/GDN] Fix engine crash when a prefill resumes off the mamba chunk grid | @jiminha |
| [#1900](https://github.com/vllm-project/vllm-gaudi/pull/1900) | [FIX_FOR_VLLM_CUSTOM=852989f580820849113d67d13fd8053e279d0444] Import KV cache, Accept FLy kwargs, Build NIXL 1.5.0 (+1 more) | @pawel-olejniczak |
| [#1873](https://github.com/vllm-project/vllm-gaudi/pull/1873) | [Nemotron-H] Avoid an eager fallback in the prefix-caching conv1d gather | @rsmyrek |
| [#1883](https://github.com/vllm-project/vllm-gaudi/pull/1883) | Fix pooling model warmup so embedding buckets compile at startup | Amir Khalaila |
| [#1899](https://github.com/vllm-project/vllm-gaudi/pull/1899) | Fix non-gated WNA16 MoE unit test for vLLM's up_proj check | @pawel-olejniczak |
| [#1808](https://github.com/vllm-project/vllm-gaudi/pull/1808) | Enable Nemotron-H WNA16 on HPU: Non-gated MoE on WNA16 and use the native int4 kernel - mirror solution | @rsmyrek |
| [#1880](https://github.com/vllm-project/vllm-gaudi/pull/1880) | [FIX_FOR_VLLM_CUSTOM=8d7b93fec2eb72ce01f2217472ac1f51b9bce600] Import chat_utils helpers, Drop HPU SeedOss (+5 more) | @pawel-olejniczak |
| [#1850](https://github.com/vllm-project/vllm-gaudi/pull/1850) | Fix sliding window prefill attention with cached context | @pawel-olejniczak |
| [#1839](https://github.com/vllm-project/vllm-gaudi/pull/1839) | fix: restore batch dim of prefill hidden states for prompt logprobs | @pawel-olejniczak |
| [#1852](https://github.com/vllm-project/vllm-gaudi/pull/1852) | Fix Qwen2.5-VL ignoring images and mmmu_val crash on HPU | @pawel-olejniczak |
| [#1871](https://github.com/vllm-project/vllm-gaudi/pull/1871) | Restore HPU block size 128 after multi-backend default selection | Dariusz Kreft |
| [#1878](https://github.com/vllm-project/vllm-gaudi/pull/1878) | [warmup][multimodal] Fix get_dummy_mm_inputs call after vllm#57674 | @pawel-olejniczak |
| [#1821](https://github.com/vllm-project/vllm-gaudi/pull/1821) | Add phi-4 GSM8K model card and validated models entry | @pawel-olejniczak |
| [#1820](https://github.com/vllm-project/vllm-gaudi/pull/1820) | Fix HPU LongRoPE rotary embedding for Phi-3 and Phi-4-mini | @pawel-olejniczak |
| [#1858](https://github.com/vllm-project/vllm-gaudi/pull/1858) | Update transformers requirement from <5.18 to <5.19 | @dependabot[bot] |
| [#1844](https://github.com/vllm-project/vllm-gaudi/pull/1844) | Release host memory on worker shutdown and guard sleep against host OOM | Dariusz Kreft |
| [#1865](https://github.com/vllm-project/vllm-gaudi/pull/1865) | Fix InputBatch unit test after #1848 | @iboiko-habana |
| [#1856](https://github.com/vllm-project/vllm-gaudi/pull/1856) | Update fastapi requirement from <0.137 to <0.143 | @dependabot[bot] |
| [#1799](https://github.com/vllm-project/vllm-gaudi/pull/1799) | Update transformers requirement from <5.17 to <5.18 | @dependabot[bot] |
| [#1810](https://github.com/vllm-project/vllm-gaudi/pull/1810) | [FIX_FOR_VLLM_CUSTOM=88aa0d287dd3abac89741bbea349350a4d49194e] read supports_multimodal_inputs from ModelConfig | @pawel-olejniczak |
| [#1806](https://github.com/vllm-project/vllm-gaudi/pull/1806) | [FIX_FOR_VLLM_CUSTOM=9a70c233cd5d3ef3b14d79c582a6f9ddfa03cf80] forward skip_padding kwarg, drop stale manager.touch() (+3 more) | @pawel-olejniczak |

## New Contributors

Welcome to the following first-time contributors to vLLM Gaudi Plugin!

- **@biondogs** — Fix `VLLM_GDN_COMPUTE_FP32` documentation ([#1859](https://github.com/vllm-project/vllm-gaudi/pull/1859))
- **@thediymaker** — Sample structured output rows with their own request params ([#1916](https://github.com/vllm-project/vllm-gaudi/pull/1916))
- **Dariusz Kreft** — Release host memory on worker shutdown and guard sleep against host OOM ([#1844](https://github.com/vllm-project/vllm-gaudi/pull/1844))
- **Amir Khalaila** — Fix pooling model warmup so embedding buckets compile at startup ([#1883](https://github.com/vllm-project/vllm-gaudi/pull/1883))
