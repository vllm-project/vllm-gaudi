# vLLM Gaudi Plugin v0.15.1 Release Notes

## Overview

This release is based on [vLLM v0.15.1](https://github.com/vllm-project/vllm/releases/tag/v0.15.1) and supports [Intel® Gaudi® Software v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html).

---

## Highlights

- Added validated support for **Granite 4.0-h** and **Qwen3-VL** (dense and MoE variants) on Intel Gaudi 3. Additionally, added significant **Llama 4** stability fixes.
- Introduced full chunked prefill attention support for HPU, enabling better memory utilization on long sequences (#821).
- Integrated FlashAttention online merge in Unified Attention for improved prefill performance (#785).
- Added KV cache sharing support for HPU, enabling more efficient multi-query scenarios (#834).
- Introduced support for NVIDIA ModelOpt FP8 quantization format for dense models (#890).
- Added HPU ops for Mamba mixer2, causal conv1d, and SSD combined kernels enabling hybrid SSM-Transformer models, such as Granite 4.0-h (#886, #897).
- Added back-to-back matmul operation for improved Multi-Latent Attention (MLA) performance (#770).
- Introduced prefill-side KV layout and block size support for heterogeneous (disaggregated) inference via NIXL (#867).

---

## New Model Support

- Add validated support for **Qwen3-VL-32B-Instruct**, **Qwen3-VL-32B-Thinking**, and **Qwen3-VL-235B-A22B** variants (Instruct, Thinking, FP8) on Gaudi 3 (#958)
- Register the `Qwen3VLMoeForConditionalGeneration` model for Qwen3-VL MoE variants (#958)
- Add **IBM Granite 4.0-h small** (hybrid SSM-Transformer) implementation for HPU (#897)

---

## Performance

- Add FlashAttention online merge in Unified Attention for faster prefill (#785)
- Add back-to-back (b2b) matmul for improved MLA attention performance (#770)
- Support loading `q_scale` and using `fp8_fused_sdpa` for MLA prefill (#909)
- Remove bucket densification for long context; apply edge buckets only for long context scenarios (#980)
- Implement bucket corrector for Mamba chunk size (#886)
- Revert "skip HPU graphs for long prefills" to restore graph capture on long sequences (#850)
- Port initialization profiling noop to reduce startup overhead (#979)

---

## Attention & KV Cache

- Add support for chunked attention on HPU (#821)
- Add KV cache sharing for HPU (#834)
- Enable support for prefill-side `kv_layout` and `block_size` update for heterogeneous runs (#867)
- Add new `VLLM_HPU_HETERO_KV_LAYOUT` environment variable to control heterogeneous KV layout (#867)
- Add heterogeneous HPU NIXL connector for disaggregated prefill/decode (#867)
- Add `hpu_attention` ops module with attention operation implementations (#785)
- Monkey-patch `Attention.forward` for HPU-specific behavior (#973)
- Platform: declare `support_hybrid_kv_cache` capability (#834)

---

## Quantization

- Add support for **ModelOpt FP8** quantization format for dense models (#890)
- Add `modelopt` to platform supported quantization list (#890)
- Add dynamic quantization configuration file example (#838)

---

## Plugin Core

- Register new ops: `hpu_attention`, `hpu_grouped_topk_router`, `hpu_mamba_mixer2`, and `hpu_modelopt` (#785, #897, #890)
- Add `ops_selector` module for HPU operation routing (#897)
- Add `pytorch_implementation` module with pure-PyTorch fallback ops (#897)
- Add `causal_conv1d_pytorch` and `ssd_combined` ops for SSM/Mamba support (#897)
- Add `hpu_grouped_topk_router` for MoE grouped top-k routing (#897)
- Source `use_qk_norm` parameter directly from config (#1035)

---

## Serving & Infrastructure

- Add GitHub Actions `action.yaml` for PR detail workflows (#1030)
- Add CI calibration smoke tests script (#853)
- Rename and consolidate CI e2e discoverable tests (#840)
- Fix Jenkins CI for Mistral model tests (#840)
- Restore `temperature=0` as server default after vLLM #32723 (#1038)
- Backport RHEL/UBI Dockerfile improvements (#1049)

---

## Fixes

- Fix **Llama 4** apply-patches flow, QK flatten positional encoding, and address performance drop (#942)
- Fix Llama 4 shape mismatch for 32k+ context window (#842, #855)
- Fix **Qwen2.5-VL** accuracy regression (#831)
- Fix **Qwen3-VL** multimodal model embedding issues (#958)
- Fix **DeepSeek** tensor device mismatch (#1029)
- Force CPU loading for INC quantization to prevent OOM during weight loading (#1005)
- Fix INC patching `_gate` twice (#955, #1020)
- Fix HPU model runner `profile_run` to work with dynamic kv-cache scales (#852)
- Fix measurement config file generation in `calibrate_model.sh` scripts (#853)
- Revert padding value change for `block_list` and slot list (#1007)
- Fix multimodal budget divergence from upstream vLLM (#837)
- Fix hourly `KeyError: <PlatformEnum.OOT: 6>` error (#968)
- Fix `torch.compile` in data-parallel mode (#722)
- Correct sliding window enabling logic (#805)
- Interleaved sliding window fix (#805)
- Fix Mamba cumsum padded calculations (#1022)
- Fix redundant transpose in HPUMambaMixer2 (#999, #1014)
- Fix Qwen3-VL MoE execution failure (#992)
- Fix `last_chunk_indices` calculations (#1024)

---

## Security

**CVE-2025-69872 (diskcache 5.6.3)**: vLLM currently depends on `diskcache` version 5.6.3, which has been reported as affected by CVE-2025-69872. The vulnerability remains unresolved upstream as of the day of this release. According to initial analysis, the vLLM architecture does not expose the vulnerable code path, meaning vLLM is **not impacted in practice**, despite the dependency being formally flagged.

---

## Deprecation & Breaking Changes

- Remove `tests/models/utils.py` to clean up unused test utilities (#864)
- `VLLM_HPU_HETERO_KV_LAYOUT` environment variable is now required for heterogeneous (disaggregated) prefill/decode with NIXL (#867)
- Remove bucket densification for long context workloads; only edge buckets are applied (#980)

---

## Full Changelog

| PR | Title | Author |
|----|-------|--------|
| #805 | Interleaved sliding window fix | @rsmyrek |
| #722 | DP: Fix for torch.compile | @xuechendi |
| #770 | Add b2b matmul | @linoybu |
| #785 | Add FlashAttention online merge in Unified Attention | @kzawora-intel |
| #805 | Correct sliding window enabling | @jbyczkow |
| #821 | Add support for chunked attention | @kfojcik-intel |
| #831 | Resolve qwen25 vl accuracy regression | @tvoas |
| #834 | KV cache sharing for HPU | @jakub-sochacki |
| #837 | Fix diverge from vllm in multiModalBudget | @linoybu |
| #838 | Add dynamic quantization configuration file example | @dudilester |
| #840 | Jenkins CI fix for Mistral | @iboiko-habana |
| #850 | Revert "skip HPU graphs for long prefills" | @adobrzyn |
| #851 | Fix for vLLM #32077 | @iboiko-habana |
| #852 | Fix HPU model runner profile_run to work with dynamic kv-cache scales | @dudilester |
| #853 | Fix measurement config file generation in calibrate_model.sh | @nirda7 |
| #864 | Remove unused test utils | @microslaw |
| #867 | Enable support for prefill side kv_layout and block_size update | @yeonsily |
| #876 | Refactor for vLLM #30623 and small fix for #32238 | @iboiko-habana |
| #886 | Implement bucket corrector for Mamba chunk size | @jbyczkow |
| #890 | Support for modelopt FP8 quantization format for dense models | @skavulya |
| #897 | HPU Granite 4.0-h small implementation | @jbyczkow |
| #905 | CODEOWNERS update | @kzawora-intel |
| #909 | Support loading q_scale and using fp8_fused_sdpa for MLA prefill | @lkk12014402 |
| #917 | Fix for hourly KeyError: PlatformEnum.OOT | @tzielinski-habana |
| #920 | Update compatibility matrix and refine installation instructions | @PatrykWo |
| #942 | Llama4 apply patches + QK flatten pos + perf drop fix | @Luca-Calabria |
| #943 | Update Dockerfiles and documentation for v0.15.1 release | @PatrykWo |
| #958 | Qwen3_VL - multimodal model embedding fixes | @slokesha |
| #968 | Fix for hourly KeyError: PlatformEnum.OOT: 6 | @tzielinski-habana |
| #973 | Monkey-patch Attention.forward | @tzielinski-habana |
| #979 | Port: Initialization profiling noop | @adobrzyn |
| #980 | Remove bucket densification for long ctx; Edge buckets only | @kfojcik-intel |
| #1003 | Remove duplicate path | @adobrzyn |
| #1005 | Force CPU loading for INC quantization to prevent OOM | @kamil-kaczor |
| #1007 | Revert padding value change for block_list and slot list | @kamil-kaczor |
| #1020 | Fix INC patching _gate twice | @kamil-kaczor |
| #1029 | Fix tensor device mismatch in deepseek | @kamil-kaczor |
| #1030 | Adding action.yaml | @iboiko-habana |
| #992 | Fix qwen3 vl moe execution failure | @shepark |
| #1014 | Fixing redundant transpose in HPUMambaMixer2 | @ksmusz |
| #1022 | Fix mamba cumsum padded calculations | @jkaniecki |
| #1024 | last_chunk_indices calculations fix | @jbyczkow |
| #1035 | use_qk_norm parameter sourced directly from config | @rsmyrek |
| #1038 | Back temperature=0 for server as default | @iboiko-habana |
| #1049 | Backport RHEL/UBI Dockerfile improvements | @PatrykWo |

---

## New Contributors

Welcome to the following first-time contributors to vLLM Gaudi Plugin! 🎉

- **@linoybu** — b2b matmul and multimodal budget fix (#770, #837)
- **@microslaw** — Test utilities cleanup (#864)
- **@nirda7** — Calibration script fixes (#853)
- **@tzielinski-habana** — Platform stability fixes and Attention.forward monkey-patch (#917, #968, #973)
- **@yeonsily** — Heterogeneous KV layout support (#867)
- **@jkaniecki** — Mamba cumsum padded calculations fix (#1022)
- **@shepark** — Qwen3-VL MoE execution fix (#992)
