# vLLM Gaudi Plugin v0.30.0 Release Notes

## Overview

This release is based on [vLLM v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) and supports [Intel® Gaudi® Software v1.24.2](https://docs.habana.ai/en/v1.24.2/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.13.

## Highlights

- Enabled the plugin on upstream [vLLM v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0), incorporating upstream API changes across MoE, MLA, KV cache, quantization, and serving.
- Added support for GLM-5.2-FP8 with DeepSeek Sparse Attention (DSA) and Nemotron-H FP8 with quantization-aware Mamba and non-gated FP8 MoE.
- Improved Qwen3.5/3.6 MoE token generation at low batch sizes with an opt-in gathered-expert MoE combine for FP8.
- Improved Qwen3.5 GDN correctness with prefix caching and compact GDN, and fixed RoPE accuracy for very long contexts.
- Reduced decode warmup memory by clamping decode block buckets to the physical KV cache size.
- Added RHEL 10.2 Dockerfiles and strengthened supply-chain security with Cosign-signed release tarballs, OpenSSF Scorecard, Dependabot, and SHA-pinned GitHub Actions.

---

## New Model Support

- Enabled GLM-5.2-FP8 with DeepSeek Sparse Attention (DSA) on HPU. ([#1760](https://github.com/vllm-project/vllm-gaudi/pull/1760))
- Enabled Nemotron-H FP8 on HPU with a quantization-aware Mamba `in_proj` and non-gated FP8 MoE. ([#1666](https://github.com/vllm-project/vllm-gaudi/pull/1666))
- Added the bidirectional (prefix-LM) image attention mask for Gemma-4 prefill on HPU. ([#1797](https://github.com/vllm-project/vllm-gaudi/pull/1797))

---

## Performance

- Added an opt-in custom gathered-expert MoE combine for SiLU/FP8, targeting the Qwen3.5 family (`VLLM_HPU_MOE_GATHER`). ([#1731](https://github.com/vllm-project/vllm-gaudi/pull/1731))
- Clamped decode block buckets to the physical KV cache size for the exponential bucketing strategy, preventing decode warmup OOM. ([#1700](https://github.com/vllm-project/vllm-gaudi/pull/1700))
- Fixed Gemma-4 decode performance with `VLLM_CONTIGUOUS_PA=true`. ([#1715](https://github.com/vllm-project/vllm-gaudi/pull/1715))
- Enabled Gemma-4 vision-tower warmup resolutions on HPU. ([#1637](https://github.com/vllm-project/vllm-gaudi/pull/1637))
- Added the opt-in `VLLM_MM_WARMUP_OUTSIDE_COMPILE_ONLY` environment variable to run multimodal warmup outside `PT_COMPILE_ONLY_MODE`. ([#1744](https://github.com/vllm-project/vllm-gaudi/pull/1744))
- Enabled `flatten_input` for the `qwen3_5` and `qwen3_5_text` model types in language-model-only mode, replacing the previous unconditional Qwen3.5 setting. ([#1713](https://github.com/vllm-project/vllm-gaudi/pull/1713), [#1738](https://github.com/vllm-project/vllm-gaudi/pull/1738))
- Bypassed the functional-collectives remap for compiled HPU `all_reduce`. ([#1819](https://github.com/vllm-project/vllm-gaudi/pull/1819))

---

## Attention and KV Cache

- Fixed Qwen3.5 GDN recurrent-state load/store slots with prefix caching when compact GDN is disabled. ([#1827](https://github.com/vllm-project/vllm-gaudi/pull/1827))
- Fixed the Qwen3.5 GDN `mamba_chunk_size` alignment bypass with prefix caching. ([#1815](https://github.com/vllm-project/vllm-gaudi/pull/1815))
- Disabled Synapse persistent-input reuse for compact GDN to prevent recurrent-state corruption. ([#1753](https://github.com/vllm-project/vllm-gaudi/pull/1753))
- Fixed RoPE numerical issues for very long contexts. ([#1817](https://github.com/vllm-project/vllm-gaudi/pull/1817))
- Tiled the FusedSDPA query dimension to avoid the 2^31-byte prompt-bias overflow and handled an attention bias of `None` in the tiling path. ([#1749](https://github.com/vllm-project/vllm-gaudi/pull/1749), [#1784](https://github.com/vllm-project/vllm-gaudi/pull/1784))
- Restricted Mamba/hybrid prefill batches to a single request. ([#1725](https://github.com/vllm-project/vllm-gaudi/pull/1725))

---

## Quantization

- Restored the MoE gate on the runner after INC conversion. ([#1737](https://github.com/vllm-project/vllm-gaudi/pull/1737))
- Registered `_FakeINCConfig` for INC through the upstream quantization registry. ([#1739](https://github.com/vllm-project/vllm-gaudi/pull/1739))

---

## Plugin Core

- Tracked upstream vLLM MoE changes: the `MoERunner` shared-experts refactor, the shared-experts overlap API, `SharedExperts` stream synchronization in the DP=1 fast path, the `FusedMoE` factory rename and Marlin removal, `select_experts` import relocation, and MoE router factory arguments. ([#1723](https://github.com/vllm-project/vllm-gaudi/pull/1723), [#1773](https://github.com/vllm-project/vllm-gaudi/pull/1773), [#1732](https://github.com/vllm-project/vllm-gaudi/pull/1732), [#1691](https://github.com/vllm-project/vllm-gaudi/pull/1691), [#1802](https://github.com/vllm-project/vllm-gaudi/pull/1802), [#1780](https://github.com/vllm-project/vllm-gaudi/pull/1780))
- Tracked upstream vLLM MLA and attention changes: `fuse_qkv_rmsnorm` on the HPU MLA wrapper, MLA `W_UV`, the removed `calculate_kv_scales` runtime path, and the removed `is_interleaved` helper. ([#1782](https://github.com/vllm-project/vllm-gaudi/pull/1782), [#1746](https://github.com/vllm-project/vllm-gaudi/pull/1746), [#1701](https://github.com/vllm-project/vllm-gaudi/pull/1701), [#1641](https://github.com/vllm-project/vllm-gaudi/pull/1641))
- Tracked upstream vLLM KV cache and quantization changes: `KVCacheTensor.shared_by` migration, `region_num_blocks`, and removed `actorder`/`g_idx`. ([#1751](https://github.com/vllm-project/vllm-gaudi/pull/1751), [#1788](https://github.com/vllm-project/vllm-gaudi/pull/1788))
- Tracked upstream vLLM engine, processor, and serving changes: relocated engine/protocol imports, pre-tokenized `ProcessorInputs` prompts, `default_chat_template_kwargs`, and HPU communicator and shared-memory updates. ([#1776](https://github.com/vllm-project/vllm-gaudi/pull/1776), [#1748](https://github.com/vllm-project/vllm-gaudi/pull/1748), [#1786](https://github.com/vllm-project/vllm-gaudi/pull/1786), [#1660](https://github.com/vllm-project/vllm-gaudi/pull/1660))
- Removed the HPU HunYuan V1 overrides after upstream vLLM deleted the native HunYuan V1 implementation; HunYuan V1 models are now served through the upstream Transformers modeling backend. ([#1759](https://github.com/vllm-project/vllm-gaudi/pull/1759))
- Migrated model weight-loader skip rules following upstream vLLM PR #53106. ([#1763](https://github.com/vllm-project/vllm-gaudi/pull/1763))
- Added an empty `record_logical_topk_ready()` call for upstream compatibility. ([#1824](https://github.com/vllm-project/vllm-gaudi/pull/1824))
- Fixed the MiniMax-M2 fused MoE import after an upstream API change. ([#1720](https://github.com/vllm-project/vllm-gaudi/pull/1720))

---

## Serving and Infrastructure

- Added RHEL 10.2 support to the Dockerfiles. ([#1801](https://github.com/vllm-project/vllm-gaudi/pull/1801))
- Made the MiniMax-M3 tool-call parser tolerate a missing `<` in the invoke tag and listed MiniMax-M3 as a validated model. ([#1724](https://github.com/vllm-project/vllm-gaudi/pull/1724))
- Fixed runtime errors in single-process Qwen model swapping. ([#1604](https://github.com/vllm-project/vllm-gaudi/pull/1604))
- Fixed model banner and runtime logging issues in multi-model runs. ([#1608](https://github.com/vllm-project/vllm-gaudi/pull/1608))
- Kept multimodal processor output on HPU via `torch_shm` in tests. ([#1706](https://github.com/vllm-project/vllm-gaudi/pull/1706))
- Updated dependency requirements: `numba>=0.67.0`, `setuptools>=77.0.3,<85.0.0`, `kaldi-native-fbank>=1.22.3`, `pandas>=2.3.3`, and `tblib` 3.2.2. ([#1729](https://github.com/vllm-project/vllm-gaudi/pull/1729), [#1758](https://github.com/vllm-project/vllm-gaudi/pull/1758), [#1682](https://github.com/vllm-project/vllm-gaudi/pull/1682), [#1730](https://github.com/vllm-project/vllm-gaudi/pull/1730), [#1703](https://github.com/vllm-project/vllm-gaudi/pull/1703), [#1681](https://github.com/vllm-project/vllm-gaudi/pull/1681), [#1702](https://github.com/vllm-project/vllm-gaudi/pull/1702))

---

## Fixes

- Fixed sampling penalties under async scheduling. ([#1761](https://github.com/vllm-project/vllm-gaudi/pull/1761))
- Preserved stale in-flight output after async-scheduling preemption. ([#1794](https://github.com/vllm-project/vllm-gaudi/pull/1794))
- Fixed the decode-block clamp misdetecting hybrid and sliding-window models. ([#1745](https://github.com/vllm-project/vllm-gaudi/pull/1745))
- Fixed a Kimi-K2.5 vision 2D-RoPE warmup crash after upstream fused-kernel inlining. ([#1754](https://github.com/vllm-project/vllm-gaudi/pull/1754))
- Patched `torch.accelerator.empty_host_cache` to fix a teardown segfault. ([#1762](https://github.com/vllm-project/vllm-gaudi/pull/1762))

---

## Security

- Signed release source tarballs with Cosign (keyless). ([#1695](https://github.com/vllm-project/vllm-gaudi/pull/1695))
- Added OpenSSF Scorecard and Dependabot, and pinned GitHub Actions to commit SHAs. ([#1678](https://github.com/vllm-project/vllm-gaudi/pull/1678))
- Added security validation unit tests. ([#1669](https://github.com/vllm-project/vllm-gaudi/pull/1669))

---

## Deprecation and Breaking Changes

- Compact GDN (`VLLM_COMPACT_GDN`) is now automatically disabled when prefix caching is enabled. Set `VLLM_COMPACT_GDN` explicitly to override. ([#1827](https://github.com/vllm-project/vllm-gaudi/pull/1827))
- When compact GDN is active in torch.compile mode, `PT_HPU_ENABLE_SYNAPSE_INPUT_REUSE` now defaults to `0`; a user-provided value is respected. ([#1753](https://github.com/vllm-project/vllm-gaudi/pull/1753))
- HunYuan V1 models (`tencent/Hunyuan-7B-Instruct`, `tencent/Hunyuan-A13B-Instruct`) are removed from the validated models list for this release, as they now run through the upstream Transformers modeling backend and have not yet been re-validated. ([#1759](https://github.com/vllm-project/vllm-gaudi/pull/1759))

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1827](https://github.com/vllm-project/vllm-gaudi/pull/1827) | [Qwen3.5] COMPACT_GDN=0 with prefix cache support | @jiminha |
| [#1824](https://github.com/vllm-project/vllm-gaudi/pull/1824) | Add record_logical_topk_ready() empty call for upstream compatibility | @jkaniecki |
| [#1815](https://github.com/vllm-project/vllm-gaudi/pull/1815) | [Qwen3.5/GDN] Fix mamba_chunk_size alignment bypass with prefix caching | @yeonsily |
| [#1819](https://github.com/vllm-project/vllm-gaudi/pull/1819) | Port of: Bypass the funcol remap for compiled HPU all_reduce- #1811 | @iboiko-habana |
| [#1817](https://github.com/vllm-project/vllm-gaudi/pull/1817) | Fix RoPE numerical issues for very long contexts | @jkaniecki |
| [#1801](https://github.com/vllm-project/vllm-gaudi/pull/1801) | dockerfiles: add support for RHEL 10.2 | @mmuszynskihabana |
| [#1797](https://github.com/vllm-project/vllm-gaudi/pull/1797) | [Gemma4] Add bidirectional (prefix-LM) image mask on HPU prefill | @jiminha |
| [#1802](https://github.com/vllm-project/vllm-gaudi/pull/1802) | [FIX_FOR_VLLM_CUSTOM=4be3dcf0fc7a9086d978eaea3c5a4d78fb98e44a] import select_experts from, kill toy_proxy_server in (+1 more) | @pawel-olejniczak |
| [#1760](https://github.com/vllm-project/vllm-gaudi/pull/1760) | GLM-5.2-FP8 with DSA enablement | @jkaniecki |
| [#1794](https://github.com/vllm-project/vllm-gaudi/pull/1794) | fix: keep stale in-flight output after async-scheduling preemption | @pawel-olejniczak |
| [#1788](https://github.com/vllm-project/vllm-gaudi/pull/1788) | [FIX_FOR_VLLM_CUSTOM=d2906091bfc579cebefe3d8e8fb9077397ce9882] drop removed actorder/g_idx, populate region_num_blocks in (+3 more) | @pawel-olejniczak |
| [#1731](https://github.com/vllm-project/vllm-gaudi/pull/1731) | feat: custom gathered-expert MoE combine for silu/FP8 (specifically Qwen 3.5 family) | @NatTuck |
| [#1786](https://github.com/vllm-project/vllm-gaudi/pull/1786) | [FIX_FOR_VLLM_CUSTOM=cd64c2dea9c72af333de7ec05293d54bbf1bd128] handle key=None on, pass default_chat_template_kwargs into | @pawel-olejniczak |
| [#1784](https://github.com/vllm-project/vllm-gaudi/pull/1784) | Port of: Handle None attn_bias in FusedSDPA tiling path- #1778 | @iboiko-habana |
| [#1782](https://github.com/vllm-project/vllm-gaudi/pull/1782) | [FIX_FOR_VLLM_CUSTOM=8f816a3f665489d7f0d222115d4f72ebab01076b] set fuse_qkv_rmsnorm on HPU MLA wrapper | @pawel-olejniczak |
| [#1753](https://github.com/vllm-project/vllm-gaudi/pull/1753) | Disable Synapse persistent-input reuse for compact-GDN | @asinbarajx |
| [#1780](https://github.com/vllm-project/vllm-gaudi/pull/1780) | [FIX_FOR_VLLM_CUSTOM=a56654d6de060495ff2db3b1d9ff0b187084d1a9] Accept bias_vl and image_sentinel_lo in HPU MoE router factory | @pawel-olejniczak |
| [#1776](https://github.com/vllm-project/vllm-gaudi/pull/1776) | [FIX_FOR_VLLM_CUSTOM=39e276eaeb9daed06a180f6a8d187bbb8790e97b] Retarget engine/protocol imports moved by vllm#54492 | @pawel-olejniczak |
| [#1773](https://github.com/vllm-project/vllm-gaudi/pull/1773) | [FIX_FOR_VLLM_CUSTOM=488b6da105222f4f8130b3aae4a0e67cfc61f522] Feature-detect MoE shared-experts overlap API | @pawel-olejniczak |
| [#1761](https://github.com/vllm-project/vllm-gaudi/pull/1761) | [HPU] Make sampling penalties correct under async scheduling | @jiminha |
| [#1763](https://github.com/vllm-project/vllm-gaudi/pull/1763) | Migrate model weight-loader skip rules, fix for #53106 | @iboiko-habana |
| [#1762](https://github.com/vllm-project/vllm-gaudi/pull/1762) | [HPU] Patch torch.accelerator.empty_host_cache to fix teardown segfault | @jiminha |
| [#1759](https://github.com/vllm-project/vllm-gaudi/pull/1759) | [FIX_FOR_VLLM_CUSTOM=17da48596c98946d3e3e6896e2ebd341e809f3bd] Drop HPU HunYuan, Import cli_args from (+3 more) | @pawel-olejniczak |
| [#1758](https://github.com/vllm-project/vllm-gaudi/pull/1758) | Update numba requirement from >=0.66.0 to >=0.67.0 | @dependabot[bot] |
| [#1751](https://github.com/vllm-project/vllm-gaudi/pull/1751) | [FIX_FOR_VLLM_CUSTOM=0ecc284790e5403f74b899524ef82ecb69f83cb3] Migrate KVCacheTensor.shared_by to (+5 more) | @pawel-olejniczak |
| [#1749](https://github.com/vllm-project/vllm-gaudi/pull/1749) | FusedSDPA: tile the query dim to avoid the 2**31-byte prompt-bias overflow | @jiminha |
| [#1754](https://github.com/vllm-project/vllm-gaudi/pull/1754) | [Bugfix][HPU] Fix Kimi-K2.5 vision 2D-RoPE warmup crash after upstream fused-kernel inlining (#50400) | @shepark |
| [#1666](https://github.com/vllm-project/vllm-gaudi/pull/1666) | Enable Nemotron-H FP8 on HPU: quant-aware Mamba in_proj + non-gated FP8 MoE | @rsmyrek |
| [#1730](https://github.com/vllm-project/vllm-gaudi/pull/1730) | Update setuptools requirement from <84.0.0,>=77.0.3 to >=77.0.3,<85.0.0 | @dependabot[bot] |
| [#1738](https://github.com/vllm-project/vllm-gaudi/pull/1738) | Enable flatten_input for model type "qwen3_5", and "qwen3_5_text" under language-model-only | @shepark |
| [#1748](https://github.com/vllm-project/vllm-gaudi/pull/1748) | [FIX_FOR_VLLM_CUSTOM=0a21947d710f5aedb1865038ebef20e141b29c58] Pre-tokenize ProcessorInputs prompt after upstream processor change | @pawel-olejniczak |
| [#1729](https://github.com/vllm-project/vllm-gaudi/pull/1729) | Update numba requirement from >=0.58.0 to >=0.66.0 | @dependabot[bot] |
| [#1744](https://github.com/vllm-project/vllm-gaudi/pull/1744) | [Bugfix][HPU] Add opt-in env to run MM warmup outside PT_COMPILE_ONLY_MODE | @libinta |
| [#1745](https://github.com/vllm-project/vllm-gaudi/pull/1745) | Fix decode-block clamp misdetecting hybrid/SWA models (regression from #1700) | @adobrzyn |
| [#1746](https://github.com/vllm-project/vllm-gaudi/pull/1746) | [FIX_FOR_VLLM_CUSTOM=14617c2b6c1257ac0d6c7b5e05b195ca30013827] 4 fixes: MLA W_UV, WNA16 MoE backend, preemption KV, launcher import | @pawel-olejniczak |
| [#1608](https://github.com/vllm-project/vllm-gaudi/pull/1608) | Fix model banner and runtime logging issues - multi models runs | @12010486 |
| [#1739](https://github.com/vllm-project/vllm-gaudi/pull/1739) | [FIX_FOR_VLLM_CUSTOM=b05ae5dc008850a620dec6de66635dec2b5913fd] Register _FakeINCConfig for inc via quantization registry | @pawel-olejniczak |
| [#1604](https://github.com/vllm-project/vllm-gaudi/pull/1604) | Fix runtime errors on Qwen single process model swapping | @12010486 |
| [#1724](https://github.com/vllm-project/vllm-gaudi/pull/1724) | MiniMax-M3: tolerate dropped "<" in invoke tag; list validated model | @mkrze |
| [#1737](https://github.com/vllm-project/vllm-gaudi/pull/1737) | Restore MoE gate on the runner after INC conversion | @pawel-olejniczak |
| [#1715](https://github.com/vllm-project/vllm-gaudi/pull/1715) | Fix Gemma4 decode performance with VLLM_CONTIGUOUS_PA=true | @jiminha |
| [#1700](https://github.com/vllm-project/vllm-gaudi/pull/1700) | Clamp decode block buckets to physical KV cache size (exponential) | @adobrzyn |
| [#1702](https://github.com/vllm-project/vllm-gaudi/pull/1702) | Bump tblib from 3.1.0 to 3.2.2 | @dependabot[bot] |
| [#1703](https://github.com/vllm-project/vllm-gaudi/pull/1703) | Update kaldi-native-fbank requirement from >=1.18.7 to >=1.22.3 | @dependabot[bot] |
| [#1725](https://github.com/vllm-project/vllm-gaudi/pull/1725) | fix(hpu): don't merge mamba/hybrid prefills (single-request only) | @pawel-olejniczak |
| [#1732](https://github.com/vllm-project/vllm-gaudi/pull/1732) | [FIX_FOR_VLLM_CUSTOM=1c3633acafd6bbde1f3636cee9799e3ab0879d13] Restore SharedExperts stream-sync in dp1 MoE fast path (+1 more) | @pawel-olejniczak |
| [#1723](https://github.com/vllm-project/vllm-gaudi/pull/1723) | [FIX_FOR_VLLM_CUSTOM=793ca6998adfa5a3ab22d6012dee78044b8ba901] Track MoERunner shared-experts refactor | @pawel-olejniczak |
| [#1720](https://github.com/vllm-project/vllm-gaudi/pull/1720) | Fix MiniMax M2 fused MoE import for vLLM API change | @NatTuck |
| [#1682](https://github.com/vllm-project/vllm-gaudi/pull/1682) | Update setuptools requirement from <80.0.0,>=77.0.3 to >=77.0.3,<84.0.0 | @dependabot[bot] |
| [#1713](https://github.com/vllm-project/vllm-gaudi/pull/1713) | Remove qwen3.5 from flatten_input | @shepark |
| [#1681](https://github.com/vllm-project/vllm-gaudi/pull/1681) | Update pandas requirement from >=2.2.3 to >=2.3.3 | @dependabot[bot] |
| [#1706](https://github.com/vllm-project/vllm-gaudi/pull/1706) | [FIX_FOR_VLLM_CUSTOM=22013f74ffb0ae22deb29233cbe9fcd3efb1b374] fix(test): keep mm processor output on HPU via torch_shm | @pawel-olejniczak |
| [#1669](https://github.com/vllm-project/vllm-gaudi/pull/1669) | test: add security validation unit tests | @adobrzyn |
| [#1701](https://github.com/vllm-project/vllm-gaudi/pull/1701) | [FIX_FOR_VLLM_CUSTOM=397094da1768c7a6f29dfa4f70079d793ac747df] Drop removed calculate_kv_scales runtime path in OOT attention (+1 more) | @pawel-olejniczak |
| [#1695](https://github.com/vllm-project/vllm-gaudi/pull/1695) | [CI] Sign release source tarballs with Cosign (keyless) | @adobrzyn |
| [#1691](https://github.com/vllm-project/vllm-gaudi/pull/1691) | [FIX_FOR_VLLM_CUSTOM=9a4fd57cac19b334b420ae24226c1d2151566799] Fix FusedMoE import after upstream factory rename and Marlin removal | @pawel-olejniczak |
| [#1678](https://github.com/vllm-project/vllm-gaudi/pull/1678) | [CI] Add OpenSSF Scorecard, Dependabot, and SHA-pin GitHub Actions | @adobrzyn |
| [#1660](https://github.com/vllm-project/vllm-gaudi/pull/1660) | [FIX_FOR_VLLM_CUSTOM=ba702e978e3bc6af3a601cee10fefdeb49e7e8b5] HPU upstream-drift fixes: shm guard, communicator kwargs, MLA wrapper, offloading test stubs | @pawel-olejniczak |
| [#1641](https://github.com/vllm-project/vllm-gaudi/pull/1641) | [FIX_FOR_VLLM_CUSTOM=439f336212227833e126526d3c5f3ef3968dfbf5] Restore is_interleaved helper removed upstream | @pawel-olejniczak |
| [#1637](https://github.com/vllm-project/vllm-gaudi/pull/1637) | [warmup][multimodal] Enable Gemma4 vision-tower warmup resolution on HPU | @slokesha |

## New Contributors

Welcome to the following first-time contributors to vLLM Gaudi Plugin!

- **@asinbarajx** — Disable Synapse persistent-input reuse for compact-GDN ([#1753](https://github.com/vllm-project/vllm-gaudi/pull/1753))
- **@NatTuck** — Custom gathered-expert MoE combine for SiLU/FP8 ([#1731](https://github.com/vllm-project/vllm-gaudi/pull/1731))
