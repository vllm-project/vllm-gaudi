# Commit Difference: `main` vs `releases/v0.17.1`

**Generated**: 2026-03-25

## Summary

| Metric | Count |
|--------|-------|
| Merge base | `8b9cfff` — [FIX_FOR_VLLM_CUSTOM=bd2659a5660a7c5ccfeb1f1579e4000ed6536250] Fix for #35503 (#1116) |
| Total commits only in `main` | 30 |
| Total commits only in `releases/v0.17.1` | 17 |
| Cherry-pick equivalents (shared by patch-id) | 8 pairs |
| **Truly unique to `main`** | **22** |
| **Truly unique to `releases/v0.17.1`** | **9** |

---

## Cherry-Picked / Equivalent Commits (8 pairs)

These commits exist in both branches with equivalent patches but different commit hashes:

| `main` commit | `releases/v0.17.1` commit | Description |
|---|---|---|
| `ba2d7dd` (#1212) | `b8c80d2` (#1219) | fix: include all sub-packages in setuptools package discovery |
| `3e181e7` (#1214) | `703b047` (#1224) | [Cherry-pick] Fix param mismatch for compute_nixl_compatibility_hash() |
| `2c970ee` (#1194) | `72714ae` (#1172) | Fix SharedFusedMoE attribute error for Llama4 MoE layers |
| `7b4e171` (#1173) | `1c88f9b` (#1178) | [docs] Update quickstart guide and supported model list |
| `1ddc974` (#1164) | `966ad36` (#1163) | Coverity fix including security, null-like values, duplicates and types |
| `33601b1` (#1093) | `02d6401` (#1170) | Set reserved mem for Torch compile |
| `5354831` (#1141) | `da59549` (#1142) | Add num_spec field to MambaMixer2 for upstream compatibility |
| `9e77a15` (#1121) | `1697491` (#1167) | Fix -u flag requiring argument in calibrate_model.sh |

---

## Commits Only in `main` (22 unique)

These commits are on `main` and have **not** been cherry-picked to `releases/v0.17.1`:

| Commit | Date | Author | Description |
|--------|------|--------|-------------|
| `fbb13ab` | 2026-03-24 | Aung San aka Mike | .cd/Dockerfile.rhel.ubi.vllm; add arg VLLM_REPO and VLLM_GAUDI_REPO (#1225) |
| `25a3187` | 2026-03-23 | Libin Tang | qwen35 initial enablement (#1153) |
| `614cd0a` | 2026-03-23 | Jan Wieczorek | Adapt Online defragmenter for torch compile (#986) |
| `b098a69` | 2026-03-23 | Tony Lin | Monkey patch for LMCache (#1176) |
| `a921a3a` | 2026-03-23 | Youlei Yang | Add real context length to the high-level profile (#1169) |
| `c2d7934` | 2026-03-21 | Iryna Boiko | [FIX_FOR_VLLM_LATEST] Remove deprecated virtual_engine from ForwardContext (#1187) |
| `0f23319` | 2026-03-20 | Iryna Boiko | Port of "[HPU][Nixl] Fix false-positive cross-layer block detection for MLA" (#1205) |
| `18d3313` | 2026-03-20 | Jakub Byczkowski | Fix grammar bitmask corruption in mixed structured-output batches (#1200) |
| `26c56f5` | 2026-03-18 | Tomasz Zielinski | [FIX_FOR_VLLM_CUSTOM=a116f969301acfdb6ea] Hourly fix (#1174) |
| `4ee8db0` | 2026-03-18 | Artur Fierka | Fix OOM crashes during high-concurrency inference (GAUDISW-246982) (#1124) |
| `20258d4` | 2026-03-18 | Iryna Boiko | Disable nixl CI tests (#1181) |
| `970a534` | 2026-03-18 | Artur Fierka | Fix multimodal prefill batching for 2D padded inputs (#1126) |
| `4855850` | 2026-03-16 | Kamil Kaczor | Fix KV cache memory regression from unconditional RowParallelLinear OOT registration (#1146) |
| `b83b76d` | 2026-03-16 | Youlei Yang | fix preempted prompts and prefill/decoding splitting (#830) |
| `9bd2ada` | 2026-03-12 | Jacek Czaja | PR-1054 revert (#1136) |
| `715451c` | 2026-03-12 | Yeonsil Yoon | fix: [vllm-hourly] exclude dummy block from NIXL KV cache registration (#1140) |
| `4f284ff` | 2026-03-12 | Jan Wieczorek | [GAUDISW-246895] Remove aggregate module HpuDeepseekOCRVisual (#1102) |
| `2749367` | 2026-03-11 | Iryna Boiko | Temporary nixl test cases disablement (#1139) |
| `070dc42` | 2026-03-11 | Kamil Kaczor | Add AI agents config files (#1123) |
| `4387b08` | 2026-03-11 | Iryna Boiko | Temporary nixl test cases disablement (#1135) |
| `c3f012e` | 2026-03-11 | Iryna Boiko | [FIX_FOR_VLLM_LATEST] Fixes for #35122 and #35953 (#1129) |
| `f4bbeaf` | 2026-03-10 | Patryk Wolsza | Parameterize EXTRA_INDEX_URL (#1131) |

---

## Commits Only in `releases/v0.17.1` (9 unique)

These commits are on `releases/v0.17.1` and have **not** been merged to `main`:

| Commit | Date | Author | Description |
|--------|------|--------|-------------|
| `025e052` | 2026-03-24 | Jakub Byczkowski | Prefix caching support for HPUMambaMixer2 (#1198) |
| `cba194a` | 2026-03-23 | Krzysztof Smusz | Improving precision of _depthwise_conv1d_tpc for bf16 (#1203) |
| `cc72769` | 2026-03-20 | Jakub Byczkowski | Fix grammar bitmask corruption in mixed structured-output batches (#1199) |
| `367a037` | 2026-03-19 | Krzysztof Smusz | Creating custom depthwise conv1d kernel for MambaMixer2 (#1175) |
| `4072009` | 2026-03-11 | github-actions[bot] | update CODEOWNERS for v0.17.1 |
| `62769b6` | 2026-03-11 | github-actions[bot] | Set vLLM stable commit for v0.17.1 |
| `b0d926b` | 2026-03-11 | Agata Dobrzyniewicz | Fix codeowners for BO |
| `ed20e7f` | 2026-03-10 | github-actions[bot] | update CODEOWNERS for v0.17.0 |
| `11fa49b` | 2026-03-10 | github-actions[bot] | Set vLLM stable commit for v0.17.0 |

---

## Categorization of Unique `main` Commits (candidates for cherry-pick to `releases/v0.17.1`)

### Bug Fixes (high priority for backport)
| Commit | Description |
|--------|-------------|
| `18d3313` | Fix grammar bitmask corruption in mixed structured-output batches (#1200) |
| `4ee8db0` | Fix OOM crashes during high-concurrency inference (#1124) |
| `970a534` | Fix multimodal prefill batching for 2D padded inputs (#1126) |
| `4855850` | Fix KV cache memory regression from unconditional RowParallelLinear OOT registration (#1146) |
| `b83b76d` | fix preempted prompts and prefill/decoding splitting (#830) |
| `715451c` | fix: [vllm-hourly] exclude dummy block from NIXL KV cache registration (#1140) |
| `0f23319` | Port of "[HPU][Nixl] Fix false-positive cross-layer block detection for MLA" (#1205) |

### Features / Enhancements
| Commit | Description |
|--------|-------------|
| `25a3187` | qwen35 initial enablement (#1153) |
| `614cd0a` | Adapt Online defragmenter for torch compile (#986) |
| `b098a69` | Monkey patch for LMCache (#1176) |
| `a921a3a` | Add real context length to the high-level profile (#1169) |

### Upstream Compatibility / Hourly Fixes
| Commit | Description |
|--------|-------------|
| `c2d7934` | [FIX_FOR_VLLM_LATEST] Remove deprecated virtual_engine from ForwardContext (#1187) |
| `26c56f5` | [FIX_FOR_VLLM_CUSTOM=a116f969301acfdb6ea] Hourly fix (#1174) |
| `c3f012e` | [FIX_FOR_VLLM_LATEST] Fixes for #35122 and #35953 (#1129) |

### CI / Infra / Config
| Commit | Description |
|--------|-------------|
| `fbb13ab` | .cd/Dockerfile.rhel.ubi.vllm; add arg VLLM_REPO and VLLM_GAUDI_REPO (#1225) |
| `20258d4` | Disable nixl CI tests (#1181) |
| `2749367` | Temporary nixl test cases disablement (#1139) |
| `4387b08` | Temporary nixl test cases disablement (#1135) |
| `070dc42` | Add AI agents config files (#1123) |
| `f4bbeaf` | Parameterize EXTRA_INDEX_URL (#1131) |

### Reverts / Cleanups
| Commit | Description |
|--------|-------------|
| `9bd2ada` | PR-1054 revert (#1136) |
| `4f284ff` | [GAUDISW-246895] Remove aggregate module HpuDeepseekOCRVisual (#1102) |

---

## Categorization of Unique `releases/v0.17.1` Commits (candidates for merge to `main`)

### Features / Enhancements (consider merging to main)
| Commit | Description |
|--------|-------------|
| `025e052` | Prefix caching support for HPUMambaMixer2 (#1198) |
| `cba194a` | Improving precision of _depthwise_conv1d_tpc for bf16 (#1203) |
| `367a037` | Creating custom depthwise conv1d kernel for MambaMixer2 (#1175) |
| `cc72769` | Fix grammar bitmask corruption in mixed structured-output batches (#1199) |

### Release-Specific (no action needed)
| Commit | Description |
|--------|-------------|
| `4072009` | update CODEOWNERS for v0.17.1 |
| `62769b6` | Set vLLM stable commit for v0.17.1 |
| `b0d926b` | Fix codeowners for BO |
| `ed20e7f` | update CODEOWNERS for v0.17.0 |
| `11fa49b` | Set vLLM stable commit for v0.17.0 |
