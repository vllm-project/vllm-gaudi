# Plan: Migrate Qwen3.5-35B-A3B to vLLM-gaudi v0.17.1

**Date:** 2026-03-27
**Complexity:** HIGH
**Estimated effort:** 2-3 working sessions

---

## Context

The Qwen3.5 branch (`a8/gaudi-qwen35moe-v0.17.1`) has 21 custom commits on top of
commit `7ceb322`, which is 97 upstream commits behind the current `upstream/releases/v0.17.1`
tip (`3297878`). The DeepSeek branch (`a8/deepseek-v32-v0.17.1`) was done correctly --
11 commits cleanly on top of `3297878`.

### Key Facts (from codebase investigation)

**Shared commits (1-4):** Already rebased in DeepSeek branch. The DeepSeek branch
modifies 12 of the same files the Qwen branch touches. For these shared files, the
DeepSeek branch provides a known-good v0.17.1-compatible baseline.

**New files (zero conflict risk):**
- `vllm_gaudi/models/qwen3_next.py` (917 lines)
- `vllm_gaudi/ops/hpu_gdn.py` (885 lines)
- `vllm_gaudi/ops/hpu_gdn_attn.py`
- `vllm_gaudi/ops/gdn_diagnostics.py`
- `vllm_gaudi/ops/causal_conv1d_pytorch.py` (20 lines added)

**High-conflict files:**
- `vllm_gaudi/v1/worker/hpu_model_runner.py` -- Qwen adds +238/-48 (GDN state, Mamba cache),
  upstream adds +1000/-374 (major refactor including Mamba prefix caching, Granite4 state,
  depthwise conv1d TPC, OOM handling). This is the hardest file.

**Medium-conflict files (12 total, all also touched by DeepSeek):**
- `vllm_gaudi/models/__init__.py`, `vllm_gaudi/extension/ops.py`,
  `vllm_gaudi/platform.py`, `vllm_gaudi/attention/backends/hpu_attn.py`,
  `vllm_gaudi/v1/attention/backends/hpu_attn.py`, `vllm_gaudi/extension/features.py`,
  `vllm_gaudi/extension/utils.py`, `vllm_gaudi/extension/bucketing/common.py`,
  `vllm_gaudi/__init__.py`, `pyproject.toml`, `vllm_gaudi/ops/hpu_rotary_embedding.py`

**Upstream Mamba improvements (potential help for GDN):**
- `#1198` Prefix caching support for HPUMambaMixer2
- `#1175` Custom depthwise conv1d TPC kernel
- `#1203` Improved conv1d precision for bf16
- `#1210` Fancy indexing replaced with select+copy for Granite4 state
- `#984`/`#1077` Mamba prefix caching (added then reverted, then re-added in #1198)

---

## Recommended Approach: Fresh Port onto DeepSeek Branch

**NOT rebase. NOT cherry-pick of all 21 commits.**

Rationale:
1. Rebase of 21 commits across 97 upstream changes = conflict at nearly every commit, especially
   in hpu_model_runner.py. Each conflict must be resolved without context of future commits. Extremely
   error-prone.
2. Cherry-pick has the same problem -- commits 1-4 are already in DeepSeek and would double-apply.
3. The Qwen-specific logic (GDN state, model file, ops) is well-isolated. A fresh port carries
   the final working state of each file onto the v0.17.1 base, resolving hpu_model_runner.py
   once with full context of both the upstream changes and the GDN requirements.

**Base branch:** `a8/deepseek-v32-v0.17.1` (already has shared commits 1-4 rebased onto v0.17.1).

---

## Guardrails

### Must Have
- All Qwen3.5 GDN functionality preserved (per-layer Mamba state, chunk-parallel prefill,
  fused decode, multi-request handling, solve_triangular)
- DeepSeek functionality unchanged (no regressions)
- Clean commit history on top of v0.17.1
- Working pip install script for Qwen use case
- Model loads and generates coherent output (at minimum: "What is 2+2?" produces `<think>`)

### Must NOT Have
- Do NOT carry over `.omc/state/` files or `.bak` files
- Do NOT modify `vllm-base/` files (deepseek_sparse is already handled in DeepSeek branch)
- Do NOT rebase -- use fresh port approach
- Do NOT combine DeepSeek and Qwen into one mega-branch (keep them as separate feature branches)

---

## Task Flow

### Phase 1: Create Clean Branch and Port New Files (LOW RISK)

**Step 1: Branch from DeepSeek and copy pure-new files**

Create branch `a8/qwen35-v0.17.1` from `a8/deepseek-v32-v0.17.1`.

Copy these files verbatim from the qwen branch (they are new, no merge needed):
- `vllm_gaudi/models/qwen3_next.py`
- `vllm_gaudi/ops/hpu_gdn.py`
- `vllm_gaudi/ops/hpu_gdn_attn.py`
- `vllm_gaudi/ops/gdn_diagnostics.py`

Acceptance criteria:
- [ ] New branch exists based on DeepSeek tip (`9324e76`)
- [ ] All 4 new files present and identical to qwen branch versions
- [ ] `git diff a8/deepseek-v32-v0.17.1` shows only additions (no modifications to existing files)

### Phase 2: Port Modifications to Shared Files (MEDIUM RISK)

**Step 2: Apply Qwen-specific changes to shared files**

For each of the 12 shared-modified files, diff the qwen branch against the DeepSeek branch
to identify ONLY the Qwen-specific additions (filtering out the shared commits 1-4 that
DeepSeek already has).

Key changes to port:
- `vllm_gaudi/models/__init__.py` -- add Qwen3Next model registration
- `vllm_gaudi/extension/features.py` -- add qwen3_5_moe feature flags
- `vllm_gaudi/ops/causal_conv1d_pytorch.py` -- HPU causal conv1d additions (NOTE: upstream
  added depthwise conv1d TPC in #1175/#1203; check if the qwen additions are still needed
  or can use the upstream TPC kernel)
- `vllm_gaudi/extension/ops.py` -- GDN op registrations
- `vllm_gaudi/platform.py` -- Qwen3.5 platform support

Strategy: Use `git diff a8/deepseek-v32-v0.17.1..a8/gaudi-qwen35moe-v0.17.1 -- <file>`
to see the pure Qwen delta, then manually apply only those changes to the v0.17.1 version
of each file.

Acceptance criteria:
- [ ] Each file compiles (no syntax errors)
- [ ] DeepSeek-specific code untouched (diff against DeepSeek branch shows only Qwen additions)
- [ ] `import vllm_gaudi` succeeds
- [ ] Qwen3Next appears in model registry

### Phase 3: Port hpu_model_runner.py (HIGH RISK -- The Hard Part)

**Step 3: Integrate GDN state management into v0.17.1 hpu_model_runner.py**

This is the critical file. The v0.17.1 version has been substantially refactored since
the qwen branch diverged. The Qwen changes to this file are:

1. **Per-layer Mamba state allocation** (the garbage output fix) -- replaces naive shared
   cache with per-layer allocation using `num_blocks // num_shared`
2. **GDN metadata translation** -- `_build_gdn_metadata()` converts HPUAttentionMetadataV1
   to GDN-compatible format
3. **State indices tensor** -- 2D `[num_groups, batch]` for GDN state block indices
4. **Multi-request state handling** -- clone state tensors during prefill after HPUGraph,
   forward_hook mark_step management

Approach:
1. Start from the v0.17.1 version (as in DeepSeek branch)
2. Read the Qwen branch's version to understand WHAT each change does
3. Apply each logical change to the new codebase, adapting to the refactored structure
4. Pay special attention to upstream Mamba improvements (#1198 prefix caching, #1175 conv1d TPC)
   which may already provide some of what the Qwen branch needed

Key upstream changes to leverage:
- Upstream MambaMixer2 prefix caching (#1198) may simplify the per-layer state allocation
- Upstream depthwise conv1d TPC kernel (#1175) may replace custom causal_conv1d_pytorch additions
- Upstream Granite4 state handling (#1210) uses select+copy pattern similar to GDN needs

Acceptance criteria:
- [ ] File has no syntax errors
- [ ] VLLM_USE_NAIVE_MAMBA_CACHE_SHARING correctly handled for GDN (per-layer state)
- [ ] GDN metadata translation present and adapted to v0.17.1 metadata structures
- [ ] Multi-request state cloning and mark_step management present
- [ ] DeepSeek-specific hpu_model_runner changes from DeepSeek branch preserved

### Phase 4: Create Install Script and Verify (LOW RISK)

**Step 4: Create `install-vllm-hpu-qwen.sh`**

Based on the existing `install-vllm-hpu.sh` pattern. Key differences from the stock script:
- Branch checkout or verification (ensure vllm-gaudi is on the qwen branch)
- Same 4-step flow: verify HPU PyTorch, install vLLM base, install vllm_gaudi, verify
- Additional verification: check Qwen3Next model is registered
- Optional: integrate fused TPC kernel build (from `install-vllm-hpu-fused.sh` pattern)
  if the fused GDN+MoE kernels apply to Qwen

Acceptance criteria:
- [ ] Script runs end-to-end in Gaudi container without errors
- [ ] `python3 -c "import vllm_gaudi"` passes
- [ ] Qwen3Next model registered: `python3 -c "from vllm_gaudi.models import qwen3_next"`
- [ ] Script is idempotent (can run twice safely)

### Phase 5: End-to-End Validation (MEDIUM RISK)

**Step 5: Load model and generate output**

Test inside Gaudi container with the actual model at `/models/Qwen3.5/Qwen3.5-35B-A3B`:

1. Start vLLM server with Qwen3.5 config (adapt from existing start scripts)
2. Send test prompt "What is 2+2?" via `/v1/chat/completions`
3. Verify response begins with `<think>` (Qwen3.5 reasoning format)
4. Verify no garbage output (the original Mamba cache sharing bug)
5. Test multi-request handling (2+ concurrent requests)

Acceptance criteria:
- [ ] Server starts without OOM or crash
- [ ] Single request produces coherent output starting with `<think>`
- [ ] Multiple sequential requests produce coherent output
- [ ] No per-layer state corruption (garbage output bug stays fixed)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| hpu_model_runner.py port breaks GDN state | HIGH | HIGH | Test per-layer state with "2+2" prompt immediately after port |
| Upstream Mamba refactor changes state allocation API | MEDIUM | HIGH | Read #1198 and #1210 code carefully before porting |
| causal_conv1d conflicts with upstream TPC kernel | MEDIUM | LOW | Compare implementations; prefer upstream TPC if equivalent |
| DeepSeek regression from shared file changes | LOW | HIGH | Run DeepSeek test after each shared file modification |
| Install script fails due to branch-specific deps | LOW | LOW | Test in clean container |

---

## Success Criteria

1. Branch `a8/qwen35-v0.17.1` exists with clean commits on top of DeepSeek branch tip
2. All Qwen3.5 GDN files ported and functional
3. hpu_model_runner.py correctly handles both DeepSeek and Qwen3.5 models
4. `install-vllm-hpu-qwen.sh` works end-to-end in Gaudi container
5. Model generates coherent `<think>` output for test prompts
