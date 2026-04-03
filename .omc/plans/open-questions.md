# Open Questions

## qwen35-v0171-migration - 2026-03-27

- [ ] Should the new branch be based on DeepSeek (`a8/deepseek-v32-v0.17.1`) or directly on `upstream/releases/v0.17.1` with shared commits re-applied? -- DeepSeek base avoids redoing shared work but couples the branches.
- [ ] Does upstream's MambaMixer2 prefix caching (#1198) provide per-layer state allocation that replaces the custom GDN fix, or is the GDN fix still needed on top? -- Determines scope of hpu_model_runner.py porting work.
- [ ] Does upstream's depthwise conv1d TPC kernel (#1175, #1203) replace the custom `causal_conv1d_pytorch.py` additions, or is additional logic needed for GDN? -- May simplify the port if upstream kernel is sufficient.
- [ ] Should the install script include fused TPC kernel build (like `install-vllm-hpu-fused.sh`) or keep it stock? -- Depends on whether fused GDN+MoE kernels are ready for Qwen use case.
- [ ] What server start configuration should be used for Qwen3.5 on v0.17.1? -- The existing config (`VLLM_SKIP_WARMUP=true`, `gpu_memory_utilization=0.7`) may need adjustment for the new upstream base.
- [ ] Should `.bak` file (`qwen3_next.py.bak`) be carried over? -- Almost certainly no, but confirming with user.
