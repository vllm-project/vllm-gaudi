# Qwen3.5 / Qwen3.8 hybrid-GDN — speculative decode support

## MTP unsupported for hybrid-GDN: guard

MTP / speculative decode is **unsupported for hybrid-GDN models on HPU**
(e.g. Qwen3.5-35B, Qwen3.8-27B-FP8). The GDN recurrent state keeps one live
slot per request with no accepted-prefix commit; enabling speculative
decode either crashes or produces divergent output.

**Guard:** `vllm_gaudi/v1/worker/hpu_model_runner.py` disables any
`speculative_config` when `num_gdn > 0` unless `VLLM_ALLOW_GDN_SPEC=1`
is set. When triggered it logs a warning naming `num_gdn` and the escape
hatch, and nulls both `self.speculative_config` and
`vllm_config.speculative_config` before drafter init. Default
`VLLM_ALLOW_GDN_SPEC=0`; escape `VLLM_ALLOW_GDN_SPEC=1` re-enters the
unverified path with no correctness guarantee.

See also issue #2022 for the underlying wall.
