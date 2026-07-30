#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit a GPQA eval config as shell `export` lines.

Reads the declarative Kimi config yaml (same convention as
.jenkins/lm-eval-harness/configs) and prints CFG_* variables that the
server / client / accuracy-check scripts source. Keeping the parsing in one
place means the yaml is the single source of truth.

Usage:  eval "$(python3 config_env.py <config.yaml>)"
"""
import sys
from pathlib import Path

import yaml


def sh_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def main():
    if len(sys.argv) != 2:
        print("usage: config_env.py <config.yaml>", file=sys.stderr)
        sys.exit(2)
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))

    tasks = cfg.get("tasks", [])
    if not tasks:
        print("config has no tasks", file=sys.stderr)
        sys.exit(2)
    task = tasks[0]

    out = {}
    out["CFG_MODEL"] = cfg["model_name"]
    out["CFG_TASK"] = task["name"]
    out["CFG_TP_SIZE"] = cfg.get("tensor_parallel_size", 1)
    out["CFG_MAX_MODEL_LEN"] = cfg.get("max_model_len", 16384)
    out["CFG_DTYPE"] = cfg.get("dtype", "bfloat16")
    out["CFG_LIMIT"] = cfg.get("limit", 16)
    out["CFG_NUM_CONCURRENT"] = cfg.get("num_concurrent", 8)
    out["CFG_GEN_KWARGS"] = cfg.get("gen_kwargs", "temperature=0.0,max_gen_toks=8192")
    out["CFG_ENABLE_EP"] = "1" if cfg.get("enable_expert_parallel") else "0"
    out["CFG_TRUST_REMOTE_CODE"] = "1" if cfg.get("trust_remote_code") else "0"
    out["CFG_THINKING"] = "1" if cfg.get("thinking") else "0"

    # metric=threshold specs, space-separated, for check_accuracy.py
    specs = []
    for m in task.get("metrics", []):
        specs.append(f"{m['name']}={m['value']}")
    out["CFG_METRIC_SPECS"] = " ".join(specs)

    for key, value in out.items():
        print(f"export {key}={sh_quote(value)}")


if __name__ == "__main__":
    main()
