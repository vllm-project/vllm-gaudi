# Benchmark Environment Setup

## Overview

This document describes how to prepare two code-base checkpoints for
benchmarking the three bucketing strategies in vllm-gaudi:

| Checkpoint | Bucketing Strategies | Git Ref |
|---|---|---|
| **Baseline** | Exponential, Linear | `main` (`05cfbe69e795dfb68e7ba1232352a74c3b874dec`) |
| **Linear-with-limits** | Exponential, Linear, Linear-with-limits | `main` + PR #762 |

## Prerequisites

- Intel Gaudi HPU hardware (Gaudi 2 or Gaudi 3)
- Python 3.10+
- Habana software stack installed (SynapseAI, habana-frameworks, PyTorch for HPU)
- `git` with worktree support (Git 2.5+)

## Quick Start

### Option 1: Automated Setup (recommended)

The `setup_benchmark_env.py` script creates isolated worktrees and virtualenvs
for both checkpoints, installs vllm-gaudi in each, runs smoke tests, and
writes a full environment manifest to `benchmark_env.json`.

```bash
# 1. Fetch PR #762 (if not already available locally)
git fetch origin pull/762/head:pr762

# 2. Run the setup script
python tools/benchmark/setup_benchmark_env.py \
    --repo-dir . \
    --env-dir /tmp/bench_envs \
    --baseline-ref main \
    --pr762-ref pr762

# 3. Check the generated manifest
cat tools/benchmark/benchmark_env.json
```

### Option 2: In-place Mode

If you already have a single virtualenv on Gaudi hardware and just want to
record the environment details:

```bash
python tools/benchmark/setup_benchmark_env.py --in-place
```

This writes `benchmark_env.json` without creating any virtualenvs.

### Option 3: Manual Setup

```bash
# --- Checkpoint 1: Baseline (exponential + linear) ---
BASELINE_SHA="05cfbe69e795dfb68e7ba1232352a74c3b874dec"
git worktree add /tmp/bench_baseline "${BASELINE_SHA}"
python -m venv /tmp/venv_baseline
source /tmp/venv_baseline/bin/activate
pip install -e /tmp/bench_baseline
python -c "import vllm_gaudi; print('OK')"
deactivate

# --- Checkpoint 2: Baseline + PR #762 (linear-with-limits) ---
git fetch origin pull/762/head:pr762
git worktree add /tmp/bench_lwl pr762
python -m venv /tmp/venv_lwl
source /tmp/venv_lwl/bin/activate
pip install -e /tmp/bench_lwl
python -c "import vllm_gaudi; print('OK')"
deactivate
```

## Commit SHAs

| Checkpoint | SHA | How to Obtain |
|---|---|---|
| Baseline (main) | `05cfbe69e795dfb68e7ba1232352a74c3b874dec` | `git rev-parse main` |
| Linear-with-limits | *(PR #762 SHA)* | `git fetch origin pull/762/head:pr762 && git rev-parse pr762` |

## Environment Details Captured

The `benchmark_env.json` manifest records the following for reproducibility:

| Field | Source |
|---|---|
| Baseline commit SHA | `git rev-parse main` |
| PR #762 commit SHA | `git rev-parse pr762` |
| Gaudi device type | `hl-smi` output |
| Driver version | `hl-smi` or `modinfo habanalabs` |
| Python version | `platform.python_version()` |
| PyTorch version | `torch.__version__` |
| Habana SW stack version | `hl-smi -v` or `habana_frameworks.__version__` |
| OS info | `platform.system()`, `platform.release()` |
| Timestamp | ISO 8601 UTC |

## Bucketing Strategies

### Exponential (default)

- **Source**: `vllm_gaudi/extension/bucketing/exponential.py`
- **Env vars**: `VLLM_EXPONENTIAL_BUCKETING=true` (or unset)
- Uses exponential spacing for warmup buckets with a configurable limit
  parameter that controls the number of buckets.
- Available in both checkpoints.

### Linear

- **Source**: `vllm_gaudi/extension/bucketing/linear.py`
- **Env vars**: `VLLM_EXPONENTIAL_BUCKETING=false`
- Uses linear spacing with a ramp-up phase (doubling from min until step,
  then stepping linearly to max).
- Available in both checkpoints.

### Linear with Limits

- **Source**: `vllm_gaudi/extension/bucketing/linear.py` (extended by PR #762)
- **Env vars**: `VLLM_EXPONENTIAL_BUCKETING=false` and `VLLM_LINEAR_WITH_LIMITS=true`
- Extends linear bucketing with absolute/relative padding limits to reduce
  wasted computation from over-padded buckets.
- **Only available in the linear-with-limits checkpoint** (requires PR #762).

## Verification

After setup, verify each checkpoint is functional:

```bash
# Baseline
/tmp/bench_envs/venv_baseline/bin/python -c "
from vllm_gaudi.extension.bucketing.exponential import ExponentialBucketingStrategy
from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
print('Baseline: exponential OK, linear OK')
"

# Linear-with-limits
/tmp/bench_envs/venv_linear_with_limits/bin/python -c "
from vllm_gaudi.extension.bucketing.exponential import ExponentialBucketingStrategy
from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
print('Linear-with-limits: all strategies OK')
"
```

## Next Steps

Once both checkpoints are installed and verified, proceed to
run the benchmark suite — see the main benchmark README for details.
