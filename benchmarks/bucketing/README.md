# Bucketing Strategy Benchmarks

## Overview

This benchmark suite compares three bucketing strategies for HPU graph
capture in vLLM on Intel Gaudi hardware:

- **Exponential** — the default strategy using power-of-two bucket sizes
- **Linear** — evenly spaced bucket sizes
- **Linear with limits** — linear buckets constrained by explicit min/max limits

Each strategy is evaluated across two models:

- **Qwen3-32B** (dense)
- **Qwen3-30B-A3B** (MoE)

The benchmarks measure warmup duration (graph compilation) and runtime
inference performance across a sweep of input lengths and concurrency
levels, producing a comparison report.

## Prerequisites

- Intel Gaudi hardware (HPU)
- `vllm-gaudi` installed — see the [repo root README](../../README.md)
  for installation instructions
- For the `linear_with_limits` strategy: ensure changes from PR #762 are
  included in your installation
- Models downloaded and accessible (e.g. via `huggingface-cli download`)

## Environment Configuration

The bucketing strategy is selected through environment variables that
`run_server.py` sets automatically based on the `--strategy` flag:

| Strategy | Environment Variables |
|---|---|
| `exponential` (default) | `VLLM_EXPONENTIAL_BUCKETING=true` (or unset) |
| `linear` | `VLLM_EXPONENTIAL_BUCKETING=false` |
| `linear_with_limits` | `VLLM_EXPONENTIAL_BUCKETING=false` + `VLLM_USE_BUCKET_LIMITS=true` |

## Quick Start

Run the full benchmark matrix (all models × all strategies) with a single
command:

```bash
bash benchmarks/bucketing/run_all.sh
```

Customise output directory and tensor-parallel size:

```bash
BENCHMARK_OUTPUT_DIR=./my_results TP_SIZE=4 bash benchmarks/bucketing/run_all.sh
```

Additional tunables exposed by `run_all.sh`:

| Variable | Default |
|---|---|
| `BENCHMARK_OUTPUT_DIR` | `benchmarks/bucketing/results` |
| `BENCHMARK_PORT` | `8000` |
| `TP_SIZE` | `2` |
| `MAX_NUM_SEQS` | `128` |
| `MAX_MODEL_LEN` | `131072` |
| `MAX_NUM_BATCHED_TOKENS` | `8192` |
| `INPUT_LENS` | `2048,8192,32768,98304,114688` |

## Individual Scripts

### `run_server.py`

Launch a vLLM server configured for a specific bucketing strategy.

```bash
python benchmarks/bucketing/run_server.py \
    --model Qwen/Qwen3-32B \
    --strategy exponential \
    --tensor-parallel-size 2 \
    --output-dir benchmarks/bucketing/results
```

The script writes `server.log` and `server_meta.json` into a
per-configuration subdirectory under the output directory.

### `run_client.py`

Run the client benchmark sweep against a running server. Requires the
`server_meta.json` produced by `run_server.py`.

```bash
python benchmarks/bucketing/run_client.py \
    --server-meta benchmarks/bucketing/results/qwen3-32b_exponential/server_meta.json \
    --input-lens 2048,8192,32768 \
    --output-len 1024
```

Results are saved as `results.csv` and `results.json` alongside the
server logs.

### `analyze_results.py`

Aggregate results across configurations and produce a comparison report.

```bash
python benchmarks/bucketing/analyze_results.py \
    --results-dir benchmarks/bucketing/results \
    --output benchmarks/bucketing/report
```

## Output Structure

```text
results/
├── qwen3-32b_exponential/
│   ├── server.log
│   ├── server_meta.json
│   ├── client_input2048_conc1.log
│   ├── ...
│   ├── results.csv
│   └── results.json
├── qwen3-32b_linear/
│   └── ...
├── qwen3-32b_linear_with_limits/
│   └── ...
├── qwen3-30b-a3b_exponential/
│   └── ...
└── report/
    ├── warmup_comparison.csv
    ├── performance_comparison.csv
    └── report.md
```

## Metrics Collected

- **Warmup duration** — time spent compiling HPU graphs at server startup
- **HPU blocks** — number of KV-cache blocks allocated on the device
- **Request throughput** — requests completed per second
- **Mean / P99 TTFT** — time to first token (mean and 99th percentile)
- **Mean / P99 TPOT** — time per output token (mean and 99th percentile)
- **Mean / P99 end-to-end latency** — total request latency (mean and 99th percentile)
