# Bucketing Strategy Benchmark Suite

## Overview

This suite benchmarks three bucketing strategies — **exponential**, **linear**, and
**linear-with-limits** — for vllm-gaudi. It automates server startup, warmup
measurement, and inference benchmarking so you can compare warmup time and
inference performance across different models, input lengths, and concurrency
levels.

## Prerequisites

- **Intel Gaudi HPU hardware** (e.g. Gaudi 2 or Gaudi 3)
- **vllm-gaudi** installed (latest release)
- **Python 3.10+**
- For the `linear_with_limits` strategy: [PR #762](https://github.com/vllm-project/vllm-gaudi/pull/762)
  must be merged or cherry-picked into your build

## Bucketing Strategies

| Strategy | Description | Environment Variables | Source |
|---|---|---|---|
| **Exponential** (default) | Uses exponential spacing for warmup buckets. This is the default when no env vars are set. | `VLLM_EXPONENTIAL_BUCKETING=true` (or unset) | `vllm_gaudi/extension/bucketing/exponential.py` |
| **Linear** | Uses linear spacing with ramp-up. | `VLLM_EXPONENTIAL_BUCKETING=false` | `vllm_gaudi/extension/bucketing/linear.py` |
| **Linear with limits** | Linear bucketing with absolute/relative padding limits (requires PR #762). | `VLLM_EXPONENTIAL_BUCKETING=false` and `VLLM_LINEAR_WITH_LIMITS=true` | `vllm_gaudi/extension/bucketing/linear.py` |

## Quick Start

```bash
# Run full benchmark suite
python tools/benchmark/run_bucketing_benchmark.py \
  --models Qwen/Qwen3-32B Qwen/Qwen3-30B-A3B \
  --strategies exponential linear linear_with_limits \
  --output-dir benchmark_results

# Generate comparison report
python tools/benchmark/analyze_results.py \
  --results-dir benchmark_results \
  --output benchmark_results/report.md
```

## Configuration

### `run_bucketing_benchmark.py`

| Argument | Default | Description |
|---|---|---|
| `--models` | `Qwen/Qwen3-32B Qwen/Qwen3-30B-A3B` | List of model names to benchmark. |
| `--strategies` | `exponential linear linear_with_limits` | Bucketing strategies to test. Choices: `exponential`, `linear`, `linear_with_limits`. |
| `--input-lengths` | `2048 8192 32768 98304 114688` | List of input lengths to benchmark. |
| `--output-len` | `1024` | Random output length. |
| `--random-range-ratio` | `0.1` | Random range ratio for output length variation. |
| `--max-num-seqs` | `128` | Max number of sequences (passed to `vllm serve`). |
| `--max-model-len` | `131072` | Max model length (passed to `vllm serve`). |
| `--max-num-batched-tokens` | `8192` | Max number of batched tokens (passed to `vllm serve`). |
| `--tensor-parallel-size` | `2` | Tensor parallel size (passed to `vllm serve`). |
| `--num-prompts-multiplier` | `10` | Multiplier for `num_prompts = multiplier * max_concurrency`. |
| `--host` | `localhost` | Server host. |
| `--port` | `8000` | Server port. |
| `--output-dir` | `benchmark_results` | Directory for results and logs. |
| `--server-start-timeout` | `600` | Seconds to wait for the server to become ready. |
| `--vllm-binary` | `vllm` | Path to the vllm binary. |

### `analyze_results.py`

| Argument | Default | Description |
|---|---|---|
| `--results-dir` | `benchmark_results` | Path to the benchmark results directory (must contain `results.jsonl`). |
| `--output` | `<results-dir>/report.md` | Path for the output report file. |

## Benchmark Matrix

The default configuration sweeps the following parameters:

- **Models**: `Qwen/Qwen3-32B`, `Qwen/Qwen3-30B-A3B`
- **Input lengths**: 2048, 8192, 32768, 98304, 114688
- **Output length**: 1024 (with `random_range_ratio=0.1`)
- **Concurrency**: powers of 2 from 1 up to the KV-cache-derived maximum
  (see [Concurrency Calculation](#concurrency-calculation) below)
- **Server params**: `--max-num-seqs 128 --max-model-len 131072
  --max-num-batched-tokens 8192 --tensor-parallel-size 2`

For each model × strategy combination a fresh vllm server is launched, warmup
metrics are recorded, and then every input-length × concurrency-level
combination is benchmarked.

## Output Files

| File / Directory | Description |
|---|---|
| `results.jsonl` | One JSON object per benchmark run, appended after each run for crash safety. |
| `results.csv` | Same data in CSV format, written once all runs complete. |
| `logs/` | Server logs for each model/strategy combination (e.g. `Qwen_Qwen3-32B_exponential.log`). |
| `raw/` | Raw stdout/stderr from each `vllm bench serve` invocation. |
| `report.md` | Generated comparison report (produced by `analyze_results.py`). |

## Concurrency Calculation

The maximum concurrency for a given input length is computed by doubling from 1
until the next doubling would exceed the available KV cache:

```
max_concurrency = 1
tokens_per_request = (input_len + output_len) * (1 + random_range_ratio)

while max_concurrency * 2 * tokens_per_request < KV_cache_size:
    max_concurrency *= 2
```

The KV cache size is extracted from the server log line:

```
GPU KV cache size: <N> tokens
```

The benchmark then runs at every power-of-2 concurrency level from 1 up to and
including `max_concurrency` (i.e. `1, 2, 4, ..., max_concurrency`).
