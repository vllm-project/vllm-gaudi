---
title: Row-Parallel Chunking
---
[](){ #row_parallel_chunking }

## Overview

Row-Parallel Chunking is an optimization method for tensor-parallel inference on Intel® Gaudi® that overlaps computation with communication in `RowParallelLinear` layers. In standard tensor-parallel inference, each `RowParallelLinear` layer performs a matrix multiplication followed by a blocking all-reduce across all TP ranks.

## How It Works

In a standard `RowParallelLinear` forward pass, the execution is sequential:

1. Compute `output = matmul(input, weight)`
2. All-reduce `output` across TP ranks (blocking)

With chunking enabled, the forward pass becomes:

1. Split the input into *N* chunks along the token dimension
2. For each chunk *i*:
   - Compute `output_i = matmul(input_i, weight)`
   - Launch `all_reduce(output_i)` **asynchronously**
3. Wait for all async all-reduce operations to complete
4. Concatenate the chunk outputs

This pipelining allows the all-reduce of chunk *i* to run concurrently with the matmul of chunk *i+1*, reducing idle time on both the compute engine and the network interface.

### Chunking Conditions

Chunking is only applied when **all** of the following conditions are met:

- `num_chunks > 1` (chunking is configured)
- `reduce_results` is enabled on the layer (all-reduce is needed)
- `tp_size > 1` (running with tensor parallelism)
- `total_tokens >= chunk_threshold` (input is large enough to benefit)

When any condition is not met, the layer falls back to the standard single-shot computation.

### Input Handling

The implementation handles both 2D `[tokens, hidden]` and 3D `[batch, seq_len, hidden]` inputs:

- **3D with `seq_len > 1`** (prefill): chunks along the sequence dimension
- **3D with `seq_len == 1`** (decode): chunks along the batch dimension
- **2D**: chunks along the first (token) dimension

## Configuration

The feature is controlled by two environment variables:

| Environment Variable | Config Name | Default | Description |
|---|---|---|---|
| `VLLM_ROW_PARALLEL_CHUNKS` | `row_parallel_chunks` | `1` (disabled) | Number of chunks to split the input into. Set to a value > 1 to enable. |
| `VLLM_ROW_PARALLEL_CHUNK_THRESHOLD` | `row_parallel_chunk_threshold` | `8192` | Minimum number of tokens required to activate chunking. Inputs below this threshold use the standard path. |

### Example Usage

```bash
# Enable chunking with 8 chunks (default threshold of 8192 tokens)
export VLLM_ROW_PARALLEL_CHUNKS=8

# Enable chunking with 16 chunks and a lower threshold
export VLLM_ROW_PARALLEL_CHUNKS=16
export VLLM_ROW_PARALLEL_CHUNK_THRESHOLD=4096
```

## Performance Characteristics

The table below shows the **speedup ratio** (chunked time / baseline time, where values > 1.0 indicate speedup) for an isolated `RowParallelLinear` layer measured across different TP sizes, chunk counts, and token counts.

### TP Size = 2

| Tokens | 2 chunks | 4 chunks | 8 chunks | 16 chunks | 32 chunks | 64 chunks |
|-------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|
| 1024   | 1.089    | 0.811    | 0.574    | 0.308     | 0.207     | 0.108     |
| 2048   | 1.076    | 1.161    | 0.753    | 0.477     | 0.334     | 0.176     |
| 4096   | 1.322    | 1.393    | 1.431    | 0.810     | 0.656     | 0.351     |
| 8192   | 1.158    | 1.482    | 1.453    | 1.204     | 1.067     | 0.620     |
| 16384  | 1.239    | 1.316    | 1.589    | 1.489     | 1.514     | 1.075     |
| 32768  | 1.246    | 1.460    | 1.434    | 1.649     | 1.563     | 1.569     |
| 65536  | 1.246    | 1.424    | 1.580    | 1.483     | 1.555     | 1.548     |
| 131072 | 1.268    | 1.442    | 1.503    | 1.676     | 1.533     | 1.514     |

### TP Size = 4

| Tokens | 2 chunks | 4 chunks | 8 chunks | 16 chunks | 32 chunks | 64 chunks |
|-------:|:--------:|:--------:|:--------:|:---------:|:---------:|:---------:|
| 1024   | 0.892    | 0.579    | 0.374    | 0.195     | 0.104     | 0.060     |
| 2048   | 1.035    | 0.888    | 0.509    | 0.307     | 0.142     | 0.088     |
| 4096   | 1.156    | 1.081    | 0.795    | 0.466     | 0.245     | 0.134     |
| 8192   | 1.171    | 1.304    | 1.255    | 0.749     | 0.485     | 0.244     |
| 16384  | 1.162    | 1.309    | 1.416    | 1.216     | 0.780     | 0.496     |
| 32768  | 1.118    | 1.237    | 1.280    | 1.427     | 1.201     | 0.766     |
| 65536  | 1.218    | 1.310    | 1.386    | 1.528     | 1.553     | 1.195     |
| 131072 | 1.193    | 1.387    | 1.332    | 1.353     | 1.560     | 1.545     |

### TP Size = 8

| Tokens | 2 chunks | 8 chunks | 16 chunks | 32 chunks | 64 chunks |
|-------:|:--------:|:--------:|:---------:|:---------:|:---------:|
| 1024   | 0.656    | 0.253    | 0.132     | 0.075     | 0.045     |
| 2048   | 0.828    | 0.324    | 0.183     | 0.087     | 0.051     |
| 4096   | 0.919    | 0.495    | 0.264     | 0.146     | 0.078     |
| 8192   | 0.993    | 0.705    | 0.470     | 0.240     | 0.157     |
| 16384  | 0.990    | 1.024    | 0.684     | 0.402     | 0.249     |
| 32768  | 0.972    | 1.118    | 0.942     | 0.760     | 0.469     |
| 65536  | 0.989    | 1.164    | 1.129     | 1.179     | 0.758     |
| 131072 | 1.018    | 1.241    | 1.277     | 1.297     | 1.090     |

### Reading the Results

- **Values > 1.0** indicate a speedup over the non-chunked baseline (e.g., 1.5 = 50% faster).
- **Values < 1.0** indicate a slowdown due to chunking overhead exceeding the overlap benefit.

### Key Takeaways

1. **Sweet spot shifts with TP size and seq_len.** There is no single setting that will benefit all benchmarks so depending on the TP size and seq_len user should experiment which setting benefits him the most. Good starting point is 2 chunks and 4k threshold.

2. **Diminishing returns with too many chunks.** Excessively fine chunking introduces overhead from graph breaks, kernel launch latency, and reduced per-chunk compute efficiency. The optimal chunk count depends on both TP size and typical token count.

## Recommended Settings

| TP Size | Recommended Chunks | Notes |
|:-------:|:------------------:|-------|
| 1       | 1 (disabled)       | No all-reduce needed, chunking adds overhead only |
| 2       | 2-64               | Beneficial for token counts ≥ 4096 |
| 4       | 2-16               | Beneficial for token counts ≥ 8192 |
| 8       | 2-8                | Beneficial for token counts ≥ 16384 |

> **Note:** These recommendations are based on isolated layer benchmarks using llama 3.3 70b. In end-to-end inference, the optimal configuration may differ depending on the model architecture, sequence lengths, and workload mix. It is recommended to benchmark with your specific setup.

## When to Use This Feature

**Use when:**
- Running with tensor parallelism (`TP > 1`)
- Serving workloads with large batch sizes or long prefill sequences
- The all-reduce communication is a significant fraction of the prefill time

**Do not use when:**
- Running without tensor parallelism (`TP = 1`) — there is no all-reduce to overlap
- Serving only very short sequences with small batches — chunking overhead outweighs the benefit

## Implementation Details

The feature is implemented in `vllm_gaudi/ops/hpu_row_parallel_linear.py` as `HPURowParallelLinear`, which registers as an OOT (out-of-tree) override for vLLM's `RowParallelLinear`. The chunking logic is entirely self-contained in the `forward` method and does not modify any other part of the model or the inference pipeline.

Each chunk boundary introduces a `torch._dynamo.graph_break()` to ensure correct async all-reduce semantics under `torch.compile`. This means the compiled graph will be split at chunk boundaries, which is a necessary trade-off for enabling async communication.
