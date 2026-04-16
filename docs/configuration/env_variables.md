# Environment Variables

This document lists the supported diagnostic and profiling, as well as performance tuning options.

## Diagnostic and Profiling Parameters

| Parameter name                            | Description                                                                                                                                                                                                                                                                                                                                                                                                             | Default value |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `VLLM_PROFILER_ENABLED`                   | Enables the high-level profiler. You can view resulting JSON traces at [perfetto.habana.ai](https://perfetto.habana.ai/#!/viewer).                                                                                                                                                                                                                                                                                      | `false`       |
| `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION`     | Logs graph compilations for each vLLM engine step, only when a compilation occurs. We recommend using it in conjunction with `PT_HPU_METRICS_GC_DETAILS=1`.                                                                                                                                                                                                                                                             | `false`       |
| `VLLM_HPU_LOG_STEP_GRAPH_COMPILATION_ALL` | Logs graph compilations for every vLLM engine step, even if no compilation occurs.                                                                                                                                                                                                                                                                                                                                      | `false`       |
| `VLLM_HPU_LOG_STEP_CPU_FALLBACKS`         | Logs CPU fallbacks for each vLLM engine step, only when a fallback occurs.                                                                                                                                                                                                                                                                                                                                              | `false`       |
| `VLLM_HPU_LOG_STEP_CPU_FALLBACKS_ALL`     | Logs CPU fallbacks for each vLLM engine step, even if no fallback occurs.                                                                                                                                                                                                                                                                                                                                               | `false`       |
| `VLLM_T_COMPILE_FULLGRAPH`                | Forces the PyTorch compile function to raise an error if any graph breaks happen during compilation. This allows for the easy detection of existing graph breaks, which usually reduce performance.                                                                                                                                                                                                                     | `false`       |
| `VLLM_T_COMPILE_DYNAMIC_SHAPES`           | Forces PyTorch to compile graphs with disabled dynamic options to use dynamic shapes only when needed.                                                                                                                                                                                                                                                                                                                  | `false`       |
| `VLLM_FULL_WARMUP`                        | Forces PyTorch to assume that the warm-up phase fully covers all possible tensor sizes, preventing further compilation. If compilation occurs after warm-up, PyTorch will crash (with this message: `Recompilation triggered with skip_guard_eval_unsafe stance. This usually means that you have not warmed up your model with enough inputs such that you can guarantee no more recompilations.`) and must be disabled. | `false`       |

## Performance Tuning Parameters

| Parameter name               | Description                                                   | Default value |
| ---------------------------- | ------------------------------------------------------------- | ------------- |
| `VLLM_GRAPH_RESERVED_MEM`    | Percentage of memory dedicated to HPUGraph capture.           | `0.1`         |
| `VLLM_EXPONENTIAL_BUCKETING` | Enables exponential bucket spacing instead of linear spacing. | `true`        |
| `VLLM_BUCKETING_FROM_FILE`   | Enables reading bucket configuration from file | `None`        |
| `VLLM_ROW_PARALLEL_CHUNKS`   | Number of chunks to split input into for pipelining matmul with all-reduce in RowParallelLinear layers. Setting to a value greater than 1 enables chunking. See [Row-Parallel Chunking](../features/row_parallel_chunking.md). | `1` (disabled) |
| `VLLM_ROW_PARALLEL_CHUNK_THRESHOLD` | Minimum number of tokens required to activate row-parallel chunking. Inputs below this threshold use the standard non-chunked path. | `8192` |
| `VLLM_LONG_CONTEXT_SPLIT_THRESHOLD` | Token count threshold for context-aware batch splitting. Requests with total sequence length (context + scheduled tokens) above this threshold are isolated into separate sub-batches with a reduced batch size. `0` disables the feature. | `0` (disabled) |
| `VLLM_LONG_CONTEXT_MAX_BATCH_SIZE` | Maximum number of long-context requests allowed in a single prefill sub-batch when `VLLM_LONG_CONTEXT_SPLIT_THRESHOLD` is active. | `1` |
| `VLLM_SPLIT_POOLS_CONFIG` | JSON string defining split KV cache pools for mixed short/long context workloads. When set, automatically configures `VLLM_LONG_CONTEXT_SPLIT_THRESHOLD` and generates pool-specific HPU graph buckets during warmup. See [Split KV Cache Pools](#split-kv-cache-pools) below. | `None` (disabled) |

## Developer Mode Parameters

To enter developer mode use `VLLM_DEVELOPER_MODE`:

| Parameter name     | Description              | Default value |
| ------------------ | ------------------------ | ------------- |
| `VLLM_SKIP_WARMUP` | Skips the warm-up phase. | `false`       |

## Additional Parameters

| Parameter name                | Description                                                                                                                                                                                   | Default value |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `VLLM_HANDLE_TOPK_DUPLICATES` | Handles duplicates outside top-k.                                                                                                                                                             | `false`       |
| `VLLM_CONFIG_HIDDEN_LAYERS`   | Sets the number of hidden layers to run per HPUGraph for model splitting among hidden layers when TP is 1. It improves throughput by reducing inter-token latency limitations in some models. | `1`           |

HPU PyTorch bridge environment variables impacting vLLM execution:

| Parameter name                     | Description                                                                                                                                           | Default value                                    |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `PT_HPU_LAZY_MODE`                 | Sets the backend for Gaudi, with `0` for PyTorch Eager and `1` for PyTorch Lazy.                                                                      | `0`                                              |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES`   | Must be set to `true` for tensor parallel inference with HPU Graphs.                                                                                  | `true`                                           |
| `PT_HPUGRAPH_DISABLE_TENSOR_CACHE` | Must be set to `false` for LLaVA, Qwen, and RoBERTa models.                                                                                           | `false`                                          |
| `VLLM_PROMPT_USE_FLEX_ATTENTION`   | Enabled only for the Llama model, allowing usage of `torch.nn.attention.flex_attention` instead of FusedSDPA. Requires `VLLM_PROMPT_USE_FUSEDSDPA=0`. | `false`                                          |
| `RUNTIME_SCALE_PATCHING`           | Enables the runtime scale patching feature, which applies only to FP8 execution and is ignored for BF16.                                              | `true` (Torch Compile mode), `false` (Lazy mode) |

## Additional Performance Tuning Parameters for Linear Bucketing Strategy

`VLLM_{phase}_{dim}_BUCKET_{param}` is a collection of environment variables configuring ranges of linear bucketing mechanism, where:

- `{phase}` is either `PROMPT` or `DECODE`
- `{dim}` is either `BS`, `SEQ` or `BLOCK`
- `{param}` is either `MIN`, `STEP` or `MAX`

The following table lists the available variables with their default values:

| Phase  | Variable name                                     | Default value                                |
| ------ | ------------------------------------------------- | -------------------------------------------- |
| Prompt | batch size min (`VLLM_PROMPT_BS_BUCKET_MIN`)      | `1`                                          |
| Prompt | batch size step (`VLLM_PROMPT_BS_BUCKET_STEP`)    | `1`                                          |
| Prompt | batch size max (`VLLM_PROMPT_BS_BUCKET_MAX`)      | `max_num_prefill_seqs`                       |
| Prompt | query length min (`VLLM_PROMPT_SEQ_BUCKET_MIN`)   | `block_size`                                 |
| Prompt | query length step (`VLLM_PROMPT_SEQ_BUCKET_STEP`) | `block_size`                                 |
| Prompt | query length max (`VLLM_PROMPT_SEQ_BUCKET_MAX`)   | `max_num_batched_tokens`                     |
| Prompt | sequence ctx min (`VLLM_PROMPT_CTX_BUCKET_MIN`)   | `0`                                          |
| Prompt | sequence ctx step (`VLLM_PROMPT_CTX_BUCKET_STEP`) | `1`                                          |
| Prompt | sequence ctx max (`VLLM_PROMPT_CTX_BUCKET_MAX`)   | `(max_model_len - block_size) // block_size` |
| Decode | batch size min (`VLLM_DECODE_BS_BUCKET_MIN`)      | `1`                                          |
| Decode | batch size step (`VLLM_DECODE_BS_BUCKET_STEP`)    | `32`                                         |
| Decode | batch size max (`VLLM_DECODE_BS_BUCKET_MAX`)      | `max_num_seqs`                               |
| Decode | block size min (`VLLM_DECODE_BLOCK_BUCKET_MIN`)   | `1`                                 |
| Decode | block size step (`VLLM_DECODE_BLOCK_BUCKET_STEP`) | `block_size`                                 |
| Decode | block size max (`VLLM_DECODE_BLOCK_BUCKET_MAX`)   | `max_model_len * max_num_seqs // block_size` <br> by default or `max_blocks` <br> if `VLLM_CONTIGUOUS_PA = True` |

When a deployed workload does not use the full context a model can handle, we
recommend you to limit the maximum values upfront, based on the expected input
and output token lengths that will be generated after serving the vLLM server.
For example, suppose you want to deploy the text generation model Qwen2.5-1.5B
with `max_position_embeddings` of 131072 (our `max_model_len`) and your workload
pattern will not use the full context length (you expect the maximum input token
size of 1K and predict generating the maximum of 2K tokens as output). In this
case, starting the vLLM server to be ready for the full context length is
unnecessary and you can limit the values upfront. It reduces the startup time
and warm-up. Recommended settings for this case are:

- `--max_model_len`: `3072`, which is the sum of input and output sequences (1+2)*1024.  
- `VLLM_PROMPT_SEQ_BUCKET_MAX`: `1024`, which is the maximum input token size that you expect to handle.

!!! note
    If the model config specifies a high `max_model_len`, set it to the sum of `input_tokens` and `output_tokens`, rounded up to a multiple of `block_size` according to actual requirements.

## Split KV Cache Pools

When serving mixed workloads with both short-context (&lt;8K tokens) and long-context (~128K tokens) requests on a single vLLM-Gaudi instance, the `VLLM_SPLIT_POOLS_CONFIG` environment variable enables pool-aware batch splitting and bucket generation.

This avoids the OOM scenario where configuring `max_num_seqs=32` with `max_model_len=128K` exhausts HPU memory, by ensuring:

- **Short-context requests** batch at high batch sizes (24-32) with small sequence lengths
- **Long-context requests** batch at low batch sizes (1-8) with large sequence lengths
- **No mixing** of short and long requests in the same prefill sub-batch

### Configuration

Set `VLLM_SPLIT_POOLS_CONFIG` to a JSON string with `short` and `long` pool definitions:

```bash
export VLLM_SPLIT_POOLS_CONFIG='{
    "short": {"max_model_len": 8192, "max_num_seqs": 32, "memory_fraction": 0.40},
    "long":  {"max_model_len": 131072, "max_num_seqs": 8, "memory_fraction": 0.55},
    "threshold": 8192
}'
```

| Field | Description |
|-------|-------------|
| `short.max_model_len` | Maximum sequence length for the short pool |
| `short.max_num_seqs` | Maximum batch size for short-context prefill sub-batches |
| `short.memory_fraction` | Fraction of KV cache memory budgeted for short requests |
| `long.max_model_len` | Maximum sequence length for the long pool |
| `long.max_num_seqs` | Maximum batch size for long-context prefill sub-batches |
| `long.memory_fraction` | Fraction of KV cache memory budgeted for long requests |
| `threshold` | Token count boundary (defaults to `short.max_model_len`). Requests with more prompt tokens are routed to the long pool |

### How it works

1. **Auto-configuration**: Sets `VLLM_LONG_CONTEXT_SPLIT_THRESHOLD` and `VLLM_LONG_CONTEXT_MAX_BATCH_SIZE` from pool parameters if not already set
2. **Pool-aware bucketing**: During warmup, generates HPU graph buckets for both pool configurations (high BS / short seq + low BS / long seq)
3. **Batch isolation**: The model runner splits prefill batches so short and long requests never co-exist in the same sub-batch
4. **Memory budgeting**: Logs recommended block allocations per pool based on available HPU memory
