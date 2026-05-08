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
| `VLLM_BUCKETING_STRATEGY`    | Selects the bucketing strategy: `exp`, `lin`, or `pad`.      | `exp`         |
| `VLLM_EXPONENTIAL_BUCKETING` | Deprecated compatibility flag. If set, it overrides `VLLM_BUCKETING_STRATEGY`: `true` forces `exp`, `false` forces `lin`. It cannot select `pad` and will be removed in a future release. | `None`        |
| `VLLM_BUCKETING_FROM_FILE`   | Enables reading bucket configuration from file.              | `None`        |
| `VLLM_ROW_PARALLEL_CHUNKS`   | Number of chunks to split input into for pipelining matmul with all-reduce in RowParallelLinear layers. Setting to a value greater than 1 enables chunking. See [Row-Parallel Chunking](../features/row_parallel_chunking.md). | `1` (disabled) |
| `VLLM_ROW_PARALLEL_CHUNK_THRESHOLD` | Minimum number of tokens required to activate row-parallel chunking. Inputs below this threshold use the standard non-chunked path. | `8192` |
| `VLLM_PROMPT_BS_BUCKET_MAX`  | Sets prefill batch size | `1` |

Use `VLLM_BUCKETING_STRATEGY=exp` for the default exponential warm-up, `VLLM_BUCKETING_STRATEGY=lin` for explicitly configured linear ranges, or `VLLM_BUCKETING_STRATEGY=pad` for padding-aware ranges with absolute and relative padding limits.

Leave `VLLM_EXPONENTIAL_BUCKETING` unset when using `VLLM_BUCKETING_STRATEGY`. The legacy flag is checked for backward compatibility and still overrides the selected strategy when present.

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
| `ENABLE_EXPERIMENTAL_FLAGS` and `ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES` | Must both be set to `true` for Qwen3.5 (GDN hybrid) models to improve graph compilation performance. | `false`                                          |

## Additional Performance Tuning Parameters for Bucketing Strategies

`VLLM_{phase}_{dim}_BUCKET_{param}` is a collection of environment variables configuring user-defined bucket ranges, where:

- `{phase}` is in `['PROMPT', 'DECODE']`.
- `{dim}` is in `['BS', 'QUERY', 'CTX']` for `PROMPT` phase or in `['BS', 'BLOCK']` for `DECODE` phase.
- `{param}` is in `['MIN', 'STEP', 'MAX']` for the `lin` strategy.
- `{param}` is in `['MIN', 'STEP', 'MAX', 'PAD_MAX', 'PAD_PERCENT']` for the `pad` strategy.

The following table lists the available variables with their default values. `PAD_MAX` and `PAD_PERCENT` are used only when `VLLM_BUCKETING_STRATEGY=pad`.

| Phase  | Variable name                                                            | Default value                                                                                                       |
|--------|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| Prompt | batch size min (`VLLM_PROMPT_BS_BUCKET_MIN`)                             | `1`                                                                                                                 |
| Prompt | batch size step (`VLLM_PROMPT_BS_BUCKET_STEP`)                           | `1`                                                                                                                 |
| Prompt | batch size max (`VLLM_PROMPT_BS_BUCKET_MAX`)                             | `max_num_prefill_seqs`                                                                                              |
| Prompt | batch size max abs padding (`VLLM_PROMPT_BS_BUCKET_PAD_MAX`)             | `ceil(max_num_prefill_seqs / 4)`                                                                                    |
| Prompt | batch size max padding percent (`VLLM_PROMPT_BS_BUCKET_PAD_PERCENT`)     | `25`                                                                                                                |
| Prompt | query length min (`VLLM_PROMPT_QUERY_BUCKET_MIN`)                        | `block_size`                                                                                                        |
| Prompt | query length step (`VLLM_PROMPT_QUERY_BUCKET_STEP`)                      | `block_size`                                                                                                        |
| Prompt | query length max (`VLLM_PROMPT_QUERY_BUCKET_MAX`)                        | `max_num_batched_tokens`                                                                                            |
| Prompt | query length max abs padding (`VLLM_PROMPT_QUERY_BUCKET_PAD_MAX`)        | `ceil(max_num_batched_tokens / 4)`                                                                                  |
| Prompt | query length max padding percent (`VLLM_PROMPT_QUERY_BUCKET_PAD_PERCENT`)| `25`                                                                                                                |
| Prompt | sequence ctx min (`VLLM_PROMPT_CTX_BUCKET_MIN`)                          | `0`                                                                                                                 |
| Prompt | sequence ctx step (`VLLM_PROMPT_CTX_BUCKET_STEP`)                        | `2`                                                                                                                 |
| Prompt | sequence ctx max (`VLLM_PROMPT_CTX_BUCKET_MAX`)                          | `ceil((max_model_len - VLLM_PROMPT_QUERY_BUCKET_MIN) / block_size)`                                                 |
| Prompt | sequence ctx max abs padding (`VLLM_PROMPT_CTX_BUCKET_PAD_MAX`)          | `ceil(max_num_batched_tokens / block_size)`                                                                         |
| Prompt | sequence ctx max padding percent (`VLLM_PROMPT_CTX_BUCKET_PAD_PERCENT`)  | `25`                                                                                                                |
| Decode | batch size min (`VLLM_DECODE_BS_BUCKET_MIN`)                             | `1`                                                                                                                 |
| Decode | batch size step (`VLLM_DECODE_BS_BUCKET_STEP`)                           | `2`                                                                                                                 |
| Decode | batch size max (`VLLM_DECODE_BS_BUCKET_MAX`)                             | `max_num_seqs`                                                                                                      |
| Decode | batch size max abs padding (`VLLM_DECODE_BS_BUCKET_PAD_MAX`)             | `ceil(max_num_seqs / 4)`                                                                                            |
| Decode | batch size max padding percent (`VLLM_DECODE_BS_BUCKET_PAD_PERCENT`)     | `25`                                                                                                                |
| Decode | num blocks min (`VLLM_DECODE_BLOCK_BUCKET_MIN`)                          | `block_size`                                                                                                        |
| Decode | num blocks step (`VLLM_DECODE_BLOCK_BUCKET_STEP`)                        | `block_size`                                                                                                        |
| Decode | num blocks max (`VLLM_DECODE_BLOCK_BUCKET_MAX`)                          | `ceil(max_model_len * max_num_seqs / block_size)` <br>by default or `max_blocks` <br>if `VLLM_CONTIGUOUS_PA = True` |
| Decode | num blocks max abs padding (`VLLM_DECODE_BLOCK_BUCKET_PAD_MAX`)          | `ceil(VLLM_DECODE_BLOCK_BUCKET_MAX / 4)`                                                                            |
| Decode | num blocks max padding percent (`VLLM_DECODE_BLOCK_BUCKET_PAD_PERCENT`)  | `25`                                                                                                                |

`VLLM_PROMPT_BS_BUCKET_MAX` no longer affects only prompt warm-up coverage. It also affects the real prefill batch size used by the Gaudi runner.

The default value of `25` for `VLLM_*_BUCKET_PAD_PERCENT` is a balance of warmup duration and runtime performance. Using smaller value like `10` introduce more buckets and reduces the padding to get better runtime performance. Setting to `0` to fall back to the original linear bucketing with minimum padding. And setting to `50` is close to the exponential bucketing except for the corresponding  `VLLM_*_BUCKET_MIN` is not `0` nor `1`.

Legacy `VLLM_PROMPT_SEQ_BUCKET_*` variables are still accepted as a fallback for prompt query settings when `VLLM_PROMPT_QUERY_BUCKET_*` is not set, but this compatibility path is deprecated and will be removed in a future release.

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
- `VLLM_PROMPT_QUERY_BUCKET_MAX`: `1024`, which is the maximum input token size that you expect to handle.

!!! note
    If the model config specifies a high `max_model_len`, set it to the sum of `input_tokens` and `output_tokens`, rounded up to a multiple of `block_size` according to actual requirements.
