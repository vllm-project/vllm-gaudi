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
