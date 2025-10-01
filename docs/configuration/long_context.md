
# Long Context Configuration

Long context feature enables support for a token context window exceeding 128K tokens. It is supported by the following models:
- [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)

## Environment Variables Settings

Set the following environment variables to avoid OOM/functional issues.  Additional environment variable settings depend on context length:

- `VLLM_ENGINE_ITERATION_TIMEOUT_S=3600`
- `VLLM_RPC_TIMEOUT=100000`
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`

## Warmup buckets preparation
Exponential bucketing mechanism automatically prepares buckets for long context. Linear bucketing mechanism requires manual flags settings.

**32K context length flags examples for linear warmup:**

- `VLLM_GRAPH_RESERVED_MEM`: The value depends on the model and context length settings. Use `VLLM_GRAPH_RESERVED_MEM=0.02` for Llama3.1-8B or `VLLM_GRAPH_RESERVED_MEM=0.1` for Llama3.1-70B.
- `VLLM_PROMPT_SEQ_BUCKET_MIN=24576`: Suggested value, depends on warmup results.
- `VLLM_PROMPT_SEQ_BUCKET_STEP=2048`: Suggested value, depends on warmup results. It is recommended to increase it to a higher value for faster warmup. `VLLM_PROMPT_SEQ_BUCKET_STEP=16384` - Suggested value for Intel Gaudi 3.
- `VLLM_PROMPT_SEQ_BUCKET_MAX=32768`: Value for context length of 32K. Use 16384 for 16K.
- `VLLM_DECODE_BLOCK_BUCKET_MIN=1024`: Suggested value, depends on warmup results.
- `VLLM_DECODE_BLOCK_BUCKET_STEP=1024`: Suggested value, depends on warmup results.
- `VLLM_DECODE_BLOCK_BUCKET_MAX=33792`: `max_num_seqs * max_decode_seq // self.block_size`, where `max_decode_seq` represents the sum of input and output sequences. For example:
  - `128 *((32 + 1)* 1024) / 128`
  - `32 *((32 + 1)* 1024) / 128`

## Batch Size Settings

The default `batch_size=256` is not optimal for long contexts (8K+). Recompilations may occur if there is not enough KV cache space for some sequence groups.

If recompilation or next recomputation warnings appear during inference, reduce `batch_size` to improve stability.

**Recompilation message example:**

```bash
Configuration: (prompt, 1, 36864) was not warmed-up!
```

**Warning message example:**

```bash
Sequence group cmpl-3cbf19b0c6d74b3f90b5d5db2ed2385e-0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory.
```

## Multi-Step Scheduling Feature Usage

Enabling Multi-Step Scheduling is recommended for better decode performance. Refer to vllm-project#6854 for more details.
