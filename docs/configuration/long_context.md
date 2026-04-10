
# Long Context Configuration

Long context feature enables support for a token context window exceeding 128K tokens. It is supported by the following models:

- [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
- [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small)

## Environment Variables Settings

Set the following environment variables to avoid OOM/functional issues.  Additional environment variable settings depend on context length:

- `VLLM_ENGINE_ITERATION_TIMEOUT_S=3600`
- `VLLM_RPC_TIMEOUT=100000`
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`

## Warm-up Buckets Preparation

With `VLLM_BUCKETING_STRATEGY=exp`, long-context buckets are prepared automatically. The `lin` and `pad` strategies usually require manual bucket tuning for long-context workloads. The following table presents 32K context-length examples for manual warm-up tuning:

| Flag | Suggested value | Notes |
|------|-----------------|-------|
| `VLLM_GRAPH_RESERVED_MEM` | `0.02` or `0.1` | It depends on the model and context length settings. Set to `0.02` for Llama3.1-8B or `0.1` for Llama3.1-70B. |
| `VLLM_PROMPT_QUERY_BUCKET_MIN` | `24576` | The value depends on the warm-up results. |
| `VLLM_PROMPT_QUERY_BUCKET_STEP` | `2048` | The value depends on the warm-up results. We recommend increasing it to a higher value for faster warm-up. For Intel Gaudi 3, we suggest setting it to `16384`. |
| `VLLM_PROMPT_QUERY_BUCKET_MAX` | `32768` | The value for context length is 32K; use 16384 for 16K. |
| `VLLM_DECODE_BLOCK_BUCKET_MIN` | `1024` | The value depends on the warm-up results. |
| `VLLM_DECODE_BLOCK_BUCKET_STEP` | `1024` | The value depends on the warm-up results. |
| `VLLM_DECODE_BLOCK_BUCKET_MAX` | `33792` | Calculate the value of `max_num_seqs * max_decode_seq // self.block_size`, where `max_decode_seq` represents the sum of input and output sequences. For example: `128 *((32 + 1)* 1024) / 128` and `32 *((32 + 1)* 1024) / 128`. |

Legacy `VLLM_PROMPT_SEQ_BUCKET_*` aliases are still accepted for prompt query tuning, but `VLLM_PROMPT_QUERY_BUCKET_*` is the preferred naming.

## Batch Size Settings

The default `batch_size=256` setting is not optimal for long contexts (8K+). Recompilations may occur if there is not enough KV cache space for some sequence groups. If recompilation or next recomputation warnings appear during inference, reduce `batch_size` to improve stability.

An example of a recompilation message:

```bash
Configuration: (prompt, 1, 36864) was not warmed-up!
```

An example of a warning message:

```bash
Sequence group cmpl-3cbf19b0c6d74b3f90b5d5db2ed2385e-0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory.
```

## Granite 4.0 Hybrid Models (128K Context)

Granite 4.0 models (`GraniteMoeHybridForCausalLM`) use a hybrid architecture combining
attention layers with Mamba2 state-space layers and MoE (Mixture of Experts).  This
architecture provides a significant advantage for long context: only the attention layers
consume KV-cache memory that grows with sequence length, while Mamba layers use a
fixed-size recurrent state that is shared across layers.

### Memory Layout

For `granite-4.0-h-small` (5 attention layers, 35 Mamba layers, TP=1, BF16):

| Component | Per block (128 tokens) | Per sequence (128K) |
|-----------|----------------------|---------------------|
| Attention KV cache (5 layers) | 2,560 KiB | 2.50 GiB |
| Mamba state (1 shared tensor) | 2,084 KiB | 2.04 GiB |
| **Total** | **4,644 KiB** | **~4.54 GiB** |

### Maximum Batch Size at 128K Context

The following table shows the estimated maximum batch size (concurrent 128K sequences)
for `granite-4.0-h-small` on a single HPU card with `VLLM_GRAPH_RESERVED_MEM=0.1`:

| `gpu_memory_utilization` | Gaudi 3 (128 GiB) | Gaudi 2 (96 GiB) |
|:------------------------:|:------------------:|:-----------------:|
| 0.50 | 12 | 8 |
| 0.70 | 16 | 12 |
| 0.80 | 19 | 14 |
| 0.90 | 21 | 16 |
| 0.95 | 23 | 16 |

!!! note
    These are theoretical upper bounds.  Actual batch sizes may be lower due to
    warm-up graph memory overhead, bucketing alignment, and runtime allocations.
    Use the analysis tool `tools/granite4_long_context_analysis.py` to explore
    different configurations.

### Recommended Settings

```bash
# Required for long context
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
VLLM_RPC_TIMEOUT=100000

# Granite 4.0 specific
VLLM_CONTIGUOUS_PA=false
PT_HPU_LAZY_MODE=0
```

Example server launch for 128K context on Gaudi 3:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_CONTIGUOUS_PA=false \
PT_HPU_LAZY_MODE=0 \
python -m vllm.entrypoints.openai.api_server \
    --model ibm-granite/granite-4.0-h-small \
    --block-size 128 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 21 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching
```

### Analysis Tool

Use `tools/granite4_long_context_analysis.py` to compute batch-size limits for
different Granite 4.0 variants, context lengths, and hardware configurations:

```bash
# Default analysis (granite-4.0-h-small, Gaudi 3, 128K)
python tools/granite4_long_context_analysis.py

# Use a specific preset
python tools/granite4_long_context_analysis.py \
    --preset granite-4.0-h-small --device GAUDI3

# Custom model parameters
python tools/granite4_long_context_analysis.py \
    --num-attention-layers 10 --num-mamba-layers 30 \
    --num-kv-heads 8 --head-dim 128 \
    --max-model-len 131072 --device GAUDI3
```
