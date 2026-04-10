
# Long Context Configuration

Long context feature enables support for a token context window exceeding 128K tokens. It is supported by the following models:

- [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b)
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
- [ibm-granite/granite-4.0-8b-base](https://huggingface.co/ibm-granite/granite-4.0-8b-base) (hybrid MoE + Mamba architecture, see [Granite 4.0 128K section](#granite-40-128k-context-on-hpu))

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

## Batch Size and Block Count Analysis

For long-context workloads, the maximum batch size is determined by the number of KV cache blocks available, which in turn depends on `gpu_memory_utilization`. Use the following formulas to estimate limits before launching:

### Block Requirements

Each sequence occupying a `max_model_len`-token context requires:

```
blocks_per_seq = ceil(max_model_len / block_size)
```

For 128K context with the default block size of 128:

```
blocks_per_seq = ceil(131072 / 128) = 1024 blocks per sequence
```

### Available KV Cache Memory

The memory available for KV cache blocks after model weights and HPUGraph capture is:

```
kv_cache_memory = free_device_memory * gpu_memory_utilization
                  * (1 - VLLM_GRAPH_RESERVED_MEM)
```

`free_device_memory` is measured **after the model weights are loaded** (not total card HBM).
vLLM prints this value at startup:

```
Free device memory: <X> GiB, <Y> GiB usable (gpu_memory_utilization=<Z>)
```

`gpu_memory_utilization` is then applied to `free_device_memory` to obtain usable HBM, and
`VLLM_GRAPH_RESERVED_MEM` carves out a fraction of that for HPUGraph capture — both reductions
happen after the model has already consumed its share of HBM.

### KV Cache Per Sequence (Dense Attention Models)

For pure attention models (e.g., Llama):

```
kv_cache_per_seq = 2 * max_model_len * num_attn_layers
                   * num_kv_heads * head_dim * dtype_bytes
```

### Maximum Batch Size

```
max_batch_size = floor(kv_cache_memory / kv_cache_per_seq)
```

For 128K context this number is typically small (single digits per Gaudi card for 8B-class models). Use tensor parallelism to scale available KV cache memory across multiple cards.

### VLLM_DECODE_BLOCK_BUCKET_MAX

Set `VLLM_DECODE_BLOCK_BUCKET_MAX` to at least:

```
VLLM_DECODE_BLOCK_BUCKET_MAX = max_num_seqs * ceil(max_model_len / block_size)
                              = max_num_seqs * 1024   # for 128K, block_size=128
```

---

## Granite 4.0 128K Context on HPU

[Granite 4.0](https://huggingface.co/ibm-granite/granite-4.0-8b-base) is a hybrid MoE + Mamba architecture (`GraniteMoeHybridForCausalLM`). It interleaves standard attention layers with GDN (global-domain normalization / linear-attention) recurrent layers. This affects memory layout and batch size estimation compared to pure-attention models.

### Key Differences from Pure Attention Models

- **KV cache is required only for attention layers**, not for GDN/Mamba layers.
- GDN layers maintain a fixed-size recurrent state per request (independent of sequence length) when `VLLM_COMPACT_GDN=1` (default).
- When `VLLM_COMPACT_GDN=0`, GDN state scales with the number of KV cache blocks, consuming additional HBM that reduces the achievable batch size.
- Paged attention requires contiguous physical blocks to be disabled: set `VLLM_CONTIGUOUS_PA=false`.

### Recommended Environment Variables for Granite 4.0 at 128K

```bash
# Required for contexts > model default max_position_embeddings
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Required for hybrid Mamba/GDN models (non-contiguous paged attention)
VLLM_CONTIGUOUS_PA=false

# Use compact GDN allocation so recurrent state does not scale with num_blocks
# (auto-enabled for GDN models, but set explicitly for clarity)
VLLM_COMPACT_GDN=1

# Increase timeouts for 128K inference
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
VLLM_RPC_TIMEOUT=100000

# Reserve enough HBM for HPUGraphs; adjust based on observed graph memory
VLLM_GRAPH_RESERVED_MEM=0.1
```

### Batch Size Estimation for Granite 4.0 at 128K

For Granite 4.0, the effective KV cache per sequence uses only the attention layers:

```
kv_cache_per_seq = 2 * max_model_len * num_attn_layers
                   * num_kv_heads_per_device * head_dim * dtype_bytes
```

where `num_kv_heads_per_device = num_kv_heads / tensor_parallel_size`.

With `VLLM_COMPACT_GDN=1`, GDN states occupy a small fixed buffer and do not reduce available KV cache blocks.

#### Example: Granite 4.0 8B on Gaudi 3 (96 GiB HBM, TP=1, BF16)

Approximate values for `ibm-granite/granite-4.0-8b-base`:

| Parameter | Value |
|-----------|-------|
| `max_model_len` | 131072 (128K) |
| `block_size` | 128 |
| `blocks_per_seq` | 1024 |
| `num_attn_layers` | ~34 (verify with `model.config.num_hidden_layers` and the layer-type list in the model config) |
| `num_kv_heads` | 8 |
| `head_dim` | 128 |
| `dtype_bytes` | 2 (BF16) |
| `gpu_memory_utilization` | 0.9 |
| `VLLM_GRAPH_RESERVED_MEM` | 0.1 |

```
kv_cache_per_seq ≈ 2 × 131072 × 34 × 8 × 128 × 2 B ≈ 18.0 GiB

# Model weight estimate: ~8B params × 2 B/param (BF16) ≈ 16 GiB
# free_device_memory is reported after the model is loaded
free_device_memory ≈ 80 GiB  (96 GiB total HBM − ~16 GiB model weights)

kv_cache_memory ≈ 80 × 0.9 × (1 - 0.1) ≈ 64.8 GiB

max_batch_size ≈ floor(64.8 / 18.0) ≈ 3 sequences
```

To increase throughput, use tensor parallelism (`--tensor-parallel-size 8`), which shards model
weights, activations, and KV heads across all cards.  Each card then holds only `1/TP` of the
model parameters and handles `1/TP` of the KV heads, so both the model footprint and the
per-sequence KV cache shrink by the TP factor:

```
num_kv_heads_per_device = 8 / 8 = 1

kv_cache_per_seq ≈ 2 × 131072 × 34 × 1 × 128 × 2 B ≈ 2.25 GiB

# Model sharded across 8 cards: ~16 GiB / 8 ≈ 2 GiB per card
# Activation memory also scales down with TP, leaving more HBM free
free_device_memory ≈ 94 GiB  (96 GiB total − ~2 GiB model shard per card)

kv_cache_memory ≈ 94 × 0.9 × (1 - 0.1) ≈ 76.1 GiB

max_batch_size ≈ floor(76.1 / 2.25) ≈ 33 sequences
```

> **Note:** These are estimates. The actual number reported by vLLM after profiling may differ due to activation memory, operator workspace, and HPUGraph overhead.

### Warm-up Bucket Settings for Granite 4.0 at 128K

| Flag | Suggested value | Notes |
|------|-----------------|-------|
| `VLLM_PROMPT_QUERY_BUCKET_MIN` | `128` | Minimum prompt query bucket |
| `VLLM_PROMPT_QUERY_BUCKET_STEP` | `16384` | Large step for faster warm-up on Gaudi 3 |
| `VLLM_PROMPT_QUERY_BUCKET_MAX` | `131072` | Full 128K context length |
| `VLLM_DECODE_BLOCK_BUCKET_MIN` | `1024` | One full 128K sequence |
| `VLLM_DECODE_BLOCK_BUCKET_STEP` | `1024` | One sequence worth of blocks |
| `VLLM_DECODE_BLOCK_BUCKET_MAX` | `max_num_seqs × 1024` | Scale with estimated batch size |

### Example Launch Command

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_CONTIGUOUS_PA=false \
VLLM_COMPACT_GDN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_GRAPH_RESERVED_MEM=0.1 \
VLLM_DECODE_BLOCK_BUCKET_MAX=4096 \
python -m vllm.entrypoints.openai.api_server \
    --model ibm-granite/granite-4.0-8b-base \
    --max-model-len 131072 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 1
```

Adjust `--max-num-seqs` and `VLLM_DECODE_BLOCK_BUCKET_MAX` to match your actual batch size estimate.  If memory-pressure warnings or preemption messages appear, reduce `--max-num-seqs` or increase `--gpu-memory-utilization` / `--tensor-parallel-size`.
