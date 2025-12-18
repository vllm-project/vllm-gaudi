# Managing and Reducing Warm-up Time

This document provides guidance on reducing warm-up time during vLLM model deployment on Intel® Gaudi® accelerators. It outlines the use of HPU graph caching, bucketing strategies,
and experimental features to improve the model performance.

## Reducing Warm-up Time with HPU Graph Caching

Intel Gaudi software supports caching of compiled HPU graphs using the `PT_HPU_RECIPE_CACHE_CONFIG` environment variable. This can significantly reduce startup time by reusing previously compiled graphs.

Setting the variable requires using the following format:

```python
export PT_HPU_RECIPE_CACHE_CONFIG=<RECIPE_CACHE_PATH>,<RECIPE_CACHE_DELETE>,<RECIPE_CACHE_SIZE_MB>
```

Where:

- `RECIPE_CACHE_PATH`: The directory for storing the compiled graph recipes.
- `RECIPE_CACHE_DELETE`: A boolean that controls cache behavior: when set to `true`, existing contents are cleared before storing new graph-compiled recipes; when set to `false`, the graph-compiled recipes stored in `RECIPE_CACHE_PATH` are reused, which speeds up the warm-up.
- `RECIPE_CACHE_SIZE_MB`: Sets the maximum size of the cache directory in MB. If the cache size limit is reached, the PyTorch bridge automatically deletes the oldest recipes, based on file creation time. We recommend adjusting the cache directory size according to the model and use case requirements.

The graph compilation process consists of two stages: GC graph compilation and HPU graph compilation. When `PT_HPU_RECIPE_CACHE_CONFIG` is enabled, the GC stage is skipped by reusing cached graphs, significantly reducing overall compilation time. The HPU graph compilation step, however, is still executed. The graph has to be regenerated in the following cases:

- PyTorch container or Intel® Gaudi® software version changes.
- Platform changes, for example Intel® Gaudi® 2 to Intel® Gaudi® 3.
- Model tensor parallelism or data type changes, for example, BF16 to FP8 or FP8 to BF16.

### Storage Recommendations

For scale-up scenarios where caching is shared across processes, we recommend using the local disk. Remote filesystems, such as NFS, should be avoided because they do not support file locking.

In Kubernetes environments, the cache can be stored on a PVC or NFS, but it should be copied to local disk before use.

For a usage example, refer to [Intel Gaudi Tutorials](https://github.com/HabanaAI/Gaudi-tutorials/blob/special/k8s/vllm-8b-cache.yaml).

### Deployment with vLLM

To cache the compiled HPU graphs and reduce the startup time, use one of the following methods.

#### Serving Command

Add the cache parameter to the serving command as shown in the following example for Llama 3.1 8B:

```python
# Store in cache
export PT_HPU_RECIPE_CACHE_CONFIG='/tmp/llama3_8b_recipe_cache/',True,8192
# Replay from cache
export PT_HPU_RECIPE_CACHE_CONFIG='/tmp/llama3_8b_recipe_cache/',False,8192
VLLM_PROMPT_BS_BUCKET_MAX=256 \
VLLM_DECODE_BS_BUCKET_MIN=128 \
VLLM_DECODE_BS_BUCKET_STEP=128 \
VLLM_DECODE_BS_BUCKET_MAX=128 \
VLLM_PROMPT_SEQ_BUCKET_MAX=1024 \
VLLM_DECODE_BLOCK_BUCKET_MAX=1024 \
PT_HPU_WEIGHT_SHARING=0 PT_HPU_MAX_COMPOUND_OP_SIZE=30 PT_HPU_LAZY_MODE=1 PT_HPU_ENABLE_LAZY_COLLECTIVES=true vllm serve meta-llama/Llama-3.1-8B-instruct -tp 1 --weights-load-device cpu --max-model-len 8192
```

This results in the following:

| Precision | Without cache | With cache | Time reduction |
| --------- | ------------- | ---------- | -------------- |
| BF16      | 66 sec        | 23 sec     | ~65% faster    |
| FP8       | 504 sec       | 34 sec     | ~93% faster    |

#### Docker

No changes are required in the Dockerfile as recipe cache is specific to the model and use case. Use the `-e` flag to set the environment variable:

```
-e PT_HPU_RECIPE_CACHE_CONFIG='/tmp/llama3_8b_recipe_cache/',True,8192
```

## Bucket Management

vLLM warm-up time is determined by the number of HPU graphs that must be compiled to support dynamic shapes. These shapes are influenced by the `batch_size`, `query_length`, and `num_context_blocks`. Setting them according to `max_num_batched_tokens` ensures that additional graphs are not compiled at runtime.

## Exponential Bucketing

The `VLLM_EXPONENTIAL_BUCKETING=True` flag, enabled by default starting with the vLLM `1.21.0-post1` release, switches the bucketing strategy from linear to exponential. This can reduce the number of buckets and warm-up time by up to 80%, while generally maintaining comparable inference performance. In some configurations, however, it may lead to a slight performance drop due to increased padding. This setting is particularly effective for BF16 and FP8 models. To use linear bucketing instead, set `VLLM_EXPONENTIAL_BUCKETING=False`.
