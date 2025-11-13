---
title: Introduction
---

# Quantization and Inference

Quantization trades off model precision for smaller memory footprint, allowing large models to be run on a wider range of devices. The Intel® Gaudi® Backend supports the following quantization backends:

- [Intel® Neural Compressor](inc.md)
- [Auto_Awq](auto_awq.md)
- [Gptqmodel](gptqmodel.md)

Use the links above to see backend-specific instructions. This document provides information on how to interpret logs, troubleshoot out-of-memory (OOM) errors, and manage performance.

## Logs Interpretation

To better understand the logs, let’s look at a simple example of running the [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) model on a Intel® Gaudi® vLLM server server with default settings:

```bash
python -m vllm.entrypoints.openai.api_server --model="meta-llama/Meta-Llama-3.1-8B" --dtype=bfloat16 --block-size=128 --max-num-seqs=4 --tensor-parallel-size=1 --max-seq_len-to-capture=2048
```

The following example shows the initial part of the generated server log:

```text hl_lines="3 4 5 7"
INFO 09-24 17:31:39 habana_model_runner.py:590] Pre-loading model weights on hpu:0 took 15.05 GiB of device memory (15.05 GiB/94.62 GiB used) and 1.067 GiB of host memory (8.199 GiB/108.2 GiB used)
INFO 09-24 17:31:39 habana_model_runner.py:636] Wrapping in HPU Graph took 0 B of device memory (15.05 GiB/94.62 GiB used) and -3.469 MiB of host memory (8.187 GiB/108.2 GiB used)
INFO 09-24 17:31:39 habana_model_runner.py:640] Loading model weights took in total 15.05 GiB of device memory (15.05 GiB/94.62 GiB used) and 1.056 GiB of host memory (8.188 GiB/108.2 GiB used)
INFO 09-24 17:31:40 habana_worker.py:153] Model profiling run took 355 MiB of device memory (15.4 GiB/94.62 GiB used) and 131.4 MiB of host memory (8.316 GiB/108.2 GiB used)
INFO 09-24 17:31:40 habana_worker.py:177] Free device memory: 79.22 GiB, 71.3 GiB usable (gpu_memory_utilization=0.9), 7.13 GiB reserved for HPUGraphs (VLLM_GRAPH_RESERVED_MEM=0.1), 64.17 GiB reserved for KV cache
INFO 09-24 17:31:40 habana_executor.py:85] # HPU blocks: 4107, # CPU blocks: 256
INFO 09-24 17:31:41 habana_worker.py:208] Initializing cache engine took 64.17 GiB of device memory (79.57 GiB/94.62 GiB used) and 1.015 GiB of host memory (9.329 GiB/108.2 GiB used)
```

The log displays memory consumption trends for the selected model. It reports device memory usage during model weight loading, profiling runs (using dummy data and without the KV cache), and the final usable device memory available before the warm-up phase begins. This usable memory is shared between HPU graphs and the KV cache. You can use this information to determine an appropriate bucketing scheme for warm-ups.

The following example shows the warm-up phase logs:

```text
INFO 09-24 17:32:13 habana_model_runner.py:1477] Graph/Prompt captured:24 (100.0%) used_mem:67.72 MiB buckets:[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 09-24 17:32:13 habana_model_runner.py:1477] Graph/Decode captured:1 (100.0%) used_mem:64 KiB buckets:[(4, 128)]
INFO 09-24 17:32:13 habana_model_runner.py:1620] Warmup finished in 32 secs, allocated 92.77 MiB of device memory
INFO 09-24 17:32:13 habana_executor.py:91] init_cache_engine took 64.26 GiB of device memory (79.66 GiB/94.62 GiB used) and 1.104 GiB of host memory (9.419 GiB/108.2 GiB used)
```

After analyzing these logs, you should have a good understanding of how much free device memory remains for overhead calculations and how much more could still be used by increasing `gpu_memory_utilization`. You can balance the memory allocation for warm-up bucketing, HPUGraphs, and the KV cache to suit your workload requirements.

## Troubleshooting OOM Errors

Factors such as available GPU memory, model size, and input sequence length may prevent the standard inference command from running successfully for your model, potentially resulting in out-of-memory (OOM) errors. To address these errors, consider the following recommendations:

- Increase `gpu_memory_utilization`: To address memory limitations, vLLM pre-allocates HPU cache using the percentage of memory defined by `gpu_memory_utilization`. Increasing this value allocates more space for the KV cache.
- Decrease `max_num_seqs` or `max_num_batched_tokens`: It may reduce the number of concurrent requests in a batch, leading to lower KV cache usage.
- Increase `tensor_parallel_size`: This method distributes the model weights across GPUs, increasing the memory available for the KV cache on each GPU.

## Performance Tuning Guidelines

Familiarize with the following notes and recommendations for guidance on performance management:

- During the development phase, when evaluating a model for inference on vLLM, you may skip the warm-up phase of the server using the `VLLM_SKIP_WARMUP=true` environment variable. This helps to achieve faster testing turnaround times. However, disabling warm-up is acceptable only for development purposes, we strongly recommend keeping it enabled in production environments. Keep [warm-up](../warm-up/warm-up.md) enabled during deployment with optimal number of [buckets](../../features/bucketing_mechanism.md). Warm-up time depends on many factors, such as input and output sequence length, batch size, number of buckets, and data type. It can even take a couple of hours, depending on the configuration. For more information, see the [Warm-up](../../features/warmup.md) document.

- HPU graphs and the KV cache share the same usable memory pool, determined by `gpu_memory_utilization`. Memory allocation between the two must be balanced to prevent performance degradation. You can control this balance using the `VLLM_GRAPH_RESERVED_MEM` environment variable, which defines the ratio of memory reserved for HPU graphs versus the KV cache. Increasing the KV cache size enables larger batch processing, improving overall throughput. Conversely, enabling [HPU graphs](../warm-up/warm-up.md#hpu-graph-capture) helps reduce host [overhead](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html#reducing-host-overhead-with-hpu-graphs) and can lower latency.

- The `VLLM_GRAPH_PROMPT_RATIO` environment variable controls the ratio of usable graph memory between prefill and decode graphs. Assigning more memory to a stage usually results in faster execution for that stage.

- [Bucketing mechanisms](../../features/bucketing_mechanism.md) can help optimize performance across different workloads. The vLLM server is pre-configured for heavy decoding scenarios with high request concurrency, using the default maximum batch size strategy (`VLLM_GRAPH_DECODE_STRATEGY`). During low-load periods, this configuration may not be ideal and can be adjusted for smaller batch sizes. For example, modifying bucket ranges via `VLLM_DECODE_BS_BUCKET_{param}` can improve efficiency. For a list of environment variables controlling bucketing behavior, see the [Environment Variables](../env_variables.md) document.

- Using the Floating Point 8-bit (FP8) data type for large language models reduces memory bandwidth requirements by half compared to BF16. In addition, the FP8 computation is twice as fast as BF16, enabling performance gains even for compute-bound workloads, such as offline inference with large batch sizes.
For more information, see the [Floating Point 8-bit](../../features/floating_point_8.md) document.
