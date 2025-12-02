# Performance Tuning

Understanding how configuration settings affect system behavior is essential for effective performance management. This document explains how you can tune and optimize performance.

## Memory Allocation

HPU graphs and the KV cache share the same usable memory pool, determined by `gpu_memory_utilization`. Memory allocation between the two must be balanced to prevent performance degradation. You can find memory consumption information for your model in the logs. They provide device memory usage during model weight loading, profiling runs (using dummy data and without the KV cache), and the final usable memory available before the warm-up phase begins. You can use this information to determine an appropriate bucketing scheme for warm-ups. The following example shows the initial part of the generated server log for the [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) model:

```text hl_lines="3 4 5 7"
INFO 09-24 17:31:39 habana_model_runner.py:590] Pre-loading model weights on hpu:0 took 15.05 GiB of device memory (15.05 GiB/94.62 GiB used) and 1.067 GiB of host memory (8.199 GiB/108.2 GiB used)
INFO 09-24 17:31:39 habana_model_runner.py:636] Wrapping in HPU Graph took 0 B of device memory (15.05 GiB/94.62 GiB used) and -3.469 MiB of host memory (8.187 GiB/108.2 GiB used)
INFO 09-24 17:31:39 habana_model_runner.py:640] Loading model weights took in total 15.05 GiB of device memory (15.05 GiB/94.62 GiB used) and 1.056 GiB of host memory (8.188 GiB/108.2 GiB used)
INFO 09-24 17:31:40 habana_worker.py:153] Model profiling run took 355 MiB of device memory (15.4 GiB/94.62 GiB used) and 131.4 MiB of host memory (8.316 GiB/108.2 GiB used)
INFO 09-24 17:31:40 habana_worker.py:177] Free device memory: 79.22 GiB, 71.3 GiB usable (gpu_memory_utilization=0.9), 7.13 GiB reserved for HPUGraphs (VLLM_GRAPH_RESERVED_MEM=0.1), 64.17 GiB reserved for KV cache
INFO 09-24 17:31:40 habana_executor.py:85] # HPU blocks: 4107, # CPU blocks: 256
INFO 09-24 17:31:41 habana_worker.py:208] Initializing cache engine took 64.17 GiB of device memory (79.57 GiB/94.62 GiB used) and 1.015 GiB of host memory (9.329 GiB/108.2 GiB used)
```

You can control the ratio between HPU graphs and KV cache using the `VLLM_GRAPH_RESERVED_MEM` environment variable. Increasing the KV cache size enables larger batch processing, improving overall throughput. Enabling [HPU graphs](warm-up/warm-up.md#hpu-graph-capture) helps reduce host [overhead](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html#reducing-host-overhead-with-hpu-graphs) and can lower latency.

The following example shows the warm-up phase logs:

```text
INFO 09-24 17:32:13 habana_model_runner.py:1477] Graph/Prompt captured:24 (100.0%) used_mem:67.72 MiB buckets:[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 09-24 17:32:13 habana_model_runner.py:1477] Graph/Decode captured:1 (100.0%) used_mem:64 KiB buckets:[(4, 128)]
INFO 09-24 17:32:13 habana_model_runner.py:1620] Warmup finished in 32 secs, allocated 92.77 MiB of device memory
INFO 09-24 17:32:13 habana_executor.py:91] init_cache_engine took 64.26 GiB of device memory (79.66 GiB/94.62 GiB used) and 1.104 GiB of host memory (9.419 GiB/108.2 GiB used)
```

After analyzing these logs, you should have a good understanding of how much free device memory remains for overhead calculations and how much more could still be used by increasing `gpu_memory_utilization`. You can balance the memory allocation for warm-up bucketing, HPU graphs, and the KV cache to suit your workload requirements.

## Bucketing Mechanism

The [bucketing mechanism](../features/bucketing_mechanism.md) can help optimize performance across different workloads. The vLLM server is pre-configured for heavy decoding scenarios with high request concurrency, using the default maximum batch size strategy (`VLLM_GRAPH_DECODE_STRATEGY`). During low-load periods, this configuration may not be ideal and can be adjusted for smaller batch sizes. For example, modifying bucket ranges via `VLLM_DECODE_BS_BUCKET_{param}` can improve efficiency. For a list of environment variables controlling bucketing behavior, see the [Environment Variables](env_variables.md) document.

## Floating Point 8-bit

Using the Floating Point 8-bit (FP8) data type for large language models reduces memory bandwidth requirements by half compared to BF16. In addition, the FP8 computation is twice as fast as BF16, enabling performance gains even for compute-bound workloads, such as offline inference with large batch sizes.
For more information, see the [Floating Point 8-bit](../features/floating_point_8.md) document.

## Warm-Up

During the development phase, when evaluating a model for inference on vLLM, you may skip the warm-up phase of the server using the `VLLM_SKIP_WARMUP=true` environment variable. This helps to achieve faster testing turnaround times. However, disabling warm-up is acceptable only for development purposes, we strongly recommend keeping it enabled in production environments. Keep warm-up enabled during deployment with optimal number of [buckets](../features/bucketing_mechanism.md).

Warm-up time depends on many factors, such as input and output sequence length, batch size, number of buckets, and data type. It can even take a couple of hours, depending on the configuration. For more information, see the [Warm-up](../features/warmup.md) document.
