---
title: Warm-up
---

# Warm-up

Warm-up is a highly recommended step that occurs before the vLLM server starts listening. It performs a forward pass for each bucket using dummy data. The goal is to precompile all graphs and eliminate any graph compilation overhead within bucket boundaries during server runtime. Each warm-up step is logged during the vLLM startup.

The following example presents the same buckets as those described in the [Bucketing Mechanism](bucketing_mechanism.md) section. Each output line corresponds to the execution of a single bucket. When a bucket is executed for the first time, its graph is compiled and can be reused later, avoiding further graph compilations.

```{.}
INFO 08-01 22:26:47 hpu_model_runner.py:1066] [Warmup][Prompt][1/24] batch_size:4 seq_len:1024 free_mem:79.16 GiB
INFO 08-01 22:26:47 hpu_model_runner.py:1066] [Warmup][Prompt][2/24] batch_size:4 seq_len:896 free_mem:55.43 GiB
INFO 08-01 22:26:48 hpu_model_runner.py:1066] [Warmup][Prompt][3/24] batch_size:4 seq_len:768 free_mem:55.43 GiB
...
INFO 08-01 22:26:59 hpu_model_runner.py:1066] [Warmup][Prompt][24/24] batch_size:1 seq_len:128 free_mem:55.43 GiB
INFO 08-01 22:27:00 hpu_model_runner.py:1066] [Warmup][Decode][1/48] batch_size:4 seq_len:2048 free_mem:55.43 GiB
INFO 08-01 22:27:00 hpu_model_runner.py:1066] [Warmup][Decode][2/48] batch_size:4 seq_len:1920 free_mem:55.43 GiB
INFO 08-01 22:27:01 hpu_model_runner.py:1066] [Warmup][Decode][3/48] batch_size:4 seq_len:1792 free_mem:55.43 GiB
...
INFO 08-01 22:27:16 hpu_model_runner.py:1066] [Warmup][Decode][47/48] batch_size:2 seq_len:128 free_mem:55.43 GiB
INFO 08-01 22:27:16 hpu_model_runner.py:1066] [Warmup][Decode][48/48] batch_size:1 seq_len:128 free_mem:55.43 GiB
```

Compiling all buckets may take some time. To skip this step, you can set the environment variable `VLLM_SKIP_WARMUP=true`. Note that doing so may trigger graph compilations the first time a particular bucket is executed.

!!! warning
    Disabling warm-up is acceptable for development purposes, we strongly recommend keeping it enabled in production environments.

## HPU Graph Capture

[HPU Graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) are currently the most performant execution method for vLLM Hardware Plugin for Intel® Gaudi®. When HPU Graphs are enabled, execution graphs are traced (recorded) after warm-up and then replayed during inference, significantly reducing host overhead. Recording can consume significant memory, which should be considered when allocating the KV cache. Enabling HPU Graphs impacts the number of available KV cache blocks, but vLLM provides user-configurable variables to help manage memory usage.

When HPU Graphs are used, they share the common memory pool, called usable memory, with the KV cache, as controlled by the `gpu_memory_utilization` flag with the default value of `0.9`. Before allocating the KV cache, the model weights are loaded onto the device, and a forward pass is executed on dummy data to estimate memory usage. Only after this step the `gpu_memory_utilization` flag is applied. By default, it designates 90% of the free device memory at that point as usable. Next, the KV cache is allocated, the model is warmed up, and HPU Graphs are captured. The `VLLM_GRAPH_RESERVED_MEM` environment variable defines the portion of memory reserved for HPU Graph capture. With the default value  of `0.1`, 10% of the usable memory is reserved for graph capture (referred to as “usable graph memory”), while the remaining 90% is allocated to the KV cache.

The `gpu_memory_utilization` parameter does not represent the absolute memory usage across the HPU. Instead, it specifies the memory margin after loading the model and running a profiling pass. For example, if a device has 100 GiB of total memory and 50 GiB of free memory after loading the model weights and executing the profiling run, the default value of `gpu_memory_utilization` will mark 90% of the 50 GiB as usable, leaving 5 GiB as a margin - regardless of the total device memory.

When many requests are pending, the vLLM scheduler attempts to fill the maximum decode batch size as quickly as possible. Once a request completes, the decode batch size decreases. When this happens, vLLM schedules a prefill iteration for requests in the waiting queue to restore the previous decode batch size. In fully loaded scenarios, the decode batch size is often at its maximum, making large-batch HPU graphs critical to capture. On the other hand, prompt iterations typically execute with very low batch sizes (1-4).

Each step outlined is logged by the vLLM server, with negative values indicating memory release:

```{.}
INFO 08-02 17:37:44 hpu_model_runner.py:493] Prompt bucket config (min, step, max_warmup) bs:[1, 32, 4], seq:[128, 128, 1024]
INFO 08-02 17:37:44 hpu_model_runner.py:499] Generated 24 prompt buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024)]
INFO 08-02 17:37:44 hpu_model_runner.py:504] Decode bucket config (min, step, max_warmup) bs:[1, 128, 4], seq:[128, 128, 2048]
INFO 08-02 17:37:44 hpu_model_runner.py:509] Generated 48 decode buckets: [(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
INFO 08-02 17:37:52 hpu_model_runner.py:430] Pre-loading model weights on hpu:0 took 14.97 GiB of device memory (14.97 GiB/94.62 GiB used) and 2.95 GiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:52 hpu_model_runner.py:438] Wrapping in HPU Graph took 0 B of device memory (14.97 GiB/94.62 GiB used) and -252 KiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:52 hpu_model_runner.py:442] Loading model weights took in total 14.97 GiB of device memory (14.97 GiB/94.62 GiB used) and 2.95 GiB of host memory (475.2 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_worker.py:134] Model profiling run took 504 MiB of device memory (15.46 GiB/94.62 GiB used) and 180.9 MiB of host memory (475.4 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_worker.py:158] Free device memory: 79.16 GiB, 39.58 GiB usable (gpu_memory_utilization=0.5), 15.83 GiB reserved for HPUGraphs (VLLM_GRAPH_RESERVED_MEM=0.4), 23.75 GiB reserved for KV cache
INFO 08-02 17:37:54 hpu_executor.py:85] # HPU blocks: 1519, # CPU blocks: 0
INFO 08-02 17:37:54 hpu_worker.py:190] Initializing cache engine took 23.73 GiB of device memory (39.2 GiB/94.62 GiB used) and -1.238 MiB of host memory (475.4 GiB/1007 GiB used)
INFO 08-02 17:37:54 hpu_model_runner.py:1066] [Warmup][Prompt][1/24] batch_size:4 seq_len:1024 free_mem:55.43 GiB
...
INFO 08-02 17:38:22 hpu_model_runner.py:1066] [Warmup][Decode][48/48] batch_size:1 seq_len:128 free_mem:55.43 GiB
INFO 08-02 17:38:22 hpu_model_runner.py:1159] Using 15.85 GiB/55.43 GiB of free device memory for HPUGraphs, 4.755 GiB for prompt and 11.095 GiB for decode (VLLM_GRAPH_PROMPT_RATIO=0.3)
INFO 08-02 17:38:22 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][1/24] batch_size:1 seq_len:128 free_mem:55.43 GiB
...
INFO 08-02 17:38:26 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][11/24] batch_size:1 seq_len:896 free_mem:48.77 GiB
INFO 08-02 17:38:27 hpu_model_runner.py:1066] [Warmup][Graph/Decode][1/48] batch_size:4 seq_len:128 free_mem:47.51 GiB
...
INFO 08-02 17:38:41 hpu_model_runner.py:1066] [Warmup][Graph/Decode][48/48] batch_size:1 seq_len:2048 free_mem:47.35 GiB
INFO 08-02 17:38:41 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][12/24] batch_size:4 seq_len:256 free_mem:47.35 GiB
INFO 08-02 17:38:42 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][13/24] batch_size:2 seq_len:512 free_mem:45.91 GiB
INFO 08-02 17:38:42 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][14/24] batch_size:1 seq_len:1024 free_mem:44.48 GiB
INFO 08-02 17:38:43 hpu_model_runner.py:1066] [Warmup][Graph/Prompt][15/24] batch_size:2 seq_len:640 free_mem:43.03 GiB
INFO 08-02 17:38:43 hpu_model_runner.py:1128] Graph/Prompt captured:15 (62.5%) used_mem:14.03 GiB buckets:[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (4, 128), (4, 256)]
INFO 08-02 17:38:43 hpu_model_runner.py:1128] Graph/Decode captured:48 (100.0%) used_mem:161.9 MiB buckets:[(1, 128), (1, 256), (1, 384), (1, 512), (1, 640), (1, 768), (1, 896), (1, 1024), (1, 1152), (1, 1280), (1, 1408), (1, 1536), (1, 1664), (1, 1792), (1, 1920), (1, 2048), (2, 128), (2, 256), (2, 384), (2, 512), (2, 640), (2, 768), (2, 896), (2, 1024), (2, 1152), (2, 1280), (2, 1408), (2, 1536), (2, 1664), (2, 1792), (2, 1920), (2, 2048), (4, 128), (4, 256), (4, 384), (4, 512), (4, 640), (4, 768), (4, 896), (4, 1024), (4, 1152), (4, 1280), (4, 1408), (4, 1536), (4, 1664), (4, 1792), (4, 1920), (4, 2048)]
INFO 08-02 17:38:43 hpu_model_runner.py:1206] Warmup finished in 49 secs, allocated 14.19 GiB of device memory
INFO 08-02 17:38:43 hpu_executor.py:91] init_cache_engine took 37.92 GiB of device memory (53.39 GiB/94.62 GiB used) and 57.86 MiB of host memory (475.4 GiB/1007 GiB used)
```

## Sampler Warm-up

The sampler converts model logits into next-token selections, using configured decoding strategies, such as greedy or probabilistic. Its warm-up phase prepares compiled graph variants or internal code paths for a representative set of batch sizes and sampling parameter combinations, so that the first real user requests avoid extra compilation or setup latency.

Warmup ensures that common hyperparameter combinations are compiled ahead of time and that both greedy and random branching strategies, as well as metadata refresh paths, are exercised and stabilized. It also handles batch growth or shrink scenarios, smoothing later scaling behavior. Skipping the sampler warm-up does not affect correctness - only the latency profile of the earliest varied sampling requests. The following list presents the results of lacking the warm-up:

- The first request using a new configuration (such as the first `high-temp` with `top-k` path, or the first batch size after scaling up load) may trigger graph recompilation, adding latency for that request.
- Tail latency variance increases as early requests with diverse workloads can trigger multiple staggered compilations.
- Batch-size transition logic, where paths are set to `batch_changed=True`, may pay initialization cost during live traffic.

This warm-up process is skipped when:

- `VLLM_SKIP_WARMUP` is set to true.
- The engine is configured to enforce eager execution in a mode where no graph capture or compilation is desired and the sampler still runs the first time on demand, but without a separate warm-up call.

When introducing new sampling behaviors, such as nucleus filtering, penalties, or speculative metadata, update `sampling_configs` in `warmup_sampler` to ensure the corresponding graph paths are precompiled and ready.

Decode bucket configuration environment variables indirectly determine which batch sizes the sampler warms up, since the sampler derives its test batch sizes from the decode buckets.

### Performing Warm-up

Implemented in `warmup_sampler`, the warm-up routine systematically exercises the sampling stack across a Cartesian set of patterns, such as batch size, temperature, top-p, and top-k along with a flag that indicates whether the batch size has changed. To perform the warm-up, follow this procedure:

1. Build a list of test batch sizes by prepending `[0, 1]` to the distinct decode bucket batch sizes, as these batches must always be warmed up.

2. Define 12 sampling configurations covering the following settings. Each configuration should appear twice — once with `batch_changed=True` and once with `batch_changed=False` — to exercise any internal fast-path or cache invalidation logic tied to batch resizing.

   - Greedy: `temperature=0.0`
   - Typical random sampling: `temperature=1.0`
   - Creative settings:`0.7/0.9/top-k=50`
   - Conservative: `0.3/0.95/top-k=20`
   - High temperature: `1.2/0.8/top-k=100`
   - Top-p only variants: e.g. `0.8/0.85/top-k=0`

3. Prepare dummy data for each batch size by creating a hidden state tensor with shape `(batch_size, hidden_size)` and compute logits using `model.compute_logits`.

4. Instantiate at least one dummy request object for each batch size, providing placeholder prompt tokens and a single KV block.

5. For each configuration, follow these substeps:

    1. Update  `SamplingParams`, such as `temperature`, `top_p`, and `top_k` in each request.
    2. Mark the request as greedy or random to test branching.
    3. Populate `req_output_token_ids` with padded placeholders and refresh internal sampling metadata.
    4. Invoke `_run_sampling` passing `batch_changed` so both changed and unchanged batch-size code paths get compiled or exercised.
    5. Reset per-iteration sampler bookkeeping sets or lists.

6. After finishing all sampling configs for a batch size, clear request maps and continue.

7. Perform an HPU synchronize and log success.

### Logs

The following example presents a typical sequence of logs that appear during warm-up:

```text
INFO 09-22 16:39:42 [hpu_model_runner.py:3347] Warming up sampler with batch sizes: [0, 1, 138] and following configs:
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=0.0, top_p=1.0, top_k=0, batch_changed=True
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=1.0, top_p=1.0, top_k=0, batch_changed=True
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=0.7, top_p=0.9, top_k=50, batch_changed=True
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=0.3, top_p=0.95, top_k=20, batch_changed=True
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=1.2, top_p=0.8, top_k=100, batch_changed=True
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=0.8, top_p=0.85, top_k=0, batch_changed=True
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=0.0, top_p=1.0, top_k=0, batch_changed=False
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=1.0, top_p=1.0, top_k=0, batch_changed=False
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=0.7, top_p=0.9, top_k=50, batch_changed=False
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=0.3, top_p=0.95, top_k=20, batch_changed=False
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=1.2, top_p=0.8, top_k=100, batch_changed=False
INFO 09-22 16:39:42 [hpu_model_runner.py:3349] temp=0.8, top_p=0.85, top_k=0, batch_changed=False
INFO 09-22 16:39:42 [hpu_model_runner.py:3350] Starting sampler warmup...
INFO 09-22 16:39:43 [hpu_model_runner.py:3411] Sampler warmup completed successfully
```

If warm-up is globally skipped, these logs do not appear.

## Defragmenter Warm-up

The defragmenter reclaims and compacts sparse KV-cache block usage at runtime by swapping rarely packed high-index blocks with lower free indices. Its warm-up phase pre-compiles the small swap graphs so that later online defragmentation can execute with near-zero graph compile latency.

Defragmentation may be triggered mid-serving when the highest allocated block index drifts far above the actual number of in-use blocks (fragmentation). The operation itself is a sequence of swap kernels applied over key and value caches. With warm-up, all representative padded sizes are precompiled ahead of time via a deterministic, minimal swap. This ensures that online defragmentation becomes a predictable, low-latency maintenance task. Skipping only the defragmenter warm-up does not compromise correctness; it only increases the risk of sporadic latency when fragmentation first exceeds the threshold that mandates compaction.

The potential consequences of omitting warm-up include:

- The first fragmentation event that requires a previously unseen padded swap size triggers graph capture and compilation on the critical path.
- Compilation latency can manifest as a sudden tail-latency spike for a user request.
- Multiple first-seen swap sizes across different processes may each trigger separate compilations.

You can disable either the warm-up step itself or the entire defragmentation feature. To skip all warm-up phases, including the defragmenter, set `VLLM_SKIP_WARMUP=true`. Alternatively, running without unified attention effectively disables the defragmenter, since it is tied to unified attention; in this case, the warm-up becomes a no-op. Note that there is no separate environment flag in this version to force-enable or disable defragmentation independently of unified attention. Additionally, if supported by your execution mode, you can avoid graph compilation for defragmenter swaps by setting `VLLM_DEFRAG_WITH_GRAPHS=false`. This causes swaps to fall back to regular execution, while the warm-up still exercises them without triggering graph capture.

Related environment variables:

- `VLLM_DEFRAG_THRESHOLD`: Sets the fragmentation trigger heuristic. The default value is 32; lower values make compaction more aggressive.
- `VLLM_DEFRAG_WITH_GRAPHS`: Determines whether swap paths are compiled or graphed. By default, this follows `bridge_mode == eager`.
- `VLLM_DEBUG=defrag`: Enables verbose defragmentation debug logging.
- `VLLM_SKIP_WARMUP`: Disables all warm-up stages including defragmentation.

!!! note
    Disabling the defragmenter warm-up does not turn off defragmentation itself, unless unified attention or the feature is entirely disabled. It simply skips ahead-of-time graph preparation, which may shift the compilation cost to the first live fragmentation event.

### Defragmenter Warm-Up Process

During the main warm-up (`warmup_model`), the system calls the internal `warmup_defragmenter` method after initializing the KV caches and defragmenter. The process is defined by following warm-up steps:

1. Confirming that the defragmenter warm-up feature is enabled, as it only runs when unified attention is enabled, and that the `cache_utils` swap utilities are ready.
2. Establishing the list of padding thresholds: `[8, 16, 32, 64, 128, 256, 512]`.
3. Choosing a minimal valid swap pair `[(1, 0)]` with two distinct block IDs. Only two real blocks are required. Internally, each swap call is padded up to the current threshold length so that a compiled graph for that exact padded size is produced.
4. Iterating through each threshold and invoking a swap. This captures or compiles, depending on the execution mode, the swap graph for that padded size.
5. Performing one extra swap with the first threshold in cases when the number of thresholds is odd. It causes the sequence of swaps to return the KV cache to its original state (net zero logical change).
6. Completing logs.

Future defragmentation swap requests always round or pad to one of these known thresholds. All operational swap sizes hit a pre-compiled path and avoid on-demand compilation latency.

### Logs

The following example presents a typical sequence of logs that appear when there are at least two KV-cache blocks available:

```text
INFO 09-22 16:26:24 [hpu_model_runner.py:3428] Warming up defragmenter with thresholds: [8, 16, 32, 64, 128, 256, 512]
INFO 09-22 16:26:27 [hpu_model_runner.py:3452] Defragmenter warmup completed successfully
```

If insufficient blocks exist, such as extremely small test configuration or allocation failure, warm-up is skipped gracefully and you may see logs similar to the following example:

```text
INFO 09-22 16:26:24 [hpu_model_runner.py:3428] Warming up defragmenter with thresholds: [8, 16, 32, 64, 128, 256, 512]
WARNING hh:mm:ss hpu_model_runner.py:#### Skipping defragmenter warmup, insufficient blocks (1)
```

To emit fine-grained debug messages during live defragmentation, not the minimal warm-up swaps only, add `VLLM_DEBUG=defrag` to the environment. This way you will be able to see the number of blocks swapped and post-compaction statistics.
