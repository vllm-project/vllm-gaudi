---
title: Warm-up
---
[](){ #Warm-up }

## Warm-up

Warm-up is a highly recommended step that occurs before the vLLM server starts listening. It performs a forward pass for each bucket using dummy data. The goal is to precompile all graphs and eliminate any graph compilation overhead within bucket boundaries during server runtime. Each warmup step is logged during vLLM startup.
This example uses the same buckets as those described in the Bucketing Mechanism section. Each output line corresponds to the execution of a single bucket. When a bucket is executed for the first time, its graph is compiled and can be reused later, avoiding further graph compilations.

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

> [!TIP]
> Compiling all buckets may take some time and can be disabled by setting the `VLLM_SKIP_WARMUP=true` environment variable. Keep in mind that if you do this, you may encounter graph compilations when executing a given bucket for the first time.

> [!WARNING]
> Disabling warmup is acceptable for development purposes, but it is strongly recommended to enable it in production deployments.

## HPU Graph Capture

[HPU Graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html) are currently the most performant execution method for vLLM on Intel Gaudi. When HPU Graphs are enabled,
execution graphs are be traced (recorded) ahead of time (after performing warmup), and later replayed during inference, significantly reducing host overheads. Recording can consume large amounts of memory, which
must be considred when allocating KV cache. Enabling HPU Graphs affects the number of available KV cache blocks, but vLLM provides user-configurable variables to manage memory usage.

When HPU Graphs are used, they share the common memory pool ("usable memory") with the KV cache, as determined by the `gpu_memory_utilization` flag (default value is `0.9`). Before allocating the KV cache,
the model weights are loaded onto the device, and a forward pass is executed on dummy data to estimate memory usage. Only then is the `gpu_memory_utilization` flag applied. At its default value,
it marks 90% of the free device memory at that point as usable. Next, the KV cache is allocated, the model is warmed up, and HPU Graphs are captured. The `VLLM_GRAPH_RESERVED_MEM` environment variable defines
the ratio of memory reserved for HPU Graph capture. With its default value (`VLLM_GRAPH_RESERVED_MEM=0.1`), 10% of the usable memory will be reserved for graph capture (referred to as "usable graph memory"), and the remaining 90% will be used for the KV cache.

> [!NOTE]
> `gpu_memory_utilization` does not represent the absolute memory usage across the HPU. Instead, it specifies the memory margin after loading the model and running a profiling pass. For example, if a device has 100 GiB of total memory and 50 GiB of free memory after loading the model weights and executing the profiling run, the default value of `gpu_memory_utilization` will mark 90% of the 50 GiB as usable, leaving 5 GiB as a margin - regardless of the total device memory.

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

## Sampler Warm-Up
The sampler converts model logits into next-token selections, using configured decoding strategies (greedy or probabilistic). Its warm-up phase prepares compiled graph variants (or internal code paths) for a representative set of batch sizes and sampling parameter combinations, so that first real user requests avoid extra compilation/setup latency.

### How the Sampler Warm-Up Works

Implemented in `warmup_sampler`, the routine systematically exercises the sampling stack across a Cartesian set of (batch size, temperature, top-p, top-k) patterns and a flag, that signals whether the batch size changed. Key steps:

1. Build a list of test batch sizes: it prepends `[0, 1]` to the distinct decode bucket batch sizes, as these need to be always warmed up.
2. Define a list of sampling configurations (12 total) covering:
   * Greedy (temperature=0.0)
   * Typical random sampling (temperature=1.0)
   * Creative settings (0.7/0.9/top-k=50)
   * Conservative (0.3/0.95/top-k=20)
   * High temperature (1.2/0.8/top-k=100)
   * Top-p only variants (e.g. 0.8/0.85/top-k=0)
    Each appears twice: once with `batch_changed=True` and once with `batch_changed=False` to exercise any internal fast-path or cache invalidation logic tied to batch resizing.
3. For every batch size:
   * Create a dummy hidden state tensor shaped `(batch_size, hidden_size)` and compute logits via `model.compute_logits`.
   * Instantiate dummy request objects (at least one) with placeholder prompt tokens and single KV block.
4. For each sampling configuration:
   * Update each request's `SamplingParams` (temperature, top_p, top_k).
   * Mark the request as greedy or random (separate sets) to test branching.
   * Populate `req_output_token_ids` with padded placeholders and refresh internal sampling metadata.
   * Invoke `_run_sampling` passing `batch_changed` so both changed/unchanged batch-size code paths get compiled/exercised.
   * Reset per-iteration sampler bookkeeping sets/lists.
5. After finishing all sampling configs for a batch size, clear request maps and continue.
6. Perform an HPU synchronize and log success.

### What the Logs Look Like

Typical sequence:

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

If warm-up is globally skipped ([see below](#how-to-turn-it-off)), none of these lines appear.

### Why We Warm Up the Sampler (and Risks If We Do Not)

Without sampler warm-up:
* The first real request using a new combination (e.g., first high-temp + top-k path, or first batch size after scaling up load) might incur graph recompilation, adding latency to that user request.
* Tail latency variance increases: early heterogeneous workloads cause multiple staggered compilations.
* Batch-size transition logic (paths where `batch_changed=True`) may pay initialization cost during live traffic.

With warm-up:
* Common sampling hyperparameter mixes are compiled ahead-of-time.
* Greedy vs random branching and metadata refresh code paths are stabilized.
* Batch growth/shrink handling is already exercised, smoothing later scaling behavior.

Skipping the sampler warm-up does not affect correctness—only the latency profile of the earliest varied sampling requests.

### How to Turn It Off

There is no dedicated flag for the sampler alone. It participates in the global warm-up sequence and is skipped when:

* `VLLM_SKIP_WARMUP=true` is set.
* The engine is configured to enforce eager execution in a mode where no graph capture/compilation is desired (sampler still runs the first time on demand, but without a separate warm-up call).

### Related Notes & Environment Variables

* `VLLM_SKIP_WARMUP` – Disables sampler warm-up along with other warm-up phases.
* Decode bucket configuration env vars indirectly influence the set of batch sizes the sampler warms up (since it derives test batch sizes from decode buckets).

> [!NOTE]
> If you introduce new sampling behaviors (e.g., new nucleus filtering, penalties, or speculative metadata), extend `sampling_configs` in `warmup_sampler` so their graph paths are primed.

## Defragmenter Warm-Up

The defragmenter reclaims and compacts sparse KV-cache block usage at runtime by swapping rarely packed high-index blocks with lower free indices. Its warm-up phase pre-compiles the small swap graphs so that later online defragmentation can execute with near-zero graph compile latency.

### How the Defragmenter Warm-Up Works

During the main warm-up (`warmup_model`) we call an internal method (`warmup_defragmenter`) after the KV caches and defragmenter have been initialized. The routine:

1. Verifies the feature is enabled (defragmenter only runs when unified attention is enabled) and that swap utilities (`cache_utils`) are prepared.
2. Determines the list of padding thresholds: `[8, 16, 32, 64, 128, 256, 512]`.
3. Chooses a minimal valid swap pair `[(1, 0)]` (two distinct block IDs). Only two real blocks are required; internally each swap call is padded up to the current threshold length so that a compiled graph for that exact padded size is produced.
4. Iterates through each threshold and invokes a swap. This captures/compiles (depending on execution mode) the swap graph for that padded size.
5. If the number of thresholds is odd, performs one extra swap with the first threshold so that the sequence of swaps returns the KV cache to its original state (net zero logical change).
6. Logs completion.

Because every future real defragmentation swap request will round/pad to one of these known thresholds, all operational swap sizes hit a pre-compiled path and avoid on-demand compilation latency.

### What the Logs Look Like

You will typically see one of two flows. If there are at least two KV-cache blocks available:

```text
INFO 09-22 16:26:24 [hpu_model_runner.py:3428] Warming up defragmenter with thresholds: [8, 16, 32, 64, 128, 256, 512]
INFO 09-22 16:26:27 [hpu_model_runner.py:3452] Defragmenter warmup completed successfully
```

If insufficient blocks exist (e.g., extremely small test configuration or allocation failure) warm-up is skipped gracefully:

```text
INFO 09-22 16:26:24 [hpu_model_runner.py:3428] Warming up defragmenter with thresholds: [8, 16, 32, 64, 128, 256, 512]
WARNING hh:mm:ss hpu_model_runner.py:#### Skipping defragmenter warmup, insufficient blocks (1)
```

Add `VLLM_DEBUG=defrag` to the environment to emit fine-grained debug messages during live defragmentation (not during the minimal warm-up swaps only) such as the number of blocks swapped and post-compaction statistics.

### Why We Warm Up (and What Happens If We Do Not)

Defragmentation may be triggered mid-serving when the highest allocated block index drifts far above the actual number of in-use blocks (fragmentation). The operation itself is a sequence of swap kernels over key & value caches. Without warm-up:

* The first fragmentation event that requires a new (previously unseen) padded swap size would incur graph capture/compilation in the critical path.
* That added latency can surface as a sudden tail-latency spike for a user request.
* Multiple different first-seen swap sizes across processes could each trigger separate compilations.

With warm-up, all representative padded sizes are compiled ahead-of-time via a deterministic, tiny swap, so online defragmentation becomes a predictable, low-latency maintenance task.

Skipping only the defragmenter warm-up does not break correctness; it only risks sporadic latency when fragmentation first crosses a threshold that mandates compaction.

### How to Turn It Off

You can disable (a) the warm-up step itself or (b) the entire defragmentation feature:

* Disable all warm-up phases (including defragmenter) by setting `VLLM_SKIP_WARMUP=true`.
* Run without unified attention (the defragmenter is tied to unified attention; if unified attention is disabled, `defrag` is not enabled and the warm-up is a no-op). There is no separate dedicated environment flag to force-enable/disable defrag beyond unified attention in this version.
* Avoid graph compilation for defragmenter swaps by setting `VLLM_DEFRAG_WITH_GRAPHS=false` (falls back to regular execution; warm-up will still exercise swaps but without graph capture), if supported by the execution mode.

Related environment variables:

* `VLLM_DEFRAG_THRESHOLD` – Fragmentation trigger heuristic (default 32). Lower values make compaction more aggressive.
* `VLLM_DEFRAG_WITH_GRAPHS` – Whether swap paths are compiled/graphed (defaults to `bridge_mode == eager`).
* `VLLM_DEBUG=defrag` – Enables verbose defragmentation debug logging.
* `VLLM_SKIP_WARMUP` – Disables all warm-up stages including this one.

> [!NOTE]
> Disabling defragmenter warm-up does not disable defragmentation itself (unless unified attention/the feature is off). It only removes ahead-of-time graph preparation, potentially pushing compile cost into the first live fragmentation event.
