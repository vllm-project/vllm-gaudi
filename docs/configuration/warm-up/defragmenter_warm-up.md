# Defragmenter Warm-Up

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

## Performing Defragmenter Warm-Up

During the main warm-up (`warmup_model`), the system calls the internal `warmup_defragmenter` method after initializing the KV caches and defragmenter. The process is defined by following warm-up steps:

1. Confirming that the defragmenter warm-up feature is enabled, as it only runs when unified attention is enabled, and that the `cache_utils` swap utilities are ready.
2. Establishing the list of padding thresholds: `[8, 16, 32, 64, 128, 256, 512]`.
3. Choosing a minimal valid swap pair `[(1, 0)]` with two distinct block IDs. Only two real blocks are required. Internally, each swap call is padded up to the current threshold length so that a compiled graph for that exact padded size is produced.
4. Iterating through each threshold and invoking a swap. This captures or compiles, depending on the execution mode, the swap graph for that padded size.
5. Performing one extra swap with the first threshold in cases when the number of thresholds is odd. It causes the sequence of swaps to return the KV cache to its original state (net zero logical change).
6. Completing logs.

Future defragmentation swap requests always round or pad to one of these known thresholds. All operational swap sizes hit a pre-compiled path and avoid on-demand compilation latency.

## Logs

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
