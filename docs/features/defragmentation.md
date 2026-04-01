---
title: Defragmentation
---

# Defragmentation

The KV-cache defragmenter is an online compaction mechanism that keeps KV-cache block allocation dense. As requests arrive and complete, the blocks they occupy are allocated and freed in arbitrary order, which over time causes **fragmentation**: a small number of live blocks are scattered across a much larger address range. Because every forward pass must index into all live blocks, a fragmented layout wastes memory bandwidth and may prevent new blocks from being allocated even though the total free capacity is sufficient.

The defragmenter monitors the gap between the highest allocated block index and the actual count of in-use blocks. When the gap exceeds a configurable threshold it performs a **swap pass**, moving the highest-index live blocks into lower free slots, and symmetrically relocating any data that occupied those free slots back into the vacated positions.

## How It Works

### Block Tracking

The `OnlineDefragmenter` maintains two data structures:

- **Reference-counted block map** (`used_blocks`): Records how many active requests reference each physical block ID. Blocks with a reference count of zero are considered free.
- **Bidirectional mapping tables** (`fwd_mapping_table` / `bwd_mapping_table`): Translate between the *original* (logical) block IDs issued by the scheduler and the *physical* block IDs that point into the actual KV-cache tensors. After a swap the scheduler still refers to the original ID, but the defragmenter resolves it to the new physical location.

Every scheduler step calls `update_state`, which:

1. Increases the reference count for every block in newly scheduled or cached requests.
2. Decreases the reference count for every block belonging to finished requests, removing blocks that reach zero.

### Fragmentation Detection

After updating state, the `defragment` method checks:

```
max_used_block_id − threshold > num_used_blocks
```

If this inequality holds, the block range is considered fragmented and a compaction pass begins. The default threshold is **32** and is controlled by `VLLM_DEFRAG_THRESHOLD`. Lowering the threshold makes compaction trigger more frequently.

### Swap Pass

When compaction is triggered:

1. **Identify candidates**: The used blocks are sorted in descending order (highest first) and paired with the lowest available free slots produced by a generator.
2. **Pair validation**: A pair `(high_block, low_free)` is only valid when `high_block > low_free`. The first pair that violates this condition ends the pass. The number of pairs is also capped at **512** (the largest padding threshold).
3. **Reference and mapping update**: For each pair the reference counts are swapped, and the forward/backward mapping tables are updated so that future `resolve()` calls translate the original block ID to the new physical position.
4. **Physical data swap**: The `CacheSwapUtils` module copies KV-cache rows between the source and destination slots on the HPU. Key caches and value caches are swapped in separate calls. For models using Multi-head Latent Attention (MLA), where only a key cache exists, the value-cache swap is skipped.

### Padded Swap Sizes

To avoid HPU graph recompilation, swap tensors are padded to a fixed set of sizes: **8, 16, 32, 64, 128, 256, 512**. The smallest threshold that accommodates the actual number of swaps is chosen. This means every swap operation maps to a pre-compiled graph when graph mode is active.

### Block ID Resolution

After defragmentation, any code that converts scheduler-provided block IDs into physical cache indices must call `resolve()` (single block) or `resolve_all()` (batch). The model runner applies this resolution transparently when building attention metadata, so callers outside the model runner do not need to be aware of block remapping.

## Configuration

| Environment Variable | Type | Default | Description |
|---|---|---|---|
| `VLLM_DEFRAG_THRESHOLD` | `int` | `32` | Fragmentation trigger gap. Compaction runs when `max_block_id − threshold > num_used`. Lower values compact more aggressively. |
| `VLLM_DEFRAG_WITH_GRAPHS` | `bool` | Follows `bridge_mode == eager` | Whether swap operations use compiled/HPU graphs. When `false`, swaps execute without graph capture. |
| `VLLM_DEBUG=defrag` | flag | — | Enables verbose logging of defragmentation events, including the number of blocks swapped and post-compaction statistics. |
| `VLLM_SKIP_WARMUP` | `bool` | `false` | Skips all warm-up stages including defragmenter warm-up. Does **not** disable defragmentation itself. |

The feature itself is controlled by the `defrag` feature flag in `vllm_gaudi/extension/features.py` (default: `False`).

## Warm-Up

Before the server begins accepting requests, the defragmenter optionally pre-compiles swap graphs for all padded sizes. This eliminates graph compilation latency during live serving. The warm-up performs a minimal swap `[(1, 0)]` for each threshold and then reverses it so the KV cache returns to its original state. For details, see [Defragmenter Warm-Up](../configuration/warm-up/defragmenter_warm-up.md).

## Integration

The defragmenter is invoked at the start of every `execute_model` call in the HPU model runner:

1. `run_defragmenter` collects newly allocated and freed blocks from the `SchedulerOutput`.
2. It calls `update_state` followed by `defragment`.
3. The rest of `execute_model` uses `_resolve_block` and `_resolve_all_blocks` to translate block IDs.

During KV-cache lifecycle events (sleep/wake), the worker destroys and recreates the defragmenter to ensure a clean mapping state.

## Prefix Caching Interaction

When automatic prefix caching (APC) is enabled, the `CacheSwapUtils` performs a **bidirectional** copy: it first reads the data from both the source and destination blocks, then writes each into the other's position. This preserves cached prefix data that might reside in nominally "free" blocks, preventing silent cache corruption.
