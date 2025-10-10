---
title: Bucketing Mechanism
---
[](){ #bucketing-mechanism }

## Bucketing Mechanism

Intel Gaudi accelerators perform best when operating on models with fixed tensor shapes. [Intel Gaudi Graph Compiler](https://docs.habana.ai/en/latest/Gaudi_Overview/Intel_Gaudi_Software_Suite.html#graph-compiler-and-runtime)
generates optimized binary code that implements the given model topology on Gaudi. In its default configuration, the produced binary code may be highly dependent on input and output tensor shapes, requiring graph recompilation
when encountering tensors with different shapes within the same topology. While these binaries efficiently utilize Gaudi, the compilation process itself can introduce noticeable overhead in end-to-end execution.
In dynamic inference serving scenarios, minimizing the number of graph compilations and reducing the risk of graph compilation occurring during server runtime is important. Currently, this is achieved by
"bucketing" the model's forward pass across three dimensions.

> [!NOTE]
> Bucketing helps significantly reduce the number of required graphs, but does not handle graph compilation or device code generation. These tasks are performed during the warmup and HPUGraph capture phase.

## Bucketing Strategies

Bucketing is focused on three dimensions:
- `batch size`: number of samples in batch
- `query lenght`: sequence length without context tokens
- `num blocks`: context length counted in blocks

Bucketing ranges are generated based on 4 parameters - `min`, `step`, `max` and `limit`, separately for the prompt and decode phase, and batch size, query length and context blocks dimensions. These parameters can be observed in logs during vLLM startup:

```{.}
INFO 07-07 19:27:37 [exponential.py:36] Prompt bucket config (min, step, max_warmup, limit) bs:[1, 1, 1, 1], seq:[128, 128, 1024, 11]
INFO 07-07 19:27:37 [common.py:85] Generated 36 prompt buckets [bs, query, num_blocks]: [(1, 128, 0), (1, 128, 1), (1, 128, 2), (1, 128, 3), (1, 128, 4), (1, 128, 5), (1, 128, 6), (1, 128, 7), (1, 256, 0), (1, 256, 1), (1, 256, 2), (1, 256, 3), (1, 256, 4), (1, 256, 5), (1, 256, 6), (1, 384, 0), (1, 384, 1), (1, 384, 2), (1, 384, 3), (1, 384, 4), (1, 384, 5), (1, 512, 0), (1, 512, 1), (1, 512, 2), (1, 512, 3), (1, 512, 4), (1, 640, 0), (1, 640, 1), (1, 640, 2), (1, 640, 3), (1, 768, 0), (1, 768, 1), (1, 768, 2), (1, 896, 0), (1, 896, 1), (1, 1024, 0)]
INFO 07-07 19:27:37 [common.py:85] Generated 42 decode buckets [bs, query, num_blocks]: [(1, 1, 128), (1, 1, 256), (1, 1, 384), (1, 1, 512), (1, 1, 640), (1, 1, 768), (1, 1, 896), (1, 1, 1024), (1, 1, 1408), (1, 1, 1792), (1, 1, 2432), (1, 1, 3328), (1, 1, 4352), (1, 1, 5888), (2, 1, 128), (2, 1, 256), (2, 1, 384), (2, 1, 512), (2, 1, 640), (2, 1, 768), (2, 1, 896), (2, 1, 1024), (2, 1, 1408), (2, 1, 1792), (2, 1, 2432), (2, 1, 3328), (2, 1, 4352), (2, 1, 5888), (4, 1, 128), (4, 1, 256), (4, 1, 384), (4, 1, 512), (4, 1, 640), (4, 1, 768), (4, 1, 896), (4, 1, 1024), (4, 1, 1408), (4, 1, 1792), (4, 1, 2432), (4, 1, 3328), (4, 1, 4352), (4, 1, 5888)]
```

> [!WARNING]
> If a request exceeds the maximum bucket size in any dimension, it will be processed without padding, and its processing may require a graph compilation, potentially significantly increasing end-to-end latency.
The boundaries of the buckets are user-configurable via environment variables, and upper bucket boundaries can be increased to avoid such scenario.

For example, if a request with 3 sequences, each having a maximum sequence length of 412, is sent to an idle vLLM server, it will be padded and executed as a `(4, 512, 0)` prefill bucket, WHERE 4=bs, 512 .... This is because the `batch_size`
(number of sequences) will be padded to 4 (the nearest batch size dimension higher than 3), and the maximum sequence length will be padded to 512 (the nearest sequence length dimension higher than 412). After the
prefill stage, it will be executed as a `(4, 1, 512)` decode bucket and will remain in this bucket until either the batch dimension changes (e.g., due to a request being completed), in which case it will become
a `(2, 1, 512)` bucket, or the context length increases beyond 512 tokens. It will become a `(4, 1, 640)` bucket at that point.

> [!NOTE]
> Bucketing is transparent to the user – padding in the sequence length dimension is never returned, and padding in the batch dimension does not create new requests.

### Exponential Strategy  - Default

Exponential strategy is the default warm-up mechanism. It is based on 4 parameters:
- `min`: the smallest value
- `step`: the rounding value for bucket boundaries
- `max`: the largest value
- `limit`: the maximum number of buckets

> [!WARNING]
> These parameters are not configurable by the user.

The exponential bucketing strategy applies exponential spacing between buckets. The `min` and `max` values are always included in the warm-up, and the intermediate values are calculated using an exponent. The base remains unchanged. If duplicate values are generated, they are removed to ensure the warm-up process is as efficient as possible. All the values generated in this way, ranging from batch size, query length and context blocks, will be warmed up with each other.

Example distribution is shown below:

```{.}
min = 128, step = 128, max = 4096, limit = 13
```

![exponential bucketing distribution for 4096 max query length](../../docs/assets/graphs/exponential_bucketing_example.png)

This strategy creates more buckets with smaller values closer to `min`. As the values increase toward `max`, the buckets become less frequent, meaning the distance between them gets larger. This helps prioritize warming up the smaller values more precisely, while still covering the full range.

### Linear Strategy

> [!NOTE]
> Starting from v1.22.0 Intel Gaudi Software release, Linear strategy is no longer the default warm-up mechanism.

Linear strategy is determined with 3 parameters only - `min`, `step` and `max`. They can be set separately for the prompt and decode phase, and batch size and sequence length dimensions, by user.

`min` determines the lowest value of the bucket. `step` determines the interval between buckets, and `max` determines the upper bound of the bucket. Furthermore, the interval between `min` and `step` has special handling: `min` is multiplied by consecutive powers of two until the multiplier is less than or equal to `step`. We refer to this as the ramp-up phase, which is used for handling lower batch sizes with minimal wastage, while allowing for larger padding on larger batch sizes.

**Example with ramp-up**

```{.}
min = 2, step = 32, max = 64
=> ramp_up = (2, 4, 8, 16)
=> stable = (32, 64)
=> buckets = ramp_up + stable => (2, 4, 8, 16, 32, 64)
```

**Example without ramp-up**

```{.}
min = 128, step = 128, max = 512
=> ramp_up = ()
=> stable = (128, 256, 384, 512)
=> buckets = ramp_up + stable => (128, 256, 384, 512)
```

### Unified Strategy

Unified strategy is dedicated strategy for Unified Attention. It's buckets are determined by different dimensions:
- `query length`: number of currently processed tokens, without context tokens
- `shared num blocks`: context length counted in blocks, including only blocks that are either shared between at least two block tables (different requests) or is used by at least two tokens in query
- `unique num blocks`: context length counted in blocks, including only blocks that are not shared between block tables and are used only by one token
- `is causal`: only two possible values: 0 and 1. Causal determines if there is at least one prompt in batch

Unified bucketing prepares buckets for both prompt and decode as one, known as `unified cfg`.

> [!WARNING]
> No parameters for unified warmup are configurable by the user.

**Alpha Version:**

Currently there are six points in ranges for query length, shared blocks and unique blocks. They are based on `max num seqs` and `max num batched tokens` values. Points are as follows whole, half and one quarter of both values resulting in six points in total.

Example distribution is shown below:

```{.}
batch size = 64, max num batched tokens = 4096
```

![exponential bucketing distribution for 4096 max query length](../../docs/assets/graphs/unified_bucketing_example.png)

Additionaly for context blocks, both shared and unique, `0` value will be added as well.

This way our bucketing will look like this:

```{.}
INFO 09-23 12:32:43 [common.py:100] Generated 375 unified buckets [query, shared_blocks, unique_blocks]: [(8, 0, 0, 1), (8, 0, 8, 0), ..., (2048, 256, 2890, 1), (2048, 256, 5781, 1)]
```

With every bucket logged separately in warm-up phase:

```{.}
(EngineCore_DP0 pid=805) INFO 09-23 12:32:50 [hpu_model_runner.py:3320] [Warmup][Unified CFG][2/375] query_len:2048 shared_blocks:256 unique_blocks:2890 (causal) free_mem:11.16 GiB
(EngineCore_DP0 pid=805) INFO 09-23 12:32:53 [hpu_model_runner.py:3320] [Warmup][Unified CFG][3/375] query_len:2048 shared_blocks:256 unique_blocks:1445 (causal) free_mem:11.16 GiB
(EngineCore_DP0 pid=805) INFO 09-23 12:32:56 [hpu_model_runner.py:3320] [Warmup][Unified CFG][4/375] query_len:2048 shared_blocks:256 unique_blocks:32 (causal) free_mem:11.16 GiB
```
