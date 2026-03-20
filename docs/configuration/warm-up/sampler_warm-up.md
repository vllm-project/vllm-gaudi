# Sampler Warm-Up

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

## Performing Warm-Up

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

## Logs

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
