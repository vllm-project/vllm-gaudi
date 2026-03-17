# Pipeline Parallelism

Pipeline parallelism is a distributed model parallelization technique that splits the model vertically across its layers, distributing different parts of the model across multiple devices. This approach is particularly useful when a model cannot fit on a single node using tensor parallelism alone and requires a multi-node setup. In such cases, the model's layers can be split across multiple nodes, allowing each node to handle a segment of the model. For example, if you have two nodes, each equipped with 8 HPUs, you no longer need to set `tensor_parallel_size=16`. Instead, you can configure `tensor_parallel_size=8` and `pipeline_parallel_size=2`.

The following example shows how to use the pipeline parallelism with vLLM on HPU:

```bash
vllm serve <model_path> --device hpu --tensor-parallel-size 8 --pipeline_parallel_size 2 --distributed-executor-backend ray
```

Since pipeline parallelism runs a `pp_size` number of virtual engines on each device, you have to lower `max_num_seqs` accordingly, as it acts as a micro batch for each virtual engine.

Currently, pipeline parallelism on the lazy mode requires the `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=0` flag.

## Optimizations

Pipeline parallelism on HPU uses non-blocking point-to-point communication to overlap the transfer of intermediate tensors between pipeline stages with ongoing computation. The key optimizations include:

- **Asynchronous intermediate tensor communication**: Non-first pipeline ranks initiate a receive of intermediate tensors from the previous stage before the forward pass begins. Non-last ranks use non-blocking sends (`isend`) to forward their intermediate tensors to the next stage, allowing the send to overlap with the next iteration's preprocessing.
- **PP-aware model execution**: Only the last pipeline rank performs logit computation and token sampling. Non-last ranks skip these steps and return `IntermediateTensors` directly, reducing unnecessary compute.
- **Scheduler-mediated token synchronization**: Sampled token IDs are propagated from the last pipeline rank back to all other ranks via the scheduler, avoiding the need for an additional broadcast or all-gather operation in the critical path.
