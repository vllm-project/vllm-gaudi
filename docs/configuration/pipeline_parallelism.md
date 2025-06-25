# Pipeline Parallelism

Pipeline parallelism is a distributed model parallelization technique that splits the model vertically across its layers, distributing different parts of the model across multiple devices.
With this feature, when running a model that does not fit on a single node with tensor parallelism and requires a multi-node solution, we can split the model vertically across its layers and distribute the slices across available nodes.
For example, if we have two nodes, each with 8 HPUs, we no longer have to use `tensor_parallel_size=16` but can instead set `tensor_parallel_size=8` with pipeline_parallel_size=2.
Because pipeline parallelism runs a `pp_size` number of virtual engines on each device, we have to lower `max_num_seqs` accordingly, since it acts as a micro batch for each virtual engine.

## Running Pipeline Parallelism

The following example shows how to use Pipeline parallelism with vLLM on HPU:

```bash
vllm serve <model_path> --device hpu --tensor-parallel-size 8 --pipeline_parallel_size 2 --distributed-executor-backend ray
```

!!! note
    Currently, pipeline parallelism on Lazy mode requires the `PT_HPUGRAPH_DISABLE_TENSOR_CACHE=0` flag.
