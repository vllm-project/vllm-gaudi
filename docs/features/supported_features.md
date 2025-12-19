---
title: Supported Features
---

# Supported Features

This document summarizes the features currently supported by the vLLM Hardware Plugin for Intel® Gaudi®, lists the features planned for future releases, and outlines the discontinued features with explanations for their deprecation.

## Supported Features

| **Feature**   | **Description**   | **References**  |
|---    |---    |---    |
| Offline batched inference     | Supports offline inference using the LLM class from vLLM Python API.    | [Quickstart](../getting_started/quickstart/quickstart_inference.md#offline-batched-inference), [Example](https://docs.vllm.ai/en/stable/examples/offline_inference/batch_llm_inference.html)   |
| Online inference via the OpenAI-Compatible Server     | Supports online inference through an HTTP server that implements the OpenAI Chat and Completions API.    | [Documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html), [Example](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)    |
| HPU autodetection     | Enables automatic target platform detection for HPU users at vLLM startup.     | N/A   |
| Paged KV cache with algorithms enabled for Intel® Gaudi® accelerators   | Provides a custom paged attention and cache operators implementations optimized for Intel® Gaudi® devices.   | N/A   |
| Custom Intel® Gaudi® operator implementations   | Provides optimized implementations of operators, such as prefill attention, Root Mean Square Layer Normalization, and Rotary Positional Encoding.     | N/A   |
| Tensor parallel inference      | Supports multi-HPU inference with tensor parallelism and multiprocessing.  | [Documentation](https://docs.vllm.ai/en/v0.10.0/serving/distributed_serving.html), [HCCL reference](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/index.html)    |
| Inference with HPU Graphs     | Reduces host overheads by using HPU Graphs, which record execution graphs ahead of time and replay them during inference.  | [Documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)   |
| Inference with `torch.compile`   | Supports inference with `torch.compile`, which is the default setting for HPU.    | [vLLM HPU backend execution modes](https://docs.vllm.ai/en/v0.10.1.1/getting_started/installation/intel_gaudi.html#execution-modes)    |
| INC quantization  | Supports the FP8 model, KV cache quantization, and calibration with Intel Neural Compressor (INC). This feature is not fully supported with the `torch.compile` execution mode.    | [Documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html)   |
| AutoAWQ quantization | Supports inference with models quantized using the AutoAWQ library. | [Library](https://github.com/casper-hansen/AutoAWQ) |
| AutoGPTQ quantization | Supports inference with models quantized using the AutoGPTQ library. | [Library](https://github.com/AutoGPTQ/AutoGPTQ) |
| LoRA/MultiLoRA support    | Supports LoRA and MultiLoRA on compatible models.     | [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html), [Example](https://docs.vllm.ai/en/stable/examples/offline_inference/multilora_inference.html)   |
| Fully async model executor     | Allows the model runner to run asynchronously with async scheduling, overlapping CPU operations , including `prepare_inputs`, and the model forward pass. It does not support speculative decoding, PP, or guided decoding. Expected speedup is 5-10% over the current async scheduling.   | [Feature description](https://github.com/vllm-project/vllm/pull/23569)   |
| Automatic Prefix Caching (APC)  | Improves prefills efficiency. This feature is enabled by default.  | [Documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)  |
| Speculative decoding (functional release)     | Supports experimental speculative decoding, which improves inter-token latency in some scenarios. The feature is configurable via the standard `--speculative_model` and `--num_speculative_tokens` parameters. It is not fully supported with the `torch.compile` execution mode.   | [Documentation](https://docs.vllm.ai/en/stable/features/spec_decode.html), [Example](https://docs.vllm.ai/en/stable/examples/offline_inference/spec_decode.html)  |
| Multiprocessing backend   | The default distributed runtime in vLLM.   | [Documentation](https://docs.vllm.ai/en/v0.10.0/serving/distributed_serving.html)  |
| Multimodal   | Supports inference for multi-modal models. It is not fully supported with the `t.compile` execution mode. |  [Documentation](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html) |
| Guided decode   | Supports a guided decoding backend for generating structured outputs.   | [Documentation](https://docs.vllm.ai/en/latest/features/structured_outputs.html)  |
| Exponential bucketing | Supports exponential bucketing spacing instead of linear spacing, automating the configuration of the bucketing mechanism. This feature is enabled by default and can be disabled via `VLLM_EXPONENTIAL_BUCKETING=false` environment variable.   | N/A |
| Data Parallel support | Replicates model weights across multiple instances or GPUs to process independent request batches. | [Documentation](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment.html), [Example](https://docs.vllm.ai/en/latest/examples/offline_inference/data_parallel.html)  |

## Experimental Features

### Runtime Scale Patching

Warm-up time for FP8 models is significantly longer than for BF16 due to additional graph
compilations triggered by varying constant scale values in quantized model layers.

You can reduce the FP8 warm-up time by setting the `RUNTIME_SCALE_PATCHING=1` environment variable and
selecting a hardware-aligned per-tensor `scale_method` provided by the `INC JSON config <json-options>`.
This feature is recommended for larger models, such as 70B and 405B. When combined with
`VLLM_EXPONENTIAL_BUCKETING` for FP8 models, it can reduce warm-up time by up to 90%.

!!!note
    This feature reduces FP8 warm-up time but may lower model throughput by 5-20%. Future releases will improve performance and extend support to more options. Currently, the feature is supported with Lazy mode (`PT_HPU_LAZY_MODE=1`) and `torch.compile`. It supports Llama workloads using FP8 execution of Linear and FSDPA layers, and casting ops between BF16 and FP8. MoE and Convolution options are not yet supported.

### Trivial Scales Optimization

The `PT_HPU_H2D_TRIVIAL_SCALES_MODE` flag controls the optimization of trivial scales, such as scale values equal to 1.0, in the `RUNTIME_SCALE_PATCHING` mode. Enabling this optimization can increase warm-up and compilation time because additional graphs are generated, but it may improve runtime performance by reducing the number of multiplication operations.

The following values are supported:

- `0`: No optimization (default).
- `1`: Removes scales equal to 1.0 in `cast_to_fp8_v2` and `cast_from_fp8`, disabling the corresponding `mult_fwd` (multiplication) node.
- `2`: Applies the same optimization as mode `1`, and additionally removes reciprocal scales in `fp8_gemm_v2`.

## Planned Features

Future plugin releases are planned to provide support for the following vLLM features:

- Sliding window attention
- P/D disaggregate support
- In-place weight update
- MLA with unified attention
- Multinode support
- Pipeline parallel inference

## Discontinued Features

| **Feature**   | **Description**   | **Reasoning**  |
|---    |---    |---    |
| Multi-step scheduling      | Multi-step scheduling support for host overhead reduction.    | Replaced by async scheduling, configurable via the `--async_scheduling` parameter.    |
| Delayed Sampling     | Support for delayed sampling scheduling for asynchronous execution.    | Replaced by async scheduling, configurable via the `--async_scheduling` parameter.   |
