---
title: Supported Features
---
[](){ #supported-features }

# Supported Features
| **Feature**   | **Description**   | **References**  |
|---    |---    |---    |
| Offline batched inference     | Offline inference using LLM class from vLLM Python API    | [Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart.html#offline-batched-inference)<br>[Example](https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference.html)   |
| Online inference via OpenAI-Compatible Server     | Online inference using HTTP server that implements OpenAI Chat and Completions API    | [Documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html)<br>[Example](https://docs.vllm.ai/en/stable/getting_started/examples/openai_chat_completion_client.html)    |
| HPU autodetection     | HPU users do not need to specify the target platform, it will be detected automatically upon vLLM startup     | N/A   |
| Paged KV cache with algorithms enabled for Intel Gaudi accelerators   | vLLM HPU backend contains a custom Paged Attention and cache operators implementations optimized for Gaudi devices.   | N/A   |
| Custom Intel Gaudi operator implementations   | vLLM HPU backend provides optimized implementations of operators such as prefill attention, Root Mean Square Layer Normalization, Rotary Positional Encoding.     | N/A   |
| Tensor parallel inference (single or multi-node multi-HPU)     | vLLM HPU backend supports multi-HPU inference across multiple nodes with tensor parallelism with multiprocessing or Ray and HCCL.  | [Documentation](https://docs.vllm.ai/en/stable/serving/distributed_serving.html)<br>[Example](https://docs.ray.io/en/latest/serve/tutorials/vllm-example.html)<br>[HCCL reference](https://docs.habana.ai/en/latest/API_Reference_Guides/HCCL_APIs/index.html)    |
| Pipeline parallel inference (single or multi-node multi-HPU)   | vLLM HPU backend supports multi-HPU inference across single or multi-node with pipeline parallelism.   | [Documentation](https://docs.vllm.ai/en/stable/serving/distributed_serving.html)<br> [Running Pipeline Parallelism](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#pipeline-parallelism)   |
| Inference with HPU Graphs     | vLLM HPU backend uses HPU Graphs by default for optimal performance. When HPU Graphs are enabled, execution graphs will be recorded ahead of time and replayed later during inference, significantly reducing host overheads.  | [Documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)<br>[vLLM HPU backend execution modes](https://docs.vllm.ai/en/stable/getting_started/gaudi-installation.html#execution-modes)<br>[Optimization guide](https://docs.vllm.ai/en/latest/getting_started/gaudi-installation.html#hpu-graph-capture)    |
| Inference with torch.compile   | vLLM HPU backend supports inference with `torch.compile`.    | [vLLM HPU backend execution modes](https://docs.vllm.ai/en/stable/getting_started/gaudi-installation.html#execution-modes)    |
| INC quantization  | vLLM HPU backend supports FP8 model and KV cache quantization and calibration with Intel Neural Compressor (INC). (Not fully supported with torch.compile execution mode)    | [Documentation](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_FP8.html)   |
| AutoAWQ quantization | vLLM HPU backend supports inference with models quantized using AutoAWQ library. | [Library](https://github.com/casper-hansen/AutoAWQ) |
| AutoGPTQ quantization | vLLM HPU backend supports inference with models quantized using AutoGPTQ library. | [Library](https://github.com/AutoGPTQ/AutoGPTQ) |
| LoRA/MultiLoRA support    | vLLM HPU backend includes support for LoRA and MultiLoRA on supported models.     | [Documentation](https://docs.vllm.ai/en/stable/models/lora.html)<br>[Example](https://docs.vllm.ai/en/stable/getting_started/examples/multilora_inference.html)<br>[vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)   |
| Fully async model executor     | This allows the model runner to function asynchronously when using async scheduling. This allows full overlap of the cpu operations (including prepare_inputs) and the model forward pass. This does not support speculative decoding, PP, or guided decoding. Expected speedup is 5-10% over the current async scheduling.   | [Feature description](https://github.com/vllm-project/vllm/pull/23569)   |
| Automatic prefix caching   | vLLM HPU backend includes automatic prefix caching (APC) support for more efficient prefills, configurable by standard `--enable-prefix-caching` parameter.   | [Documentation](https://docs.vllm.ai/en/stable/automatic_prefix_caching/apc.html)<br>[Details](https://docs.vllm.ai/en/stable/automatic_prefix_caching/details.html)  |
| Speculative decoding (functional release)     | vLLM HPU backend includes experimental speculative decoding support for improving inter-token latency in some scenarios, configurable via standard `--speculative_model` and `--num_speculative_tokens` parameters. (Not fully supported with torch.compile execution mode)   | [Documentation](https://docs.vllm.ai/en/stable/models/spec_decode.html)<br>[Example](https://docs.vllm.ai/en/stable/getting_started/examples/mlpspeculator.html)  |
| Multiprocessing backend   | Multiprocessing is the default distributed runtime in vLLM. The vLLM HPU backend supports it alongside Ray.   | [Documentation](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)  |
| Multimodal   | vLLM HPU backend supports the inference for multi-modal models. (Not fully supported with t.compile execution mode) |  [Documentation](https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html) |
| Multinode support   | vLLM HPU backend supports distributed, multiple-node inference with Ray.    | <REF>  |
| Guided decode   | vLLM HPU supports a guided decoding backend for generating structured outputs.   | [Documentation](https://docs.vllm.ai/en/latest/features/structured_outputs.html)  |
| Exponential bucketing | vLLM HPU supports exponential bucketing spacing instead of linear to automate configuration of bucketing mechanism, enabled by default. It can be disabled via `VLLM_EXPONENTIAL_BUCKETING=false` environment variable.   | N/A |
| Data Parellel support | vLLM HPU supports Data Parellel | [Documentation](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment.html)<br>[Example](https://docs.vllm.ai/en/latest/examples/offline_inference/data_parallel.html)  |

# Comming Soon
- Sliding window attention
- P/D disaggregate support
- In-place weight update
- MLA with Unified Attention
