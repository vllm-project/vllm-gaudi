# Release Notes

This document provides an overview of the features, changes, and fixes introduced in each release of the vLLM Hardware Plugin for Intel® Gaudi®.

## 0.13.0

This version is based on [vLLM 0.13.0](https://github.com/vllm-project/vllm/releases/tag/v0.13.0) and supports [Intel® Gaudi® v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html).

The release includes experimental dynamic quantization for MatMul and KV‑cache operations. This feature improves performance, with minimal expected impact on accuracy. To enable the feature, see the [Dynamic Quantization for MatMul and KV‑cache Operations](features/supported_features.md#dynamic-quantization-for-matmul-and-kv-cache-operations) section.

This release also introduces support for the following models supported on Gaudi 3:

- [bielik-11b-v2.6-instruct](https://huggingface.co/speakleash/Bielik-11B-v2.6-Instruct)
- [bielik-1.5b-v3.0-instruct](https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct)
- [bielik-4.5b-v3.0-instruct](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct)
- [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
- [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

Additionally, the following models were successfully validated:

- [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B)
- [meta-llama/Meta-Llama-3.1-70B](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B)
- [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
- [meta-llama/Meta-Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B)
- [meta-llama/Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

For the list of all supported models, see [Validated Models](getting_started/validated_models.md).

## 0.11.2

This version is based on [vLLM 0.11.2](https://github.com/vllm-project/vllm/releases/tag/v0.11.2) and supports [Intel® Gaudi® v1.22.2](https://docs.habana.ai/en/v1.22.2/Release_Notes/GAUDI_Release_Notes.html) and [v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html).

This release introduces the production-ready vLLM Hardware Plugin for Intel® Gaudi®, a community-driven integration layer based on the [vLLM v1 architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html). It enables efficient, high-performance large language model (LLM) inference on [Intel® Gaudi®](https://docs.habana.ai/) AI accelerators. The plugin is an alternative to the [vLLM fork](https://github.com/HabanaAI/vllm-fork), which reaches end of life with this release and will be deprecated in v1.24.0, remaining functional only for legacy use cases. We strongly encourage all fork users to begin planning their migration to the plugin.

The plugin provides [feature parity](features/supported_features.md) with the fork, including mature, production-ready implementations of Automatic Prefix Caching (APC) and async scheduler. Two legacy features - multi-step scheduling and delayed sampling - have been discontinued, as their functionality is now covered by the async scheduler.

For more details on the plugin's implementation, see [Plugin System](dev_guide/plugin_system.md).

To start using the plugin, follow the [Basic Quick Start Guide](getting_started/quickstart/quickstart.md) and explore the rest of this documentation.
