<figure markdown="span" style="display: flex; justify-content: center; align-items: center; gap: 10px; margin: auto;">
  <img src="./assets/logos/vllm-logo-text-light.png" alt="vLLM" style="width: 30%; margin: 0;"> x
  <img src="./assets/logos/gaudi-logo.png" alt="Intel-Gaudi" style="width: 30%; margin: 0;">
</figure>

<p style="text-align:center">
</p>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm-gaudi" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/vllm-gaudi/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/vllm-gaudi/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

# Overview

The vLLM Hardware Plugin for Intel® Gaudi® is a community-driven integration layer that enables efficient, high-performance large language model (LLM) inference on Intel® Gaudi® AI accelerators.

The vLLM Hardware Plugin for Intel® Gaudi® connects the [vLLM serving engine](https://docs.vllm.ai/) with [Intel® Gaudi®](https://docs.habana.ai/) hardware, offering optimized inference capabilities for enterprise-scale LLM workloads. It is developed and maintained by the Intel® Gaudi® team and follows the [hardware pluggable RFC](https://github.com/vllm-project/vllm/issues/11162) and [vLLM plugin architecture RFC](https://github.com/vllm-project/vllm/issues/19161) for modular integration.

## 🚀 Advantages

The vLLM Hardware Plugin for Intel® Gaudi® offers the following key benefits:

- **Optimization for Intel® Gaudi®**: Supports advanced features, such as the bucketing mechanism, Floating Point 8-bit (FP8) quantization, and custom graph caching for fast warm-up and efficient memory use.
- **Scalability and efficiency**: Designed to maximize throughput and minimize latency for large-scale deployments, making it ideal for production-grade LLM inference.
- **Community support**: Actively maintained on [GitHub](https://github.com/vllm-project/vllm-gaudi) by contributions from the Intel® Gaudi® team and the broader vLLM ecosystem.

## ✅ Getting Started

To get started with vLLM Hardware Plugin for Intel® Gaudi®:

- [ ] **Set up your environment** using the [quickstart](getting_started/quickstart/quickstart.md) and plugin locally or in your containerized environment.
- [ ] **Run inference** using supported models like Llama 3.1, Mixtral, or DeepSeek.
- [ ] **Explore advanced features** such as FP8 quantization, recipe caching, and expert parallelism.
- [ ] **Join the community** by contributing to the [vLLM-Gaudi GitHub repo](https://github.com/vllm-project/vllm-gaudi).

For more information, see:

- 📚 [Intel® Gaudi® Documentation](https://docs.habana.ai/en/latest/index.html)  
- 📦 [vLLM Plugin System Overview](https://docs.vllm.ai/en/latest/design/plugin_system/)
