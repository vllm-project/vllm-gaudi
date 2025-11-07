# Intel® Gaudi® vLLM Plugin

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

Welcome to the **vLLM-Gaudi plugin**, a community-maintained integration layer that enables high-performance large language model (LLM) inference on Intel® Gaudi® AI accelerators.

## 🔍 What is vLLM-Gaudi?

The **vLLM-Gaudi plugin** connects the vLLM serving engine with Intel Gaudi hardware, offering optimized inference capabilities for enterprise-scale LLM workloads. It is developed and maintained by Intel/Gaudi team and follows the Hardware Pluggable [RFC](https://github.com/vllm-project/vllm/issues/11162) and vLLM Plugin Architecture [RFC](https://github.com/vllm-project/vllm/issues/19161) for modular integration.

## 🚀 Why Use It?

- **Optimized for Gaudi**: Supports advanced features like bucketing mechanism, FP8 quantization, and custom graph caching for fast warm-up and efficient memory use.
- **Scalable and Efficient**: Designed to maximize throughput and minimize latency for large-scale deployments, making it ideal for production-grade LLM inference.
- **Community-Ready**: Actively maintained on [GitHub](https://github.com/vllm-project/vllm-gaudi) with contributions from Intel, Gaudi team, and the broader vLLM ecosystem.

## ✅ Action Items

To get started with the Intel® Gaudi® vLLM Plugin:

- [ ] **Set up your environment** using the [quickstart](getting_started/quickstart.md) and plugin locally or in your containerized environment.
- [ ] **Run inference** using supported models like Llama 3.1, Mixtral, or DeepSeek.
- [ ] **Explore advanced features** such as FP8 quantization, recipe caching, and expert parallelism.
- [ ] **Join the community** by contributing to the [vLLM-Gaudi GitHub repo](https://github.com/vllm-project/vllm-gaudi).

### Learn more

📚 [Intel Gaudi Documentation](https://docs.habana.ai/en/latest/index.html)  
📦 [vLLM Plugin System Overview](https://docs.vllm.ai/en/latest/design/plugin_system.html)
