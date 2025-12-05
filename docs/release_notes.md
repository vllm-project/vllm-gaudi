# Release Notes

This document provides an overview of the features, changes, and fixes introduced in each release of the vLLM Hardware Plugin for Intel® Gaudi®.

## 0.12.0

This release upgraded the plugin to [vLLM 0.12.0](https://github.com/vllm-project/vllm/releases/tag/v0.12.0) and added support for [Intel® Gaudi® v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html).

## 0.11.2

This version is based on [vLLM 0.11.2](https://github.com/vllm-project/vllm/releases/tag/v0.11.2) and supports [Intel® Gaudi® v1.22.2](https://docs.habana.ai/en/v1.22.2/Release_Notes/GAUDI_Release_Notes.html).

This release introduces the production-ready vLLM Hardware Plugin for Intel® Gaudi®, a community-driven integration layer based on the [vLLM v1 architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html). It enables efficient, high-performance large language model (LLM) inference on [Intel® Gaudi®](https://docs.habana.ai/) AI accelerators. The plugin is an alternative to the [vLLM fork](https://github.com/HabanaAI/vllm-fork), which reaches end of life with this release and will be deprecated in v1.24.0, remaining functional only for legacy use cases. We strongly encourage all fork users to begin planning their migration to the plugin.

The plugin provides [feature parity](features/supported_features.md) with the fork, including mature, production-ready implementations of Automatic Prefix Caching (APC) and async scheduler. Two legacy features - multi-step scheduling and delayed sampling - have been discontinued, as their functionality is now covered by the async scheduler.

For more details on the plugin's implementation, see [Plugin System](dev_guide/plugin_system.md).

To start using the plugin, follow the [Basic Quick Start Guide](getting_started/quickstart/quickstart.md) and explore the rest of this documentation.
