<p align="center">
  <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png" alt="vLLM" width="30%">
  <span style="font-size: 24px; font-weight: bold;">x</span>
  <img src="./docs/assets/logos/gaudi-logo.png" alt="Intel-Gaudi" width="30%">
</p>

<h2 align="center">
vLLM Hardware Plugin for Intel® Gaudi®
</h2>

<p align="center">
| <a href="https://vllm-gaudi.readthedocs.io/en/latest/index.html"><b>Documentation</b></a> | <a href="https://docs.habana.ai/en/latest/index.html"><b>Intel® Gaudi® Documentation</b></a> | <a href="https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html"><b>Optimizing Training Platform Guide</b></a> |
</p>

---
*Latest News* 🔥

- [2025/11] The 0.10.2 release introduces the production-ready version of the vLLM Hardware Plugin for Intel® Gaudi® v1.23.0. The plugin is an alternative to the [vLLM fork](https://github.com/HabanaAI/vllm-fork), which reaches end of life with this release and will be deprecated in v1.24.0, remaining functional only for legacy use cases. We strongly encourage all fork users to begin planning their migration to the plugin. For more information about this release, see the [Release Notes](docs/release_notes.md).
- [2025/06] We introduced an early developer preview of the vLLM Hardware Plugin for Intel® Gaudi®, which is not yet intended for general use.

---

## About

The vLLM Hardware Plugin for Intel® Gaudi® integrates [Intel® Gaudi® AI accelerators](https://docs.habana.ai/en/latest/index.html) with [vLLM](https://docs.vllm.ai/en/latest/) to optimize large language model inference. It follows the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162) and [[RFC]: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161) principles, providing a modular interface for Intel® Gaudi® hardware. For more information, see the [Plugin System](https://vllm-gaudi.readthedocs.io/en/latest/dev_guide/plugin_system.html) document.

## Getting Started

1. [Set up](https://docs.habana.ai/en/latest/Installation_Guide/index.html) your execution environment. Additionally, to achieve the best performance on HPU, follow the methods outlined in the [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).

2. Get the last verified vLLM commit. While vLLM Hardware Plugin for Intel® Gaudi® follows the latest vLLM commits, upstream API updates may introduce compatibility issues. The saved commit has been thoroughly validated.

    ```bash
    git clone https://github.com/vllm-project/vllm-gaudi
    cd vllm-gaudi
    export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
    cd ..
    ```

3. Install vLLM using `pip` or [build it from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

    ```bash
    # Build vLLM from source for empty platform, reusing existing torch installation
    git clone https://github.com/vllm-project/vllm
    cd vllm
    git checkout $VLLM_COMMIT_HASH
    pip install -r <(sed '/^torch/d' requirements/build.txt)
    VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
    cd ..
    ```

4. Install vLLM Hardware Plugin for Intel® Gaudi® from source:

    ```bash
    cd vllm-gaudi
    pip install -e .
    cd ..
    ```

    To see all the available installation methods, such as NIXL, see the [Installation](https://vllm-gaudi.readthedocs.io/en/latest/getting_started/installation.html) guide.

## Contributing

We welcome and value any contributions and collaborations.

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm-gaudi/issues).
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai).
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai).
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature.
<!-- --8<-- [end:contact-us] -->
