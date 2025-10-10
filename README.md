<p align="center">
  <img src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png" alt="vLLM" width="30%">
  <span style="font-size: 24px; font-weight: bold;">x</span>
  <img src="./docs/assets/logos/gaudi-logo.png" alt="Intel-Gaudi" width="30%">
</p>

<h2 align="center">
Welcome to vLLM x Intel Gaudi
</h2>

<p align="center">
| <a href="https://vllm-gaudi.readthedocs.io/en/latest/index.html"><b>Documentation</b></a> | <a href="https://docs.habana.ai/en/latest/index.html"><b>Intel® Gaudi® Documentation</b></a> | <a href="https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html"><b>Optimizing Training Platform Guide</b></a> |
</p>

---
*Latest News* 🔥

- [2025/06] We are introduced an early developer preview of the vLLM Gaudi Plugin and is not yet intended for general use. For a more stable experience, consider using the [HabanaAI/vllm-fork](https://github.com/HabanaAI/vllm-fork) or the in-tree Gaudi implementation available in [vllm-project/vllm](https://github.com/vllm-project/vllm).

---

## About
vLLM Gaudi plugin (vllm-gaudi) integrates Intel Gaudi accelerators with vLLM to optimize large language model inference.

This plugin follows the [[RFC]: Hardware pluggable](https://github.com/vllm-project/vllm/issues/11162) and [[RFC]: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161) principles, providing a modular interface for Intel Gaudi hardware.

Learn more: 🚀 [vLLM Plugin System Overview](https://docs.vllm.ai/en/latest/design/plugin_system.html)

## Getting Started
0. Preparation of the Setup

    To set up the execution environment, please follow the instructions in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).
    To achieve the best performance on HPU, please follow the methods outlined in the
    [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).

1. Get Last good commit on vllm
   NOTE: vllm-gaudi is always follow latest vllm commit, however, vllm upstream
   API update may crash vllm-gaudi, this commit saved is verified with vllm-gaudi
   in a hourly basis

    ```bash
    git clone https://github.com/vllm-project/vllm-gaudi
    cd vllm-gaudi
    export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)
    ```

2. Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

    ```bash
    # Build vLLM from source for empty platform, reusing existing torch installation
    git clone https://github.com/vllm-project/vllm
    cd vllm
    git checkout $VLLM_COMMIT_HASH
    pip install -r <(sed '/^[torch]/d' requirements/build.txt)
    VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
    cd ..
    ```

3. Install vLLM-Gaudi from source:

    ```bash
    cd vllm-gaudi
    pip install -e .
    cd ..
    ```

4. To uncover all installation methods, sucha as NixL, follow the  [link](https://vllm-gaudi.readthedocs.io/en/latest/getting_started/installation.html)

## Contributing

We welcome and value any contributions and collaborations.

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm-gaudi/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
<!-- --8<-- [end:contact-us] -->
