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
- [2026/03] Version 0.16.0 is now available, built on [vLLM 0.16.0](https://github.com/vllm-project/vllm/releases/tag/v0.16.0) and fully compatible with [Intel® Gaudi® v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html).

  This release introduces validated support and critical stability fixes for Qwen3-VL models leveraging HPUMMEncoderAttention. Performance and stability were improved through backported Mamba architecture optimizations, Docker and UBI infrastructure enhancements, and a forced CPU loading mechanism for INC quantization to prevent OOM errors.

- [2026/02] Version 0.15.1 is now available, built on [vLLM 0.15.1](https://github.com/vllm-project/vllm/releases/tag/v0.15.1) and fully compatible with [Intel® Gaudi® v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html).

  This release introduces validated support for Granite 4.0-h and Qwen3-VL (dense and MoE variants) on Intel Gaudi 3, alongside significant Llama 4 stability fixes. It also features major prefill performance improvements via full chunked prefill attention, FlashAttention online merge, b2b matmul operations, and KV cache sharing. Additionally, this version adds HPU ops for Mamba/SSM architectures to enable hybrid models, and introduces new support for ModelOpt FP8 quantization.

- [2026/02] Version 0.14.1 is now available, built on [vLLM 0.14.1](https://github.com/vllm-project/vllm/releases/tag/v0.14.1) and fully compatible with [Intel® Gaudi® v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html). It introduces support for Granite 4.0h and Qwen 3 VL models.

- [2026/01] Version 0.13.0 is now available, built on [vLLM 0.13.0](https://github.com/vllm-project/vllm/releases/tag/v0.13.0) and fully compatible with [Intel® Gaudi® v1.23.0](https://docs.habana.ai/en/v1.23.0/Release_Notes/GAUDI_Release_Notes.html). It introduces experimental dynamic quantization for MatMul and KV‑cache operations to improve performance and also supports additional models.

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

5. Install torchaudio (required by some upstream vLLM models such as QWEN3_5). Use the CPU wheel with `--no-deps` to avoid pulling a conflicting CUDA torch:

    ```bash
    pip install --no-deps torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
    ```

    To see all the available installation methods, such as NIXL, see the [Installation](https://vllm-gaudi.readthedocs.io/en/latest/getting_started/installation.html) guide.

## Contributing

We welcome and value any contributions and collaborations.

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm-gaudi/issues).
<!-- --8<-- [end:contact-us] -->
