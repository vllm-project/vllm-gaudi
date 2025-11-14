---
title: Frequently Asked Questions
---
[](){ #faq }

## Prerequisites and System Requirements

### What are the system requirements for running vLLM on Intel® Gaudi®?

- Ubuntu 22.04 LTS OS.
- Python 3.10.
- Intel Gaudi 2 or Intel Gaudi 3 AI accelerator.
- Intel Gaudi software version 1.23.0 and above.

### What is the vLLM plugin and where can I find its GitHub repository?

Intel develops and maintains its own vLLM plugin project called vLLM Hardware Plugin for Intel® Gaudi® and located in the [vLLM-gaudi](https://github.com/vllm-project/vllm-gaudi) repository on GitHub.

### How do I verify that the Intel® Gaudi® software is installed correctly?

1. Run ``hl-smi`` to check if Intel® Gaudi® accelerators are visible. Refer to [System Verifications and Final Tests](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html#system-verification) for more details.

2. Run ``apt list --installed | grep habana`` to verify installed packages. The output should look similar to the following example:

    ```text
    $ apt list --installed | grep habana
    habanalabs-container-runtime
    habanalabs-dkms
    habanalabs-firmware-tools
    habanalabs-graph
    habanalabs-qual
    habanalabs-rdma-core
    habanalabs-thunk
    habanalabs-tools
    ```

3. Check the installed Python packages by running ``pip list | grep habana`` and ``pip list | grep neural``. The output should look similar to this example:

    ```text
    $ pip list | grep habana
    habana_gpu_migration              1.19.0.561
    habana-media-loader               1.19.0.561
    habana-pyhlml                     1.19.0.561
    habana-torch-dataloader           1.19.0.561
    habana-torch-plugin               1.19.0.561
    lightning-habana                  1.6.0
    Pillow-SIMD                       9.5.0.post20+habana
    $ pip list | grep neural
    neural_compressor_pt              3.2
    ```

### How can I quickly set up the environment for vLLM using Docker?

Use the `Dockerfile.ubuntu.pytorch.vllm` file provided in the [.cd directory on GitHub](https://github.com/vllm-project/vllm-gaudi/tree/main/.cd) to build and run a container with the latest Intel® Gaudi® software release.

For more details, see [Quick Start Using Dockerfile](../getting_started/quickstart/quickstart.md).

## Building and Installing vLLM

### How can I install vLLM on Intel Gaudi?

There are two different installation methods:

- [Running vLLM Hardware Plugin for Intel® Gaudi® using a Dockerfile](../getting_started/installation.md#running-vllm-hardware-plugin-for-intel-gaudi-using-dockerfile): We recommend this method as it is the most suitable option for production deployments.

- [Building vLLM Hardware Plugin for Intel® Gaudi® from source](../getting_started/installation.md#building-vllm-hardware-plugin-for-intel-gaudi-from-source): This method is intended for developers working with experimental code or new features that are still under testing.

## Examples and Model Support

### Which models and configurations have been validated on Intel® Gaudi® 2 and Intel® Gaudi® 3 devices?

The list of validated models is available in the [Validated Models](../getting_started/validated_models.md) document. The list includes models such as:

- Llama 2, Llama 3, and Llama 3.1 (7B, 8B, and 70B versions). Refer to Llama-3.1 jupyter notebook example.

- Mistral and Mixtral models.

- Different tensor parallelism configurations , such as single HPU, 2x, and 8x HPU.

## Features Support

### Which key features does vLLM support on Intel® Gaudi®?

The list of the supported features is available in the [Supported Features](../features/supported_features.md) document. It includes features such as:

- Offline Batched Inference

- OpenAI-Compatible Server

- Paged KV cache optimized for Intel® Gaudi® devices

- Speculative decoding (experimental)

- Tensor parallel inference

- FP8 models and KV Cache quantization and calibration with Intel® Neural Compressor (INC). For more details, see the [Intel® Neural Compressor](../configuration/quantization/inc.md) quantization and inference guide.

## Performance Tuning

### Which execution modes does the plugin support?

- PyTorch Eager mode (default)

- torch.compile (default)

- HPU Graphs (recommended for best performance)

- PyTorch Lazy mode

### How does the bucketing mechanism work in vLLM Hardware Plugin for Intel® Gaudi®?

The bucketing mechanism optimizes performance by grouping tensor shapes. This reduces the number of required graphs and minimizes compilations during server runtime. Buckets are determined by parameters for batch size and sequence length. For more information, see [Bucketing Mechanism](../features/bucketing_mechanism.md).

### What should I do if a request exceeds the maximum bucket size?

Consider increasing the upper bucket boundaries using environment variables to avoid potential latency increases due to graph compilation.
