---
title: vLLM with Intel Gaudi Frequently Asked Questions
---
[](){ #faq }

## Prerequisites and System Requirements

### What are the system requirements for running vLLM on Intel® Gaudi®?

- Ubuntu 22.04 LTS OS.
- Python 3.10.
- Intel Gaudi 2 or Intel Gaudi 3 AI accelerator.
- Intel Gaudi software version 1.23.0 and above.

### What is vLLM plugin and where can I find this GitHub repository?

- Intel develops and maintains its own vLLM plugin project called [vLLM-gaudi](https://github.com/vllm-project/vllm-gaudi).

### How do I verify that the Intel Gaudi software is installed correctly?

- Run ``hl-smi`` to check if Gaudi accelerators are visible. Refer to [System Verifications and Final Tests](https://docs.habana.ai/en/latest/Installation_Guide/System_Verification_and_Final_Tests.html#system-verification) for more details.

- Run ``apt list --installed | grep habana`` to verify installed packages. The output should look similar to the below:

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

- Check the installed Python packages by running ``pip list | grep habana`` and ``pip list | grep neural``. The output should look similar to the below:

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

Use the Dockerfile.ubuntu.pytorch.vllm file provided in the vllm-plugin/vllm-gaudi/.cd GitHub repo to build and run a container with the latest Intel Gaudi software release.

For more details, see [Quick Start Using Dockerfile](../getting_started/quickstart.md).

## Building and Installing vLLM

### How can I install vLLM on Intel Gaudi?

- There are two different installation methods:

- (Recommended) Running vLLM Hardware Plugin for Intel® Gaudi® Using Dockerfile. This version is most suitable for production deployments.

- Building vLLM Hardware Plugin for Intel® Gaudi® from Source. This version is suitable for developers who would like to work on experimental code and new features that are still being tested.

## Examples and Model Support

### Which models and configurations have been validated on Gaudi 2 and Gaudi 3 devices?

- Various Llama 2, Llama 3 and Llama 3.1 models (7B, 8B and 70B versions). Refer to Llama-3.1 jupyter notebook example.

- Mistral and Mixtral models.

- Different tensor parallelism configurations (single HPU, 2x, and 8x HPU).

- See [Validated Models](../models/validated_models.md) for more details.

## Features and Support

### Which key features does vLLM support on Intel Gaudi?

- Offline Batched Inference.

- OpenAI-Compatible Server.

- Paged KV cache optimized for Gaudi devices.

- Speculative decoding (experimental).

- Tensor parallel inference.

- FP8 models and KV Cache quantization and calibration with Intel® Neural Compressor (INC). See [FP8 Calibration and Inference with vLLM](../features/quantization/inc.md) for more details.

- See [Supported Features](../features/supported_features.md) for more details.

## Performance Tuning

### Which execution modes does vLLM support on Intel Gaudi?

- PyTorch Eager mode (default).

- torch.compile (default).

- HPU Graphs (recommended for best performance).

- PyTorch Lazy mode.

- See [Execution Modes]() for more details.

### How does the bucketing mechanism work in vLLM for Intel Gaudi?

- The bucketing mechanism optimizes performance by grouping tensor shapes. This reduces the number of required graphs and minimizes compilations during server runtime.

- Buckets are determined by parameters for batch size and sequence length.

- See [Bucketing Mechanism](../features/bucketing_mechanism.md) for more details.

### What should I do if a request exceeds the maximum bucket size?

- Consider increasing the upper bucket boundaries using environment variables to avoid potential latency increases due to graph compilation.

## Troubleshooting

### How to troubleshoot Out-of-Memory errors encountered while running vLLM on Intel Gaudi?

- Increase ``--gpu-memory-utilization`` (default: 0.9) - This addresses insufficient available memory per card.

- Increase ``--tensor-parallel-size`` (default: 1) - This approach shards model weights across the devices and may help in loading a model (which is too big for a single card) across multiple cards.

- Disable HPU Graphs completely (switch to any other execution mode) to maximize KV Cache space allocation.
