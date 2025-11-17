---
title: Basic Quick Start Guide
---

# Basic Quick Start Guide

The 1.22 release of vLLM Hardware Plugin for Intel® Gaudi® introduces ready-to-run container
images that package vLLM together with the Intel® Gaudi® software. This release
enables a fast and simple launch of vLLM Hardware Plugin for Intel® Gaudi® using prebuilt Docker
images and Docker Compose, with support for custom runtime parameters and
benchmarking.

This guide explains the easiest way of running vLLM Hardware Plugin for Intel® Gaudi® on Ubuntu.
It includes features for model benchmarking, runtime customization, and
selecting validated models such as LLaMA, Mistral, and Qwen. Advanced
configuration can be performed through environment variables or YAML
configuration files.

If you prefer to build vLLM Hardware Plugin for Intel® Gaudi® from source or with a custom
Dockerfile, refer to the [Installation](../installation.md) guide.

## Requirements

Before you start, ensure that your environment meets the following requirements:

- Ubuntu 22.04 or 24.04
- Python 3.10
- Intel® Gaudi® 2 or 3 AI accelerator
- Intel® Gaudi® software version 1.21.0 or later

Additionally, ensure that the Intel® Gaudi® execution environment is properly set up. If
it is not, complete the setup by following the [Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) instructions.

## Running vLLM Hardware Plugin for Intel® Gaudi® with Docker Compose

Follow these steps to run the vLLM server or launch benchmarks on Gaudi using Docker Compose.

1. Clone the vLLM plugin repository to get the required files and Docker Compose configurations.

    ```bash
    git clone https://github.com/vllm-project/vllm-gaudi.git
    ```

2. Navigate to the appropriate directory.

    ```bash
    cd vllm-gaudi/.cd/
    ```

3. Select your preferred values of the following variables.

    | **Variable**   | **Description**                                                                                                                                                              |
    | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `MODEL`        | Preferred large language model. For a list of the available models, see the next table.                                                                                      |
    | `HF_TOKEN`     | Hugging Face token generated from <https://huggingface.co>.                                                                                                                  |
    | `DOCKER_IMAGE` | Docker image name or URL for the vLLM Gaudi container. When using the Gaudi repository, make sure to select Docker images with the *vllm-installer* prefix in the file name. |

    The following table lists the supported vLLM models:

    | **Model Name**                            | **Validated TP Size** |
    | ----------------------------------------- | --------------------- |
    | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 8                     |
    | meta-llama/Llama-3.1-70B-Instruct         | 4                     |
    | meta-llama/Llama-3.1-405B-Instruct        | 8                     |
    | meta-llama/Llama-3.1-8B-Instruct          | 1                     |
    | meta-llama/Llama-3.3-70B-Instruct         | 4                     |
    | mistralai/Mistral-7B-Instruct-v0.2        | 1                     |
    | mistralai/Mixtral-8x7B-Instruct-v0.1      | 2                     |
    | mistralai/Mixtral-8x22B-Instruct-v0.1     | 4                     |
    | Qwen/Qwen2.5-7B-Instruct                  | 1                     |
    | Qwen/Qwen2.5-VL-7B-Instruct               | 1                     |
    | Qwen/Qwen2.5-14B-Instruct                 | 1                     |
    | Qwen/Qwen2.5-32B-Instruct                 | 1                     |
    | Qwen/Qwen2.5-72B-Instruct                 | 4                     |
    | ibm-granite/granite-8b-code-instruct-4k   | 1                     |
    | ibm-granite/granite-20b-code-instruct-8k  | 1                     |

4. Set the selected environment variables using the following example as a reference.

    ```bash
    MODEL="Qwen/Qwen2.5-14B-Instruct" \
    HF_TOKEN="<your huggingface token>" \
    DOCKER_IMAGE="vault.habana.ai/gaudi-docker/|Version|/ubuntu24.04/habanalabs/vllm-installer-|PT_VERSION|:latest"
    ```

5. Run the vLLM server using Docker Compose.

    ```bash
    docker compose up
    ```

    To automatically run benchmarking for a selected model using default settings, add the  `--profile benchmark up` option.

    ```bash
    docker compose --profile benchmark up
    ```

After completing this step, the vLLM server will be running, and the associated benchmark suite will start automatically. Optionally, to align the setup to your specific needs, you can use [advanced configuration options](quickstart_configuration.md). For most users, the basic setup is sufficient, but advanced users may benefit from additional customizations.

After setting up and running vLLM Hardware Plugin for Intel® Gaudi®, you can begin performing inference to generate model outputs. For detailed instructions, see the [Executing Inference](quickstart_inference.md) guide.

To achieve the best performance on HPU, follow the methods outlined in the
[Optimizing Training Platform
Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
