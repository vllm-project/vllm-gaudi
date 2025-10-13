---
title: Quickstart
---

## vLLM Quick Start Guide

This guide shows how to quickly launch vLLM on Gaudi using a prebuilt Docker
image with Docker Compose which is supported on Ubuntu only. It supports model benchmarking, custom runtime parameters,
and a selection of validated models — including the LLama, Mistral, and Qwen.
The advanced configuration is available via environment variables or YAML files.

## Requirements

- Python 3.10
- Intel Gaudi 2 or 3 AI accelerators
- Intel Gaudi software version 1.21.0 or above

!!! note
    To set up the execution environment, please follow the instructions in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html).
    To achieve the best performance on HPU, please follow the methods outlined in the
    [Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).

## Running vLLM on Gaudi with Docker Compose

Follow the steps below to run the vLLM server or launch benchmarks on Gaudi using Docker Compose.

### 1. Clone the vLLM fork repository and navigate to the appropriate directory

    git clone https://github.com/HabanaAI/vllm-fork.git
    cd vllm-fork/.cd/

This ensures you have the required files and Docker Compose configurations.

### 2. Set the following environment variables

| **Variable** | **Description** |
| --- |--- |
| `MODEL` | Choose a model name from the [`vllm supported models`][supported-models] list.  |
| `HF_TOKEN` | Your Hugging Face token (generate one at <https://huggingface.co>). |
| `DOCKER_IMAGE` | The Docker image name or URL for the vLLM Gaudi container. When using the Gaudi repository, make sure to select Docker images with the *vllm-installer* prefix in the file name. |

### 3. Run the vLLM server using Docker Compose

    MODEL="Qwen/Qwen2.5-14B-Instruct" \
    HF_TOKEN="<your huggingface token>" \
    DOCKER_IMAGE="vault.habana.ai/gaudi-docker/|Version|/ubuntu22.04/habanalabs/vllm-installer-|PT_VERSION|:latest" \
    docker compose up

To automatically run benchmarking for a selected model using default settings, add the  `--profile benchmark up` option

    MODEL="Qwen/Qwen2.5-14B-Instruct" \
    HF_TOKEN="<your huggingface token>" \
    DOCKER_IMAGE=="vault.habana.ai/gaudi-docker/|Version|/ubuntu22.04/habanalabs/vllm-installer-|PT_VERSION|:latest" \
    docker compose --profile benchmark up

This command launches the vLLM server and runs the associated benchmark suite.

## Advanced Options

The following steps cover optional advanced configurations for
running the vLLM server and benchmark. These allow you to fine-tune performance,
memory usage, and request handling using additional environment variables or configuration files.
For most users, the basic setup is sufficient, but advanced users may benefit from these customizations.

=== "Run vLLM Using Docker Compose with Custom Parameters"

    To override default settings, you can provide additional environment variables when starting the server. This advanced method allows fine-tuning for performance and memory usage.

    **Environment variables**

    | **Variable** | **Description** |
    |---|---|
    |  `PT_HPU_LAZY_MODE`              | Enables Lazy execution mode, potentially improving performance by batching operations. |
    |  `VLLM_SKIP_WARMUP`              | Skips the model warmup phase to reduce startup time (may affect initial latency).                     |
    |  `MAX_MODEL_LEN`                 | Sets the maximum supported sequence length for the model.               |    
    |  `MAX_NUM_SEQS`                  | Specifies the maximum number of sequences processed concurrently.       |
    |  `TENSOR_PARALLEL_SIZE`          | Defines the degree of tensor parallelism.                               |
    |  `VLLM_EXPONENTIAL_BUCKETING`    | Enables or disables exponential bucketing for warmup strategy.          |
    |  `VLLM_DECODE_BLOCK_BUCKET_STEP` | Configures the step size for decode block allocation, affecting memory granularity.         |
    |  `VLLM_DECODE_BS_BUCKET_STEP`    | Sets the batch size step for decode operations, impacting how decode batches are grouped.             |
    |  `VLLM_PROMPT_BS_BUCKET_STEP`    | Adjusts the batch size step for prompt processing.                      |
    |  `VLLM_PROMPT_SEQ_BUCKET_STEP`   | Controls the step size for prompt sequence allocation.                  |

    **Example**

    ```bash
    MODEL="Qwen/Qwen2.5-14B-Instruct" \
    HF_TOKEN="<your huggingface token>" \
    DOCKER_IMAGE="vault.habana.ai/gaudi-docker/|Version|/ubuntu22.04/habanalabs/vllm-installer-|PT_VERSION|:latest" \
    TENSOR_PARALLEL_SIZE=1 \
    MAX_MODEL_LEN=2048 \
    docker compose up
    ```

=== "Run vLLM and Benchmark with Custom Parameters"

    You can customize benchmark behavior by setting additional environment variables before running Docker Compose.

    **Benchmark parameters:**

    | **Variable** | **Description** |
    |---|---|
    |  `INPUT_TOK`  | Number of input tokens per prompt.                           |
    |  `OUTPUT_TOK` | Number of output tokens to generate per prompt.              |
    |  `CON_REQ`    | Number of concurrent requests during benchmarking.           |
    |  `NUM_PROMPTS`| Total number of prompts to use in the benchmark.             |

    **Example:**

    ```bash
    MODEL="Qwen/Qwen2.5-14B-Instruct" \
    HF_TOKEN="<your huggingface token>" \
    DOCKER_IMAGE="vault.habana.ai/gaudi-docker/|Version|/ubuntu22.04/habanalabs/vllm-installer-|PT_VERSION|:latest" \
    INPUT_TOK=128 \
    OUTPUT_TOK=128 \
    CON_REQ=16 \
    NUM_PROMPTS=64 \
    docker compose --profile benchmark up
    ```

    This launches the vLLM server and runs the benchmark using your specified parameters.

=== "Run vLLM and Benchmark with Combined Custom Parameters"

    You can launch the vLLM server and benchmark together, providing any combination of server and benchmark-specific parameters.

    **Example:**

    ```bash
    MODEL="Qwen/Qwen2.5-14B-Instruct" \
    HF_TOKEN="<your huggingface token>" \
    DOCKER_IMAGE="vault.habana.ai/gaudi-docker/|Version|/ubuntu22.04/habanalabs/vllm-installer-|PT_VERSION|:latest" \
    TENSOR_PARALLEL_SIZE=1 \
    MAX_MODEL_LEN=2048 \
    INPUT_TOK=128 \
    OUTPUT_TOK=128 \
    CON_REQ=16 \
    NUM_PROMPTS=64 \
    docker compose --profile benchmark up
    ```

    This command starts the server and executes benchmarking with the provided configuration.

=== "Run vLLM and Benchmark Using Configuration Files"

    You can also configure the server and benchmark via YAML configuration files. Set the following environment variables:

    | **Variable** | **Description** |
    |---|---|
    |  `VLLM_SERVER_CONFIG_FILE`          | Path to the server config file inside the Docker container. |
    |  `VLLM_SERVER_CONFIG_NAME`          | Name of the server config section.                          |
    |  `VLLM_BENCHMARK_CONFIG_FILE`       | Path to the benchmark config file inside the container.     |
    |  `VLLM_BENCHMARK_CONFIG_NAME`       | Name of the benchmark config section.                       |

    **Example**

    ```bash
    HF_TOKEN=<your huggingface token> \
    VLLM_SERVER_CONFIG_FILE=server_configurations/server_text.yaml \
    VLLM_SERVER_CONFIG_NAME=llama31_8b_instruct \
    VLLM_BENCHMARK_CONFIG_FILE=benchmark_configurations/benchmark_text.yaml \
    VLLM_BENCHMARK_CONFIG_NAME=llama31_8b_instruct \
    docker compose --profile benchmark up
    ```

    !!! note
        When using configuration files, you do not need to set the  `MODEL` variable as the model details are included in the config files. However, the  `HF_TOKEN` flag is still required.

=== "Run vLLM Directly Using Docker"

    For maximum control, you can run the server directly using the  `docker run` command, allowing full customization of Docker runtime settings.

    **Example:**

    ```bash
    docker run -it --rm \
        -e MODEL=$MODEL \
        -e HF_TOKEN=$HF_TOKEN \
        -e http_proxy=$http_proxy \
        -e https_proxy=$https_proxy \
        -e no_proxy=$no_proxy \
        --cap-add=sys_nice \
        --ipc=host \
        --runtime=habana \
        -e HABANA_VISIBLE_DEVICES=all \
        -p 8000:8000 \
        --name vllm-server \
        <docker image name>
    ```

    This method provides full flexibility over how the vLLM server is executed within the container.

---

## Supported Models

| **Model Name**                                                | **Validated TP Size**  |
|---|---|
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B                 | 8                  |
| meta-llama/Llama-3.1-70B-Instruct                         | 4                  |
| meta-llama/Llama-3.1-405B-Instruct                        | 8                  |
| meta-llama/Llama-3.1-8B-Instruct                          | 1                  |
| meta-llama/Llama-3.3-70B-Instruct                         | 4                  |
| mistralai/Mistral-7B-Instruct-v0.2                        | 1                  |
| mistralai/Mixtral-8x7B-Instruct-v0.1                      | 2                  |
| mistralai/Mixtral-8x22B-Instruct-v0.1                     | 4                  |
| Qwen/Qwen2.5-7B-Instruct                                  | 1                  |
| Qwen/Qwen2.5-VL-7B-Instruct                               | 1                  |
| Qwen/Qwen2.5-14B-Instruct                                 | 1                  |
| Qwen/Qwen2.5-32B-Instruct                                 | 1                  |
| Qwen/Qwen2.5-72B-Instruct                                 | 4                  |
| ibm-granite/granite-8b-code-instruct-4k                   | 1                  |
| ibm-granite/granite-20b-code-instruct-8k                  | 1                  |

## Executing inference

=== "Offline Batched Inference"

    [](){ #quickstart-offline }

    ```python
    from vllm import LLM, SamplingParams

    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="facebook/opt-125m")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

=== "OpenAI Completions API"

    WIP

=== "OpenAI Chat Completions API with vLLM"

    WIP
