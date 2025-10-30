---
title: Advanced Configuration Options
---

# Advanced Configuration Options

To align the setup to your specific needs, you can use optional advanced
configurations for running the vLLM server and benchmark. These configurations
let you fine-tune performance, memory usage, and request handling using
additional environment variables or configuration files. For most users, the
basic setup is sufficient, but advanced users may benefit from these
customizations.

## Running vLLM Using Docker Compose with Custom Parameters

This configuration allows you to override the default settings by providing additional environment variables when starting the server. This allows fine-tuning for performance and memory usage.

The following table lists the available variables:

| **Variable**                    | **Description**                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| `PT_HPU_LAZY_MODE`              | Enables Lazy execution mode, potentially improving performance by batching operations.    |
| `VLLM_SKIP_WARMUP`              | Skips the model warm-up phase to reduce startup time. It may affect initial latency.       |
| `MAX_MODEL_LEN`                 | Sets the maximum supported sequence length for the model.                                 |
| `MAX_NUM_SEQS`                  | Specifies the maximum number of sequences processed concurrently.                         |
| `TENSOR_PARALLEL_SIZE`          | Defines the degree of tensor parallelism.                                                 |
| `VLLM_EXPONENTIAL_BUCKETING`    | Enables or disables exponential bucketing for warm-up strategy.                            |
| `VLLM_DECODE_BLOCK_BUCKET_STEP` | Configures the step size for decode block allocation, affecting memory granularity.       |
| `VLLM_DECODE_BS_BUCKET_STEP`    | Sets the batch size step for decode operations, impacting how decode batches are grouped. |
| `VLLM_PROMPT_BS_BUCKET_STEP`    | Adjusts the batch size step for prompt processing.                                        |
| `VLLM_PROMPT_SEQ_BUCKET_STEP`   | Controls the step size for prompt sequence allocation.                                    |

Set the preferred variable when running the vLLM server using Docker Compose, as presented in the following example:

```bash
MODEL="Qwen/Qwen2.5-14B-Instruct" \
HF_TOKEN="<your huggingface token>" \
DOCKER_IMAGE="vault.habana.ai/gaudi-docker/|Version|/ubuntu24.04/habanalabs/vllm-installer-|PT_VERSION|:latest" \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=2048 \
docker compose up
```

## Running vLLM and Benchmark with Custom Parameters

This configuration allows you to customize benchmark behavior by setting additional environment variables before running Docker Compose.

The following table lists the available variables:

| **Variable**  | **Description**                                    |
| ------------- | -------------------------------------------------- |
| `INPUT_TOK`   | Number of input tokens per prompt.                 |
| `OUTPUT_TOK`  | Number of output tokens to generate per prompt.    |
| `CON_REQ`     | Number of concurrent requests during benchmarking. |
| `NUM_PROMPTS` | Total number of prompts to use in the benchmark.   |

Set the preferred variable when running the vLLM server using Docker Compose, as presented in the following example:

```bash
MODEL="Qwen/Qwen2.5-14B-Instruct" \
HF_TOKEN="<your huggingface token>" \
DOCKER_IMAGE="vault.habana.ai/gaudi-docker/|Version|/ubuntu24.04/habanalabs/vllm-installer-|PT_VERSION|:latest" \
INPUT_TOK=128 \
OUTPUT_TOK=128 \
CON_REQ=16 \
NUM_PROMPTS=64 \
docker compose --profile benchmark up
```

This launches the vLLM server and runs the benchmark using your specified parameters.

## Running vLLM and Benchmark with Combined Custom Parameters

This configuration allows you to launch the vLLM server and benchmark together. You can set any combination of server and benchmark-specific variables mentioned earlier. Set the preferred variable when running the vLLM server using Docker Compose, as presented in the following example:

```bash
MODEL="Qwen/Qwen2.5-14B-Instruct" \
HF_TOKEN="<your huggingface token>" \
DOCKER_IMAGE="vault.habana.ai/gaudi-docker/|Version|/ubuntu24.04/habanalabs/vllm-installer-|PT_VERSION|:latest" \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=2048 \
INPUT_TOK=128 \
OUTPUT_TOK=128 \
CON_REQ=16 \
NUM_PROMPTS=64 \
docker compose --profile benchmark up
```

This command starts the server and executes benchmarking with the provided configuration.

## Running vLLM and Benchmark Using Configuration Files

This configuration allows you to configure the server and benchmark via YAML configuration files.

The following table lists the available environment variables:

| **Variable**                 | **Description**                                             |
| ---------------------------- | ----------------------------------------------------------- |
| `VLLM_SERVER_CONFIG_FILE`    | Path to the server config file inside the Docker container. |
| `VLLM_SERVER_CONFIG_NAME`    | Name of the server config section.                          |
| `VLLM_BENCHMARK_CONFIG_FILE` | Path to the benchmark config file inside the container.     |
| `VLLM_BENCHMARK_CONFIG_NAME` | Name of the benchmark config section.                       |

Set the preferred variable when running the vLLM server using Docker Compose, as presented in the following example:

```bash
HF_TOKEN=<your huggingface token> \
VLLM_SERVER_CONFIG_FILE=server/server_scenarios_text.yaml \
VLLM_SERVER_CONFIG_NAME=llama31_8b_instruct \
VLLM_BENCHMARK_CONFIG_FILE=benchmark/benchmark_scenarios_text.yaml \
VLLM_BENCHMARK_CONFIG_NAME=llama31_8b_instruct \
docker compose --profile benchmark up
```

!!! note
    When using configuration files, you do not need to set the `MODEL` variable as the model details are included in the config files. However, the `HF_TOKEN` flag is still required.

## Running vLLM Directly Using Docker

For maximum control, you can run the server directly using the `docker run` command, allowing full customization of Docker runtime settings, as in the following example:

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
