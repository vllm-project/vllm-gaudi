# Calibrating a Single Node

This procedure explains how to perform calibration for a single Intel® Gaudi® node.

## Prerequisites

Before you start, familiarize with [notes and recommendations](calibration.md#notes-and-recommendations).

## Calibration procedure

1. Build and install the latest version of vLLM Hardware Plugin for Intel® Gaudi® by following the [Installation](../../getting_started/installation.md) procedure.

2. In the plugin project directory, navigate to the `calibration` subdirectory and install the required dependencies.

    ```bash
    cd calibration
    pip install -r requirements.txt
    ```

3. Download the dataset.

4. Run the `calibrate_model.sh` script with the obligatory `-m`, `-d`, and `-o` arguments, as in the following example:

    ```bash
    ./calibrate_model.sh -m /path/to/local/llama3.1/Meta-Llama-3.1-405B-Instruct/ -d dataset-processed.pkl -o /path/to/measurements/vllm-benchmarks/inc -b 128 -t 8 -l 4096
    # OR
    ./calibrate_model.sh -m facebook/opt-125m -d dataset-processed.pkl -o inc/
    # OR Calibrate DeepSeek models with dataset NeelNanda/pile-10k
    PT_HPU_LAZY_MODE=1 ./calibrate_model.sh -m deepseek-ai/DeepSeek-R1  -d NeelNanda/pile-10k -o inc/ -t 8
    ```

    Where:

   - `-m <path/ID>`: Path to a locally stored model or the model ID from the [Hugging Face](https://huggingface.co/models) hub.
   - `-d <path>`: Path to the source dataset in the pickle (`.pkl`) format.
   - `-o <path>`: Path to the directory where you want to save the generated measurements.
