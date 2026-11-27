# Calibrating a Single Node

!!! warning
    `calibrate_model.sh` is deprecated and will be removed in a future release. Use `vllm-gaudi-calibrate` instead: see the [Calibration Quick Start](quickstart.md), and [Migration](migration.md) for the option mapping. This page documents the deprecated script until then.

This procedure explains how to perform calibration for a single Intel® Gaudi® node.

## Prerequisites

Before you start, familiarize with [notes and recommendations](#notes-and-recommendations).

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
    ./calibrate_model.sh -m deepseek-ai/DeepSeek-R1  -d NeelNanda/pile-10k -o inc/ -t 8
    ```

    Where:

   - `-m <path/ID>`: Path to a locally stored model or the model ID from the [Hugging Face](https://huggingface.co/models) hub.
   - `-d <path>`: Path to the source dataset in the pickle (`.pkl`) format.
   - `-o <path>`: Path to the directory where you want to save the generated measurements.

## Notes and Recommendations

### Device Recommendation

For calibration, use the same device type that you plan to use for inference. The generated measurements are device-dependent, so scales collected on Intel® Gaudi® 3 cannot be reused on Intel® Gaudi® 2, and vice versa. Using measurements generated on a different device type may cause accuracy issues.

### Mandatory Parameters

To simplify the calibration process, we offer the `calibrate_model.sh` script that generates the `maxabs_quant_g3.json` file for FP8 inference. The script requires providing the following arguments:

- `-m <path/ID>`: Path to a locally stored model or the model ID from the [Hugging Face](https://huggingface.co/models) hub.
- `-d <path>`: Path to the source dataset in the pickle (`.pkl`) format.
- `-o <path>`: Path to the directory where you want to save the generated measurements. We recommend storing unification results in the source directory. This allows you to run the vLLM server with FP8 precision and different tensor parallelism values without modifying the directory specified in the `QUANT_CONFIG` environment variable.

The script also offers optional arguments that you can explore by executing the script with the `-h` flag. The more common optional parameters are:

- `-b <size>`: Sets the batch size used for running the measurements (default: 32).
- `-l <samples>`: Sets the limit of the samples in the calibration dataset.
- `-t <size>`: Sets the tensor parallel size (default: 1).

### Dataset

The calibration procedure works with any dataset that contains the `system_prompt` and `question` fields. These fields prepare a calibration dataset with prompts formatted specifically for your model. We recommend using a public dataset from MLCommons, as used in the [Llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b#preprocessed) inference submission.

### DeepSeek Models

For the [DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) series models, which contain 256 experts, provide a diverse and sufficiently large sample set to ensure that all experts are properly activated during calibration. Through testing, we observed that using [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) and selecting 512 samples, each with at least 1,024 tokens, provides effective calibration coverage.

> **Important (MoE models calibrated with expert parallelism).** When you calibrate a Mixture-of-Experts model with `-u` (expert parallelism) on more than one card, each rank only measures its **EP-local** experts (for example, 16 of 128 experts per rank for Llama-4 Maverick on TP8/EP8). These per-rank measurements must be unified so that every expert has its own scale, otherwise the non-local experts fall back to coarse quantization at serve time and FP8 accuracy drops. The unify step runs only when a target rank is given, so pass **both `-u` and `-r`** to `calibrate_model.sh` (for example `-r 1` to unify onto a single card), then use `step-6-expand-measurements.py` to expand to the deployment card count. **Passing `-u` with `TP>1` but without `-r` skips unification** and leaves only the EP-local experts measured. (Small-expert models such as Mixtral are not affected.) See the [MoE recommendations](calibration_multi_node.md#recommendations-for-advanced-usage-for-moe-models) for the full unify-and-expand procedure.

### Configuration

The `calibrate_model.sh` script automatically generates appropriate configuration files for the calibration process. However, if you require advanced customization, you can use JSON configuration files. The `quantization_config` directory contains JSON templates that you can use directly or modify to suit your specific requirements.

To apply custom configurations for calibration, add the `QUANT_CONFIG` environment variable pointing to your configuration JSON file to the `step-2-measure-scales.py` and `step-4-quantize-scales.py` calibration steps. To apply the configuration when running FP8 inference, set `QUANT_CONFIG` to point to the quantization configuration file, either the one generated by calibration or your custom configuration.

For the configuration options, backoff factors, and the `scale_format` optimization, see [INC Configuration](reference.md#inc-configuration).

### Troubleshooting

If you encounter the following error when running the script, ensure you set a valid tensor parallelism value, for example `-t 8`:

```
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed for size::939524096 (896)MB
```
