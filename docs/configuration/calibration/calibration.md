# Introduction

vLLM Hardware Plugin for Intel® Gaudi® supports running inference on HPU with 8-bit floating point (FP8) precision using [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package. Inference requires prior calibration to generate the necessary measurements, quantization files, and configuration data that are required for running quantized models.

This document explains how to perform calibration. It provides separate procedures for a single Intel® Gaudi® node and multiple nodes. Before proceeding, review the notes and recommendations and troubleshooting information to ensure proper execution.

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

## Calibration Procedures

Refer the following chapters to follow the calibration procedure for your setup:

- [Simple calibration for a single Intel® Gaudi® node](calibration_one_node.md)
- [Calibration for multiple Intel® Gaudi® nodes](calibration_multi_node.md)

## Troubleshooting

If you encounter the following error when running the script, ensure you set a valid tensor parallelism value, for example `-t 8`:

```
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed for size::939524096 (896)MB
```
