---
title: Intel® Neural Compressor
---
[](){ #inc }

# Intel® Neural Compressor

vLLM Hardware Plugin for Intel® Gaudi® supports 8-bit floating point (FP8) weight and activation quantization using Intel® Neural Compressor on Intel® Gaudi® 2 and Intel® Gaudi® 3 AI accelerators. Intel® Gaudi® supports quantization of various modules and functions, including `Linear`, `KVCache`, `Matmul`, and `Softmax`. For more information, see [Custom Patched Modules](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#custom-patched-modules). Currently, quantization is validated only on Llama models.

This guide describes the steps for quantization and inference. Because inference depends on prior calibration, the process begins with calibrating your model. Calibration generates the necessary measurements, quantization files, and configuration data that are required for running quantized models. These calibration outputs are essential as they provide model-specific metrics and define the quantization parameters applied during inference. To calibrate your model, follow the [Calibration](../calibration/calibration.md) guide. Once calibration is complete, you can proceed with offline or online inference described in this document.

For an end-to-end example tutorial for quantizing a BF16 Llama 3.1 model to FP8 and then inferencing, see this [this guide](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/vLLM_Tutorials/FP8_Quantization_using_INC/FP8_Quantization_using_INC.ipynb).

## Recommendations

- Before running inference, ensure your measurements were generated on the same device type that you use for inference. Measurements are device-dependent, so scales collected during calibration on Intel® Gaudi® 3 cannot be reused on Intel® Gaudi® 2, and vice versa. Using measurements generated on a different device type may cause accuracy issues.

- For prototyping or testing your model with FP8, you can set the environment variable `VLLM_SKIP_WARMUP=true` to disable the time-consuming warm-up stage. However, we do not recommend disabling it in production environments, as it can lead to a significant performance drop.

- When using FP8 models, you may experience timeouts caused by the long compilation time required of FP8 operations. To address this issue, you can configure the following environment variables:

    | Variable | Description |
    |----------|-------------|
    | `VLLM_ENGINE_ITERATION_TIMEOUT_S` | Adjusts the vLLM server timeout. The value is specified in seconds. |
    | `VLLM_RPC_TIMEOUT` | Adjusts the RPC protocol timeout used by the OpenAI-compatible API. The value is specified in microseconds. |

    Setting higher timeout values can help prevent interruptions during model compilation and improve stability when running FP8 models.

- For benchmarking FP8 models with `scale_format=const`, the `VLLM_DISABLE_MARK_SCALES_AS_CONST=true` setting can help speed up the warm-up stage.

- When running FP8 models with `scale_format=scalar` and lazy mode (`PT_HPU_LAZY_MODE=1`), you can set `RUNTIME_SCALE_PATCHING=1` to significantly reduce warm-up time. However, it may introduce a small performance degradation. Runtime Scale Patching is enabled by default for Torch compile.

## Running Online Inference

To run FP8 inference with vLLM, use the following command:

```bash
export QUANT_CONFIG=/path/to/quant/config/inc/meta-llama-3.1-405b-instruct/maxabs_quant_g3.json
vllm serve meta-llama/Llama-3.1-405B-Instruct --dtype bfloat16 --max-model-len  2048 --block-size 128 --max-num-seqs 32 --quantization inc --kv-cache-dtype fp8_inc --tensor-parallel-size 8
```

Where:

- `QUANT_CONFIG`: Environment variable that specifies the path to the quantization configuration file generated during the calibration process.

- The `--quantization inc` and `--kv-cache-dtype fp8_inc` parameter enable the FP8 quantization using INC and `QUANT_CONFIG`.

## Running Offline Inference

To run offline inference follow these steps:

1. Set the `QUANT_CONFIG` environment variable to the path of the JSON configuration file generated during calibration.

    ```bash
    export QUANT_CONFIG=/path/to/quant/config/inc/meta-llama-3.1-405b-instruct/maxabs_quant_g3.json
    ```

2. Pass `quantization="inc"` and `kv_cache_dtype="fp8_inc"` as parameters to the `LLM` object.

    ```python
    from vllm import LLM

    llm = LLM("llama3.1/Meta-Llama-3.1-8B-Instruct", quantization="inc", kv_cache_dtype="fp8_inc")
    ```

## Model Weight Loading Process

The unquantized weights are initially loaded onto the CPU, where they are quantized before being transferred to the target device (HPU) for model execution. This process reduces the model’s memory footprint on the device, since only the quantized weights are stored in device memory.
