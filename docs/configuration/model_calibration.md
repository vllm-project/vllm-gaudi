# Quantization, FP8 Inference and Model Calibration Process

To run quantized models using vLLM Hardware Plugin for Intel® Gaudi®, you need measurement files, which you can get by following the FP8 model calibration procedure available in the [FP8 Calibration and Inference with vLLM](https://docs.habana.ai/en/v1.21.0/PyTorch/Inference_on_PyTorch/vLLM_Inference/vLLM_FP8_Inference.html) guide. For an end-to-end example tutorial for quantizing a BF16 Llama 3.1 model to FP8 and then inferencing, see this [this guide](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/vLLM_Tutorials/FP8_Quantization_using_INC/FP8_Quantization_using_INC.ipynb).

Once you have completed the model calibration process and collected the measurements, you can run FP8 inference with vLLM using the following commands:

```bash
export QUANT_CONFIG=/path/to/quant/config/inc/meta-llama-3.1-405b-instruct/maxabs_quant_g3.json
vllm serve meta-llama/Llama-3.1-405B-Instruct --dtype bfloat16 --max-model-len  2048 --block-size 128 --max-num-seqs 32 --quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu --tensor-parallel-size 8
```

`QUANT_CONFIG` is an environment variable that points to the measurement or quantization configuration file. The measurement configuration file is required during calibration to collect
measurements for a given model. The quantization configuration is needed during inference.

Here are a few recommendations for this process:

- For prototyping or testing your model with FP8, you can use the `VLLM_SKIP_WARMUP=true` environment variable to disable the time-consuming warm-up stage. However, we do not recommend disabling this feature in production environments, as it can lead to a significant performance decrease.

- For benchmarking an FP8 model with `scale_format=const`, the `VLLM_DISABLE_MARK_SCALES_AS_CONST=true` setting can help speed up the warm-up stage.

- When using FP8 models, you may experience timeouts caused by the long compilation time of FP8 operations. To mitigate this, set the following environment variables:

  - `VLLM_ENGINE_ITERATION_TIMEOUT_S`: Adjusts the vLLM server timeout to the provided value, in seconds.
  - `VLLM_RPC_TIMEOUT`: Adjusts the RPC protocol timeout used by the OpenAI-compatible API, in miliseconds.

- When running FP8 models with `scale_format=scalar` and lazy mode (`PT_HPU_LAZY_MODE=1`) in order to reduce warm-up time, it is useful to set `RUNTIME_SCALE_PATCHING=1`. This may introduce a small performance degradation but the warm-up time should be significantly reduced. Runtime Scale Patching is enabled by default for Torch compile.
