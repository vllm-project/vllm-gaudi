# Quantization, FP8 Inference and Model Calibration Process

!!! note
    Measurement files are required to run quantized models with vLLM on Gaudi accelerators. The FP8 model calibration procedure is described in detail in [docs.habana.ai vLLM Inference Section](https://docs.habana.ai/en/v1.21.0/PyTorch/Inference_on_PyTorch/vLLM_Inference/vLLM_FP8_Inference.html).

An end-to-end example tutorial for quantizing a BF16 Llama 3.1 model to FP8 and then inferencing is provided in this [Gaudi-tutorials repository](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/vLLM_Tutorials/FP8_Quantization_using_INC/FP8_Quantization_using_INC.ipynb).

Once you have completed the model calibration process and collected the measurements, you can run FP8 inference with vLLM using the following command:

```bash
export QUANT_CONFIG=/path/to/quant/config/inc/meta-llama-3.1-405b-instruct/maxabs_quant_g3.json
vllm serve meta-llama/Llama-3.1-405B-Instruct --dtype bfloat16 --max-model-len  2048 --block-size 128 --max-num-seqs 32 --quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu --tensor-parallel-size 8
```

`QUANT_CONFIG` is an environment variable that points to the measurement or quantization configuration file. The measurement configuration file is used during the calibration procedure to collect
measurements for a given model. The quantization configuration is used during inference.

!!! tip
    If you are prototyping or testing your model with FP8, you can use the `VLLM_SKIP_WARMUP=true` environment variable to disable the warmup stage, which is time-consuming.
    However, disabling this feature in production environments is not recommended, as it can lead to a significant performance decrease.

!!! tip
    If you are benchmarking an FP8 model with `scale_format=const`, setting `VLLM_DISABLE_MARK_SCALES_AS_CONST=true` can help speed up the warmup stage.

!!! tip
    When using FP8 models, you may experience timeouts caused by the long compilation time of FP8 operations. To mitigate this, set the following environment variables:
    - `VLLM_ENGINE_ITERATION_TIMEOUT_S` - to adjust the vLLM server timeout. You can set the value in seconds, e.g., 600 equals 10 minutes.
    - `VLLM_RPC_TIMEOUT` - to adjust the RPC protocol timeout used by the OpenAI-compatible API. This value is in microseconds, e.g., 600000 equals 10 minutes.
