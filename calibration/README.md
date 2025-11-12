# Calibration

vLLM Hardware Plugin for Intel® Gaudi® supports running inference on HPU with 8-bit floating point (FP8) precision using [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package. Inference requires prior calibration to generate the necessary measurements, quantization files, and configuration data that are required for running quantized models.

Detailed calibration procedures for a single Intel® Gaudi® node and multiple nodes are available in the `docs` folder:

- [Introduction](../docs/configuration/calibration/calibration.md): Overview, recommendations, and troubleshooting guidance  
- [Simple calibration procedure for a single Intel® Gaudi® node](../docs/configuration/calibration/calibration_one_node.md)
- [Calibration procedure for multiple Intel® Gaudi® nodes](../docs/configuration/calibration/calibration_multi_node)
