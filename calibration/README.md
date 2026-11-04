# Calibration

vLLM Hardware Plugin for Intel® Gaudi® runs FP8 inference through [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8). Models are calibrated with the `vllm-gaudi-calibrate` command (also available as `python -m vllm_gaudi.calibration`), which handles text and multimodal models:

```bash
pip install -r calibration/requirements.txt
vllm-gaudi-calibrate run <model> -o <output_dir>
```

For details, see the calibration documentation:

- [Calibration Overview](../docs/configuration/calibration/calibration.md)
- [Quick Start](../docs/configuration/calibration/quickstart.md)
- [Reference](../docs/configuration/calibration/reference.md)
- [Advanced Usage](../docs/configuration/calibration/advanced.md)
- [Troubleshooting](../docs/configuration/calibration/troubleshooting.md)
- [Migration from `calibrate_model.sh`](../docs/configuration/calibration/migration.md)

The `calibrate_model.sh` scripts in this directory are deprecated stubs that print the equivalent `vllm-gaudi-calibrate` command.
