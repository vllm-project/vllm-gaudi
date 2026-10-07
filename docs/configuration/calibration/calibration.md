# Calibration Overview

vLLM Hardware Plugin for Intel® Gaudi® runs inference with 8-bit floating point (FP8) weights and activations through [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8). Before a model can be served in FP8, it has to be calibrated: INC observes the value ranges of the model's activations and weights on sample data and turns them into FP8 scales.

The plugin ships one calibration tool for text and multimodal models, `vllm-gaudi-calibrate` (also available as `python -m vllm_gaudi.calibration`). It replaces the `calibrate_model.sh` scripts and the `step-*.py` scripts used in earlier releases. If you used those, see [Migration](migration.md).

## How Calibration Works

`vllm-gaudi-calibrate run` performs these steps:

1. **Detect.** The tool reads the HPU device type (`g2` for Intel® Gaudi® 2, `g3` for Intel® Gaudi® 3) and inspects the model: modality (text or multimodal), Mixture of Experts (MoE) layers, encoder-decoder architecture, and whether a chat template exists. Encoder-decoder models are rejected, because INC calibration supports decoder-only models.
2. **Configure.** It selects a model family preset (see [Presets](reference.md#presets)) and writes two INC configuration files, one for each phase.
3. **MEASURE phase.** A vLLM engine is started with `QUANT_CONFIG` pointing to the measure config. INC installs observers, and lm-eval tasks feed data through the model. When the engine shuts down, INC writes the measurement files from `finalize_calibration`.
4. **Postprocess.** The tool copies the KV cache input ranges into the attention matmul measurements, so the FP8 KV cache uses the cached tensor ranges. Use `--no-postprocess` to skip this step.
5. **QUANTIZE phase.** A second engine is started with the quant config. INC computes the scales from the measurements and writes the scale files. By default, the phase then runs a short smoke evaluation on the quantized model.
6. **Optionally unify or expand** the measurements for another tensor parallel or expert parallel size (see [Advanced Usage](advanced.md)).
7. **Write the manifest** and print the `vllm serve` command for the calibrated model.

Each phase runs in its own Python process. INC reads `QUANT_CONFIG` when the HPU model runner is imported, and writes the measurement files only when the engine shuts down, so each phase needs a fresh interpreter with its own environment.

### Calibration Data

The data comes from [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) tasks, run through lm-eval's vLLM model (`VLLM` for text models, `VLLM_VLM` for multimodal models). The defaults are:

| Modality | Default tasks | Samples |
|----------|---------------|---------|
| Text | `pile_10k gsm8k` | 512 per task (`--limit`) |
| Multimodal | `mmmu_val` | all 900: `mmmu_val` is a group of 30 subtasks with 30 samples each, and `--limit` applies to each subtask |

`pile_10k` is a loglikelihood task that exercises prefill, and `gsm8k` is a generation task that also exercises decode and the KV cache. You can choose other tasks with `--tasks`, or provide your own lm-eval task YAML files with `--include-path`. For details, see [Default Tasks and Limits](reference.md#default-tasks-and-limits) and [Custom Calibration Data](advanced.md#custom-calibration-data).

### Device Type

Calibrate on the same device type that you use for inference. The scales depend on the device, so scales generated on Intel® Gaudi® 3 cannot be reused on Intel® Gaudi® 2, and vice versa. The device type is part of every output path, so calibrations for both device types can share one output directory.

## Output Layout

For a model `Org/Model-Name`, the output directory given with `-o` contains:

```text
<output_dir>/
  model-name/                          # last path component of the model, lower-cased
    maxabs_measure_<device>.json       # INC config of the MEASURE phase
    maxabs_quant_<device>.json         # INC config of the QUANTIZE phase; use it for serving
    <device>/                          # g2 or g3
      inc_output_hooks_maxabs_<rank>_<world>.json             # measurements
      inc_output_hooks_maxabs_<rank>_<world>.npz
      inc_output_hooks_maxabs_<rank>_<world>_mod_list.json    # measured module list
      inc_output_hooks_maxabs_MAXABS_HW_<rank>_<world>.json   # scales
      inc_output_hooks_maxabs_MAXABS_HW_<rank>_<world>.npz
      calibration_manifest.json        # what was calibrated, how, with which versions
      logs/                            # measure.log, quantize.log; removed after success unless --keep-logs
```

`<world>` is the tensor parallel size of the run and `<rank>` goes from `0` to `<world> - 1`. The layout and the config file names are the same as with the old `calibrate_model.sh`, so existing `QUANT_CONFIG` paths keep working.

A run into a directory that already holds a calibration of the same model and device replaces it. Before the MEASURE phase, the tool removes all `inc_output_hooks_*` files of the earlier run, and before the QUANTIZE phase, the scale files. INC would otherwise keep the earlier scales and compute new ones only for modules that have none.

## Serving the Calibrated Model

Serving reads the quant config through the `QUANT_CONFIG` environment variable. The config's `dump_stats_path` points to the measurement and scale files. At the end of a run, the tool prints the serve command and stores it in the manifest:

```bash
QUANT_CONFIG=<output_dir>/<model_name>/maxabs_quant_<device>.json \
    vllm serve <model> --quantization inc --kv-cache-dtype fp8_inc --tensor-parallel-size <N>
```

The tensor parallel size must match the world size of the files in `<device>/`: the `--tp` of the run, or the target of `--unify-to-tp` or `--expand-to-ep`. If the run used expert parallelism, for example with the `deepseek` preset, the files hold the experts of each rank, so the printed command adds `--enable-expert-parallel` whenever the world size is greater than 1. For a checkpoint that is already quantized, such as the FP8 DeepSeek-R1, the tool does not pass `--quantization inc` in the phases or in the printed command: vLLM keeps the quantization method of the checkpoint, and `QUANT_CONFIG` enables INC. For more serving options, see [Intel® Neural Compressor](../quantization/inc.md).

## Prerequisites

- An Intel® Gaudi® 2 or Intel® Gaudi® 3 machine with vLLM and vLLM Hardware Plugin for Intel® Gaudi® installed (see [Installation](../../getting_started/installation.md)).
- The calibration dependencies: `pip install -r calibration/requirements.txt` (`lm_eval>=0.4.12`, `datasets`, `Pillow`, and `ray`, which lm-eval's multimodal vLLM model imports). Do not install `lm_eval[vllm]`, because it replaces the HPU build of vLLM with the one from PyPI.
- Access to the model and to the datasets of the lm-eval tasks, either through the Hugging Face Hub or a local cache.
- Enough HPUs for the tensor parallel size of the run. A model that does not fit on one card needs `--tp`.

## Further Reading

- [Quick Start](quickstart.md): Calibrate and serve a model with one command.
- [Reference](reference.md): Every subcommand and option, presets, environment variables, and the INC configuration schema.
- [Advanced Usage](advanced.md): Unifying and expanding measurements, multi-node calibration, custom data, and custom INC configs.
- [Troubleshooting](troubleshooting.md): Common failures and how to fix them.
- [Migration](migration.md): Moving from `calibrate_model.sh` and the step scripts.
