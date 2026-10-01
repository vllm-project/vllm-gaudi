# Calibration Quick Start

This page calibrates a model for FP8 inference on a single Intel® Gaudi® node and serves it. For the concepts behind each step, see the [Calibration Overview](calibration.md).

## 1. Install the Dependencies

Install vLLM and vLLM Hardware Plugin for Intel® Gaudi® by following the [Installation](../../getting_started/installation.md) procedure. Installing the plugin provides the `vllm-gaudi-calibrate` command. Then, from the plugin project directory, install the calibration dependencies:

```bash
pip install -r calibration/requirements.txt
```

Do not install `lm_eval[vllm]`. The `vllm` extra installs vLLM from PyPI over the HPU build.

## 2. Calibrate the Model

Calibrate a text model:

```bash
vllm-gaudi-calibrate run Qwen/Qwen2.5-0.5B-Instruct -o ./fp8_output
```

Calibrate a vision-language model with the same command. The tool detects the modality and switches to the multimodal defaults (`mmmu_val` task, vision tower and `lm_head` kept in BF16):

```bash
vllm-gaudi-calibrate run Qwen/Qwen2.5-VL-3B-Instruct -o ./fp8_output
```

If the MEASURE phase of Qwen2.5-VL stops with an `mm_embeddings` assertion, see [Multimodal Measure Phase Stops on mmmu_val](troubleshooting.md#multimodal-measure-phase-stops-on-mmmu_val).

`MODEL` is a Hugging Face model ID or a local model directory. For a model that needs more than one card, add `--tp N`. The command runs the MEASURE phase, the QUANTIZE phase, and a short smoke evaluation of the quantized model. With the defaults, the MEASURE phase processes 512 samples of each task (`--limit`).

To preview the generated INC configs and the manifest without loading the model, add `--dry-run`. On a host without an HPU, also pass `--device g2` or `--device g3`.

## 3. Serve the Model

When the run finishes, the tool prints the serve command, for example:

```bash
QUANT_CONFIG=<output_dir>/qwen2.5-0.5b-instruct/maxabs_quant_g3.json vllm serve Qwen/Qwen2.5-0.5B-Instruct --quantization inc --kv-cache-dtype fp8_inc --tensor-parallel-size 1
```

`<output_dir>` is the absolute path of the directory given with `-o`. Add any other `vllm serve` options you need. For the full serving guide, see [Intel® Neural Compressor](../quantization/inc.md).

## 4. Check the Manifest

Each run writes `<output_dir>/<model_name>/<device>/calibration_manifest.json`. It records how the scales were produced:

| Key | Content |
|-----|---------|
| `manifest_version`, `created_at` | Manifest format version and UTC creation time. |
| `versions` | Installed versions of vLLM, vLLM Hardware Plugin for Intel® Gaudi®, lm-eval, INC, PyTorch, the Habana PyTorch plugin, and transformers, where found. |
| `status` | `ok` for a completed run, `dry-run` for `--dry-run`. |
| `model`, `model_name`, `device` | Model argument, output directory name of the model, and device type. |
| `model_info` | Detected facts: `model_type`, `architectures`, `is_multimodal`, `is_moe`, `num_experts`, `is_encoder_decoder`, `has_chat_template`, and `source` (the library used for detection). |
| `preset` | The resolved preset: blocklist, allowlist, scale method and format, engine arguments, environment, and notes. |
| `tasks`, `limit`, `num_fewshot`, `apply_chat_template`, `fewshot_as_multiturn` | The lm-eval settings of the MEASURE phase. |
| `tensor_parallel_size` | The `--tp` of the run. |
| `configs` | Paths of the measure and quant configs. |
| `model_args` | The lm-eval vLLM model arguments of each phase. |
| `phases` | Per phase: `duration_s`, `metrics` (lm-eval results, `null` without evaluation), and `env` (variables the tool added or changed, with credential-like names redacted). For `--dry-run`, only `env`. |
| `postprocess` | Number of KV cache inputs fixed per measurement file. |
| `unify`, `expand` | Target world size and written files, when `--unify-to-tp` or `--expand-to-ep` was used. |
| `serve_command` | The command printed at the end of the run. Not written for `--dry-run`. |
| `files` | The INC output files in the `<device>` directory. |
| `duration_s` | Total run time. |

The `phases.quantize.metrics` entry holds the smoke evaluation results. Compare them with a BF16 run of the same task to spot an accuracy problem early. To evaluate the full task set instead, pass `--quantize-eval full`.

## Next Steps

- [Reference](reference.md): All options of `vllm-gaudi-calibrate`.
- [Advanced Usage](advanced.md): Serving with another tensor or expert parallel size, multi-node calibration, and custom data.
- [Troubleshooting](troubleshooting.md): What to do when a run fails.
