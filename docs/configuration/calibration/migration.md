# Migrating from calibrate_model.sh

`vllm-gaudi-calibrate` replaces `calibration/calibrate_model.sh`, `calibration/vlm-calibration/calibrate_model.sh`, and the `step-*.py` scripts. Both `calibrate_model.sh` files are kept as stubs: they no longer calibrate, print the equivalent `vllm-gaudi-calibrate run` command to standard error, and exit with code `2`. With `-h`, they print the same message to standard output and exit with code `0`. The stubs will be removed in a future release.

The output layout and the config file names are unchanged, so existing `QUANT_CONFIG` paths and serve commands keep working. See [Output Layout](calibration.md#output-layout).

## Options of calibrate_model.sh

The same table applies to the text and the vision-language script. The new command detects the modality, so both use `vllm-gaudi-calibrate run`.

| Old option | New option | Notes |
|------------|------------|-------|
| `-m <model>` | `MODEL` (positional) | Hugging Face model ID or local directory. |
| `-o <dir>` | `-o <dir>` | Same layout below the directory. |
| `-d <dataset>` | `--tasks`, `--include-path` | Datasets are replaced by lm-eval tasks. See [Calibration Data](#calibration-data). |
| `-l <n>` | `--limit <n>` | Samples per task. Default `512`. Now always honored. |
| `-b <n>` | `--batch-size <n>` | Default changed from `32` to `auto`. |
| `-t <n>` | `--tp <n>` | Values above 8 need `--multi-node`. See [Multi-Node Calibration with Ray](advanced.md#multi-node-calibration-with-ray). |
| `-r <n>` | `--unify-to-tp <n>` | Must be smaller than `--tp` and divide it. |
| `-u` | `--expert-parallel` | Measures and quantizes with expert parallelism and unifies with the expert parallel rule. The old vision-language script used `-u` only for the unify rule. |
| `-e` | `--enforce-eager` | |
| `-h` | `vllm-gaudi-calibrate run --help` | The old `-h` was declared as taking an argument, so it printed the usage as an error and exited with code `1`. |

For example:

```bash
# Old
./calibrate_model.sh -m meta-llama/Llama-3.1-405B-Instruct -d <dataset>.pkl -o <output_dir> -b 128 -t 16 -l 4096 -r 8

# New
vllm-gaudi-calibrate run meta-llama/Llama-3.1-405B-Instruct -o <output_dir> --tp 16 --limit 4096 --unify-to-tp 8 --multi-node
```

Options that the old script passed to the step scripts and that are gone:

| Old setting | Replacement |
|-------------|-------------|
| `--block-quant` and `VLLM_HPU_FORCE_CHANNEL_FP8=0` for DeepSeek | Not needed. The plugin forces `VLLM_HPU_FORCE_CHANNEL_FP8` to false whenever `QUANT_CONFIG` is set. |
| `--max-num-prefill-seqs` | Removed. Use `--max-num-seqs`, or `--engine-arg` for any other vLLM engine argument. |
| Chat templates in `calibration/template/*.jinja` | The model's own chat template is used through lm-eval. Models without one run on raw prompts. |

## Step Scripts

| Old script | New command | Notes |
|------------|-------------|-------|
| `step-0-detect-device.py` | `vllm-gaudi-calibrate detect MODEL` | Prints the device type and the model facts. The old script exited with the device digit as its exit code. |
| `step-1-prepare-calibration-dataset.py` | `vllm-gaudi-calibrate run` | Data comes from lm-eval tasks. |
| `step-2-measure-scales.py` | `vllm-gaudi-calibrate run --phases measure` | |
| `step-3-postprocess-measure.py -m D -o O` | `vllm-gaudi-calibrate postprocess -m D -o O` | `-d` (DeepSeek) is gone: the MLA `latent_cache_k` inputs are matched automatically. `run` applies this step by default. |
| `step-4-quantize-scales.py` | `vllm-gaudi-calibrate run --phases quantize` | |
| `step-5-unify_measurements.py -m D -r N -o O -u -s` | `vllm-gaudi-calibrate unify -m D -r N -o O --ep --skip-scales` | `-u` becomes `--ep`, `-s` becomes `--skip-scales`. |
| `step-6-expand-measurements.py -m D -w N -o O` | `vllm-gaudi-calibrate expand -m D -w N -o O` | |
| `deepseek_gaudi2_converter.py` | Removed. | |

The old step 3, 5, and 6 scripts wrote to the current directory when `-o` was omitted. The new `postprocess`, `unify`, and `expand` subcommands write in place, into the measurement directory.

## Calibration Data

The old scripts read a `.pkl` dataset (`-d`) or downloaded a Hugging Face dataset. The new tool takes its data from lm-eval tasks: by default `pile_10k gsm8k` for text models and `mmmu_val` for multimodal models.

- **`.pkl` datasets are not supported.** Convert the DataFrame to JSON Lines and wrap it in an lm-eval task YAML file. See [Custom Calibration Data](advanced.md#custom-calibration-data) for a conversion command and a task that uses the old `system_prompt` and `question` fields.
- **The local MMMU cache (`-d` of the vision-language script)** is replaced by the Hugging Face cache. Set `HF_HOME` or `HF_DATASETS_CACHE` before the run; the phases inherit the environment.

## Behavior Changes

| Area | Old behavior | New behavior |
|------|--------------|--------------|
| Preset selection | Substring match on the model path, for example `deepseek` or `mixtral` in the directory name. | Match on the `model_type` of the model config. The model name is used only when `model_type` is empty. `DeepSeek-R1-Distill-Llama` models now get the default preset, and Mixtral is detected regardless of the directory name. See [Presets](reference.md#presets). |
| Vision-language blocklist | `lm_head` only. | `lm_head` and the vision tower and projector modules. Pass `--quantize-vision-tower` to quantize the vision modules. |
| Image size | Images kept their original size. | lm-eval resizes images to a longest side of 1280 pixels, which bounds the vision encoder memory. Pass `--image-max-side 0` to keep the original size. |
| `--limit` | Ignored on the Hugging Face dataset path, which forced 1 sample with batch size 1 and 32 output tokens. | Always honored, 512 samples per task by default. |
| `max_model_len` | 2048 on the Hugging Face dataset path. | 4096 for text models, 8192 for multimodal models, 2048 for the `granite4` preset. |
| Fused MoE detection in unify and expand | Any node whose name contains `moe` was treated as a fused MoE op, including Mixtral's `block_sparse_moe.gate`. | Only nodes with per-expert child nodes are fused MoE ops. |
| MoE with `--tp` > 1 and no unify | No warning. | A reminder is logged that serving with another world size needs `--unify-to-tp`. |
| Multi-node `QUANT_CONFIG` | You exported `QUANT_CONFIG` to a shared buffer file on every node before `ray start`. | The tool sets `QUANT_CONFIG` for each phase and the plugin forwards it to the Ray workers. `--quant-config-buffer` keeps the old buffer approach available. See [Shared Config Buffer](advanced.md#shared-config-buffer). |
| Result check | None. | The tool checks that every rank wrote fresh measurement files, runs a smoke evaluation of the quantized model, and writes `calibration_manifest.json`. |

## Dependencies

Install the calibration dependencies from `calibration/requirements.txt`, which now covers text and multimodal models. `calibration/vlm-calibration/requirements.txt` is removed. Do not install `lm_eval[vllm]`, because the `vllm` extra replaces the HPU build of vLLM with the one from PyPI.

```bash
pip install -r calibration/requirements.txt
```

## Removed Files

- `calibration/step-0-detect-device.py` to `calibration/step-6-expand-measurements.py`
- `calibration/template/llama-2-chat.jinja`, `calibration/template/mistral_mixtral.jinja`
- `calibration/deepseek_gaudi2_converter.py`
- `calibration/unify-and-expand.png` (now `docs/assets/calibration/unify-and-expand.png`)
- `calibration/vlm-calibration/README.md`, `calibration/vlm-calibration/requirements.txt`, `calibration/vlm-calibration/vision_lm_eval.py`
- `docs/configuration/calibration/calibration_one_node.md`, `docs/configuration/calibration/calibration_multi_node.md` (replaced by these pages)
