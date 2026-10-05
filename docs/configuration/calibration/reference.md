# Calibration Reference

This page lists every subcommand and option of `vllm-gaudi-calibrate`, the model family presets, the environment set by the tool, and the INC configuration schema. To print the help of a subcommand, run `vllm-gaudi-calibrate <command> --help`.

## Command Line

```text
vllm-gaudi-calibrate [-v] COMMAND [options]
python -m vllm_gaudi.calibration [-v] COMMAND [options]
```

| Command | Purpose |
|---------|---------|
| `run` | Calibrate a model end to end. |
| `unify` | Merge per-rank measurements to a smaller world size. |
| `expand` | Expand a world size 1 MoE measurement to expert parallel ranks. |
| `postprocess` | Fix the KV cache inputs of attention matmuls in measurements. |
| `detect` | Print the device type and the detected model facts. |
| `print-config` | Print the INC configs `run` would write. |
| `list-tasks` | List lm-eval tasks. |

The global option `-v`/`--verbose` enables debug logging; it goes before the command. The exit code is `0` on success and `1` when a phase fails or the options are invalid; argument parsing errors exit with `2`.

### `run`

```text
vllm-gaudi-calibrate run MODEL -o OUTPUT_DIR [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `MODEL` | required | Local model directory or Hugging Face model ID. |
| `-o`, `--output-dir` | required | Root output directory. See [Output Layout](calibration.md#output-layout). |

#### INC Configuration Options

These options are shared with `print-config`.

| Option | Default | Description |
|--------|---------|-------------|
| `--device {g2,g3}` | detected | Device type. Detected from the HPU when not set. Required for `--dry-run` on a host without an HPU. |
| `--modality {text,multimodal}` | detected | Override the detected modality. |
| `--trust-remote-code` | off | Trust custom model code. Also added to the printed serve command. |
| `--scale-method` | `maxabs_hw` | INC `scale_method`. |
| `--scale-format` | preset | INC `scale_format`, for example `CONST` or `scalar`. Overrides the preset value. |
| `--blocklist NAME [NAME ...]` | none | Module name patterns kept in high precision, added to the preset blocklist. |
| `--allowlist NAME [NAME ...]` | none | INC allowlist module names. |
| `--quantize-vision-tower` | off | Quantize the vision tower of a multimodal model. By default, it stays in BF16. |
| `--input-backoff` | unset | INC `scale_params.input_backoff`. |
| `--weight-backoff` | unset | INC `scale_params.weight_backoff`. |
| `--device-for-scales` | unset | INC `device_for_scales`, for example `GAUDI2` for scales compatible with Intel® Gaudi® 2. |
| `--measure-exclude` | unset | INC `measure_exclude`, for example `NONE` or `OUTPUT`. |
| `--dynamic-quantization` | unset | Write `"dynamic_quantization": true` to the quant config. |
| `--measure-config PATH` | generated | Use this INC measure JSON instead of generating one. See [Custom INC Configs](advanced.md#custom-inc-configs). |
| `--quant-config PATH` | generated | Use this INC quant JSON instead of generating one. |

The options marked `unset` are written to the quant config only when given. `--blocklist` and `--allowlist` apply to both configs.

#### lm-eval Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tasks TASK [TASK ...]` | `pile_10k gsm8k` (text), `mmmu_val` (multimodal) | lm-eval tasks, groups, or tags fed to the model. |
| `--limit` | `512` | Samples per task. Must be at least 1. |
| `--num-fewshot` | task default | Number of few-shot examples. |
| `--max-gen-toks` | `256` | Generation budget for tasks that do not set their own. |
| `--include-path DIR` | none | Directory with custom lm-eval task YAML files. |
| `--no-chat-template` | off | Do not format prompts with the chat template. Also disables few-shot as multi-turn. |
| `--no-fewshot-as-multiturn` | off | Put few-shot examples in one user turn instead of separate chat turns. |

#### Engine Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tp` | `1` | Tensor parallel size. |
| `--expert-parallel` | off (preset) | Enable expert parallelism. Also selects the expert parallel merge rule for `--unify-to-tp` and adds `--enable-expert-parallel` to the printed serve command. The `deepseek` preset enables it; to turn it off, pass `--engine-arg enable_expert_parallel=false`. |
| `--batch-size` | `auto` | lm-eval batch size, `auto` or a positive integer. |
| `--max-num-seqs` | preset (`32`) | vLLM `max_num_seqs`. |
| `--max-model-len` | preset (`4096` text, `8192` multimodal, `2048` granite4) | vLLM `max_model_len`. |
| `--enforce-eager` | off | Run without HPU graphs. |
| `--gpu-memory-utilization` | vLLM default (preset) | vLLM `gpu_memory_utilization`. The `granite4` preset sets `0.1`. |
| `--dtype` | `bfloat16` | Model dtype. |
| `--max-images` | `1` | Images per prompt, multimodal models only. |
| `--image-max-side <pixels>` | `1280` | Multimodal models only. lm-eval resizes each image so that its longest side is at most this many pixels, keeping the aspect ratio. `0` keeps the original size. |
| `--engine-arg KEY=VALUE` | none | Extra `vllm.LLM` argument. `VALUE` is parsed as JSON when possible (`true`, `0.5`, `{"a": 1}`), otherwise kept as a string. Overrides every other engine argument, except `pretrained` and `tensor_parallel_size`, which are rejected; use the `MODEL` argument and `--tp`. Repeatable. |
| `--env KEY=VALUE` | none | Environment variable for the MEASURE and QUANTIZE phases. Overrides the tool and preset defaults. Repeatable. |

#### Flow Options

| Option | Default | Description |
|--------|---------|-------------|
| `--phases` | `measure,quantize` | Phases to run, a comma-separated subset of `measure,quantize`. |
| `--quantize-eval {none,smoke,full}` | `smoke` | Evaluation in the QUANTIZE phase: `none`, a smoke run of the first task, or the full task set with `--limit`. |
| `--smoke-limit` | `8` | Samples of the smoke evaluation. The smoke run uses `min(--smoke-limit, --limit)`. |
| `--no-postprocess` | off | Skip the KV cache input fix of the measurements. |
| `--unify-to-tp N` | none | Unify the measurements to world size `N`. `N` must be smaller than `--tp` and divide it. |
| `--expand-to-ep N` | none | Expand a MoE measurement to `N` expert parallel ranks. `N` must be at least 2 and differ from `--tp`. With `--tp` above 1, the measurements are first unified to world size 1. Combining it with `--unify-to-tp` other than `1` is an error. |

#### Environment Options

| Option | Default | Description |
|--------|---------|-------------|
| `--multi-node` | off | Run the engine on a Ray cluster (`distributed_executor_backend="ray"`). |
| `--quant-config-buffer PATH` | none | Shared file that every node's `QUANT_CONFIG` points to, for clusters that do not forward `QUANT_CONFIG` to the Ray workers. Requires `--multi-node`. |
| `--dry-run` | off | Detect the model and write the configs and the manifest without loading the model. |
| `--keep-logs` | off | Keep `<device>/logs/` after a successful run. Logs are always kept when a phase fails. |

### `unify`

Merges per-rank measurement and scale files to a smaller world size. The source world size is read from the `*_mod_list.json` files.

```text
vllm-gaudi-calibrate unify -m DIR -r N [-o OUT] [--ep] [--skip-scales] [--observer NAME]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--measurements` | required | Directory with the measurement files, for example `<output_dir>/<model_name>/<device>`. |
| `-r`, `--rank` | required | Target world size. Must be smaller than the measured world size and divide it. |
| `-o`, `--out` | in place | Output directory. |
| `--ep` | off | The measurements were taken with expert parallelism. Experts of the merged ranks are concatenated instead of maxed. |
| `--skip-scales` | off | Unify only the measurement files, not the scale files. |
| `--observer` | `maxabs` | INC observer in the file names. Pass the `observer` of a custom measure config. |

### `expand`

Splits a world size 1 MoE measurement into `N` expert parallel ranks. Every rank gets a copy of the measurement in which each fused MoE op keeps only the intermediate maxima of the experts that rank owns; all other nodes are copied unchanged. The number of experts must divide evenly by `N`. Only the measurement files are written; INC computes the scales from them when the model is served.

```text
vllm-gaudi-calibrate expand -m DIR -w N [-o OUT] [--observer NAME]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--measurements` | required | Directory with exactly one world size 1 measurement (`*_0_1.json`). |
| `-w`, `--world-size` | required | Target expert parallel world size, at least 2. |
| `-o`, `--out` | in place | Output directory. |
| `--observer` | `maxabs` | INC observer in the file names. Pass the `observer` of a custom measure config. |

### `postprocess`

Copies the input range of the KV cache modules (`k_cache`, `v_cache`, or `latent_cache_k` for MLA) into the second input of `matmul_qk` and `matmul_av`. `run` does this automatically after the MEASURE phase. The standalone command processes every measurement and scale file in the directory.

```text
vllm-gaudi-calibrate postprocess -m DIR [-o OUT] [--observer NAME]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-m`, `--measurements` | required | Directory with the measurement files. |
| `-o`, `--out` | in place | Output directory. |
| `--observer` | `maxabs` | INC observer in the file names. Pass the `observer` of a custom measure config. |

### `detect`

Prints the device type and the model facts the presets use.

```text
vllm-gaudi-calibrate detect MODEL [--trust-remote-code] [--no-device] [--json]
```

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL` | required | Local model directory or Hugging Face model ID. |
| `--trust-remote-code` | off | Trust custom model code. |
| `--no-device` | off | Skip the HPU device query, for example on a host without an HPU. |
| `--json` | off | Print JSON. |

Example output:

```text
device: g3
model_type: qwen2
architectures: ['Qwen2ForCausalLM']
is_multimodal: False
is_moe: False
num_experts: None
is_encoder_decoder: False
has_chat_template: True
source: vllm
```

Detection uses vLLM's `ModelConfig` when vLLM is importable and falls back to transformers otherwise (`source: transformers`).

### `print-config`

Prints the detected model facts, the resolved preset, and the measure and quant configs that `run` would write, keyed by their file paths. Nothing is written.

```text
vllm-gaudi-calibrate print-config MODEL [-o OUTPUT_DIR] [INC configuration options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `MODEL` | required | Local model directory or Hugging Face model ID. |
| `-o`, `--output-dir` | `.` | Root output directory the printed paths refer to. |

It accepts every option listed in [INC Configuration Options](#inc-configuration-options). Pass `--device` on a host without an HPU.

### `list-tasks`

```text
vllm-gaudi-calibrate list-tasks [--modality {text,multimodal,all}] [--include-path DIR]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--modality` | `all` | `text` or `multimodal` prints the default tasks of that modality. `all` prints every task, group, and tag lm-eval knows. |
| `--include-path` | none | Directory with custom lm-eval task YAML files, included in the `all` listing. |

## Presets

A preset holds the settings that differ between model families. The tool matches presets on the Hugging Face `model_type` of the model. The model name regex is used only when the model type is unknown, for example in `--dry-run` when detection fails. The regex is matched against the lower-cased last path component of `MODEL`.

| Preset | `model_type` | Name regex fallback | Blocklist | `scale_format` | Engine arguments | Environment |
|--------|--------------|---------------------|-----------|----------------|------------------|-------------|
| `mixtral` | `mixtral` | `^mixtral` | `self_attn`, `lm_head` | `CONST` | | |
| `deepseek` | `deepseek_v2`, `deepseek_v3` | `^deepseek` | `lm_head`, `mlp\.gate\b` | `scalar` | `enable_expert_parallel=True` | |
| `granite4` | `granitemoehybrid` | `^granite-4` | `mamba`, `self_attn` | INC default | `gpu_memory_utilization=0.1`, `max_model_len=2048` | `VLLM_CONTIGUOUS_PA=false` |
| `default` | any other | | none | INC default | | |

Every preset starts from `max_num_seqs=32` and `max_model_len=4096`. The family engine arguments are applied on top of these values.

For a multimodal model, a multimodal overlay is applied to the matched family, and the preset name gets a `+multimodal` suffix (for example `default+multimodal`). The overlay:

- Adds `lm_head` to the blocklist.
- Adds the vision tower and projector names `visual`, `vision_tower`, `vision_model`, `multi_modal_projector`, and `mm_projector` to the blocklist, unless `--quantize-vision-tower` is given.
- Sets `max_model_len=8192` and `disable_log_stats=True` before the family engine arguments are applied.

Explicit options take precedence over the preset: `--blocklist` names are appended to the preset blocklist, `--scale-format` replaces the preset format, and `--max-num-seqs`, `--max-model-len`, `--gpu-memory-utilization`, `--expert-parallel`, and `--engine-arg` replace the preset engine arguments. `--env` overrides the preset environment.

To see the resolved preset for a model, run `vllm-gaudi-calibrate print-config MODEL --device g3`.

## Default Tasks and Limits

| Modality | Default `--tasks` |
|----------|-------------------|
| Text | `pile_10k gsm8k` |
| Multimodal | `mmmu_val` |

- `--limit` is the number of samples per task, not in total, and it is always honored. With the text defaults, the MEASURE phase processes up to 1024 samples.
- The tasks are validated in the phase process before the model is loaded. An unknown task fails the phase; use `list-tasks` to find valid names.
- The QUANTIZE phase evaluates according to `--quantize-eval`: `smoke` runs the first task with `min(--smoke-limit, --limit)` samples, `full` runs all tasks with `--limit`, and `none` only loads the quantized model, which is enough for INC to write the scales.
- If the tokenizer (or, for multimodal models, the processor) has no chat template, the tool logs a warning and runs the tasks on raw prompts, with `--no-chat-template` behavior.
- Each phase uses seed `42`, the lm-eval vLLM model with `quantization="inc"`, and the KV cache dtype `auto` in MEASURE and `fp8_inc` in QUANTIZE.

## Environment Variables

The tool sets these variables for the MEASURE and QUANTIZE phase processes, on top of the inherited environment:

| Variable | Value | When |
|----------|-------|------|
| `VLLM_SKIP_WARMUP` | `true` | Always. Warmup would feed synthetic activations to the observers. |
| `PT_HPU_WEIGHT_SHARING` | `0` | Always. Weight sharing breaks INC module patching. |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES` | `true` | When `--tp` is greater than 1. |
| Preset environment | see [Presets](#presets) | For example `VLLM_CONTIGUOUS_PA=false` for `granite4`. |
| `--env KEY=VALUE` | user value | Applied after the variables above, so it overrides them. |
| `QUANT_CONFIG` | the phase's INC config, or the `--quant-config-buffer` file | Always, set last. A `QUANT_CONFIG` exported in your shell is ignored, and `--env` cannot override it. |

The tool does not set `VLLM_HPU_FORCE_CHANNEL_FP8`. The plugin forces it to false whenever `QUANT_CONFIG` is set.

The variables that the tool added or changed are recorded per phase in the manifest (`phases.<phase>.env`). Values of variables whose names contain `TOKEN`, `SECRET`, `PASSWORD`, `KEY`, or `CREDENTIAL` are recorded as `<redacted>`.

## INC Configuration

`run` generates both INC configs from the preset and the options. The keys and their order are the same as the configs written by the old `calibrate_model.sh`. A generated measure config for a text model with the `default` preset:

```json
{
    "method": "HOOKS",
    "mode": "MEASURE",
    "observer": "maxabs",
    "allowlist": {"types": [], "names": []},
    "blocklist": {"types": [], "names": []},
    "quantize_weight": false,
    "dump_stats_path": "<output_dir>/<model_name>/<device>/inc_output",
    "calibration_sample_interval": 1
}
```

The matching quant config:

```json
{
    "mode": "QUANTIZE",
    "observer": "maxabs",
    "scale_method": "maxabs_hw",
    "allowlist": {"types": [], "names": []},
    "blocklist": {"types": [], "names": []},
    "dump_stats_path": "<output_dir>/<model_name>/<device>/inc_output"
}
```

`scale_format` is written after `scale_method` when the preset or `--scale-format` sets it. `scale_params`, `device_for_scales`, `measure_exclude`, and `dynamic_quantization` are appended when the matching options are given. The `calibration/quantization_config` directory contains further templates that you can pass with `--quant-config` or set as `QUANT_CONFIG` when serving.

### Supported Configuration Options

The following table summarizes the options that you can set in a configuration file:

| Attribute            | Description | Values |
|----------------------|-------------|--------|
| `mode`             | The mode to run INC with. | - `MEASURE`: Measures statistics of all modules and emits the results to `dump_stats_path`.<br>- `QUANTIZE` (default): Quantizes and runs the model according to the provided measurements. |
| `observer`         | The method used to observe and track tensor statistics. | - `maxabs` (default): Tracks the maximum absolute values of tensors.<br>- `save`: Saves all tensors to files. |
| `allowlist`        | The list of `nn.Module` names or types to quantize. Empty list means all supported modules are quantized by default. See [Custom Patched Modules](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#supported-modules). | Default: empty list |
| `blocklist`        | List of `nn.Module` names or types not to quantize. | Default: empty list |
| `dump_stats_path`  | The path to save and load measurements. Directory structure is created up to the last `/`; the string after the last `/` is used as a prefix for measurement files. | Default: `stats` |
| `scale_method`     | The method for calculating the scale from measurements. | - `unit_scale` (default): Always uses the scale of 1.<br>- `maxabs_arbitrary`: Stretches or compresses maxabs to the full-scale of FP8.<br>- `maxabs_hw`: Stretches or compresses maxabs to full-scale of FP8, then replaces it with hardware-accelerated scale based on `device_for_scales`.<br>- `maxabs_pow2`: Stretches or compresses maxabs to full-scale of FP8, then replaces it with hardware-accelerated scale based on `device_for_scales`, rounded to the power of 2.<br>- `maxabs_hw_opt_weight`: The weight scale chosen for the minimal MSE among hardware-accelerated scales; activations use `maxabs_hw`.<br>- `act_maxabs_pow2_weights_pcs_opt_pow2`: Per-channel weights use `maxabs_hw_opt_weight`; activations use `maxabs_pow2`.<br>- `act_maxabs_hw_weights_pcs_maxabs_pow2`: Per-channel weights use `maxabs_pow2`; activations use `maxabs_hw`.<br>- `act_maxabs_pcs_pow2_weight_maxabs_pts_pow2_hw`: Only for dynamic quantization. Per-tensor weights use `maxabs_hw`; activations use per-token `maxabs_pow2`. |
| `measure_exclude`  | Tensor types to exclude from measurement. | - `NONE`: Measures all tensors.<br>- `OUTPUT` (default): Skips output tensors. |
| `scale_format`     | The format of scales passed to custom PyTorch operations. | - `const`: Scales passed as tensors.<br>- `scalar` (default): Scales passed as scalar values for compilation time and throughput optimizations. |
| `device_for_scales`| Exponent-bias values for converting FP32/BF16 to FP8-143. | - `GAUDI3`: The expanded exponent-bias range (0 to 63).<br>- `GAUDI2`: Four possible exponent biases (3, 7, 11, 15), default is 7. |
| `dynamic_quantization` | Enables dynamic FP8 quantization with per-token scales. Only supported with `act_maxabs_pcs_pow2_weight_maxabs_pts_pow2_hw`. | - `true`: Enable.<br>- `false` (default): Disable. |

The defaults in this table are INC defaults, which apply when a key is missing. The configs generated by `vllm-gaudi-calibrate` set `scale_method` to `maxabs_hw`.

### Configuring Backoff Factors

When using any of the maxabs-based `scale_method` options, you can fine-tune the quantization behavior by configuring backoff factors. The `input_backoff` and `weight_backoff` factors provide a safety margin when converting inputs and weights to FP8. For example, if an activation has a larger absolute value than observed in calibration, the maxabs value is scaled to:

```text
input_backoff * FP8_143_FULLSCALE
```

Similarly, for weights:

```text
weight_backoff * FP8_143_FULLSCALE
```

By default, the backoff factors are set to:

- `input_backoff`: 0.25
- `weight_backoff`: 0.5

To change these values, pass `--input-backoff` and `--weight-backoff` to `run`, or add the following to the quantization configuration JSON file:

```json
"scale_params": {"input_backoff": <INPUT_BACKOFF>, "weight_backoff": <WEIGHT_BACKOFF>}
```

### Compilation Time and Throughput Optimization

The `scale_format` configuration option provides performance optimizations for FP8 inference. When set to `scalar` (default), it improves both compilation speed and runtime throughput by reducing the number of compiled recipes and minimizing host-side overhead when launching FP8 operations. Note that the compilation time improvement varies depending on your model's properties, such as the recipe count and scale distribution.

This optimization is not applicable to Per-Channel Quantization (PCQ).

## Measurement File Names

INC names its output files after `dump_stats_path` (`<prefix>`), the observer, the rank, and the world size:

| File | Content |
|------|---------|
| `<prefix>_hooks_maxabs_<rank>_<world>.json` and `.npz` | Measurements of one rank. The `.npz` file holds the same data as numpy arrays. |
| `<prefix>_hooks_maxabs_<rank>_<world>_mod_list.json` | Modules measured by the rank. |
| `<prefix>_hooks_maxabs_<SCALE_METHOD>_<rank>_<world>.json` and `.npz` | Scales computed in the QUANTIZE phase, for example `MAXABS_HW`. |

A world size 1 file set (after `--unify-to-tp 1`) uses rank `0` and world `1`, for example `inc_output_hooks_maxabs_0_1.json`.
