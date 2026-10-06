# Calibration Troubleshooting

When a phase fails, `vllm-gaudi-calibrate` exits with code `1` and names the phase and its log file. The phase logs are in `<output_dir>/<model_name>/<device>/logs/` (`measure.log`, `quantize.log`). They are kept when a run fails and removed after a successful run unless you pass `--keep-logs`. The phase output is also printed to the console. Add `-v` before the command (`vllm-gaudi-calibrate -v run ...`) for debug logging of the tool itself.

## Measurement Files Are Missing

```text
The measure phase did not write [...] in <output_dir>/<model_name>/<device>. INC writes them from finalize_calibration when the engine shuts down cleanly; check the measure log for an engine crash.
```

Before the MEASURE phase, the tool removes the files of an earlier run. After it, the tool checks that every rank wrote `inc_output_hooks_maxabs_<rank>_<world>.json`, `.npz`, and `_mod_list.json`. INC writes these files from `finalize_calibration`, which runs only when the engine shuts down cleanly. If they are missing:

- Check `measure.log` for an engine crash, an out-of-memory error, or a worker that died before shutdown. Rerun with `--keep-logs` if you need the logs of a run that later succeeds.
- Make sure that the measure config was used. `QUANT_CONFIG` is set by the tool for each phase; a `QUANT_CONFIG` exported in your shell is ignored. With `--measure-config`, check that `dump_stats_path` points to the directory named in the error.
- With `--multi-node`, the output directory must be on a file system shared by all nodes, and the workers must receive `QUANT_CONFIG` (see [Environment Variables on Ray](#environment-variables-on-ray)).

If the QUANTIZE phase leaves no scale files, the tool only logs a warning, because INC can compute the scales from the measurements when the model is served.

## Out of Memory

If you encounter the following error, the model does not fit on the cards of the run. Set a larger tensor parallel size, for example `--tp 8`:

```text
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed for size::939524096 (896)MB
```

If the model fits but the KV cache or the batch does not, lower `--max-model-len`, `--max-num-seqs`, or `--gpu-memory-utilization`, or set `--batch-size` to a small integer instead of `auto`.

A multimodal model can fail in the middle of a phase with the following error:

```text
RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_DEVMEM OOM: No enough memory for defragment.
```

On HPU, the KV cache takes most of the free device memory, and the vision encoder of a large image needs more than what is left. The tool resizes images to a longest side of 1280 pixels by default. Set a smaller `--image-max-side`, for example `--image-max-side 896`, or leave more memory outside the KV cache with `--gpu-memory-utilization 0.5`.

## Engine Crash on Loglikelihood Tasks

On some models, including DeepSeek-V2 and Qwen2.5-VL, the engine dies in the measure phase as soon as a loglikelihood task such as `pile_10k` starts:

```text
RuntimeError: Index tensor must have the same number of dimensions as input tensor
```

The same models fail outside calibration too, with `a and b must have same reduction dim` in a BF16 run. Loglikelihood tasks request prompt logprobs, and these models return the prefill hidden states in a shape that the HPU model runner indexed incorrectly. The issue is not specific to calibration and is fixed in [vllm-gaudi#1839](https://github.com/vllm-project/vllm-gaudi/pull/1839). On a version without the fix, calibrate on generation tasks only, for example `--tasks gsm8k`.

## Multimodal Measure Phase Stops on mmmu_val

On Qwen2.5-VL, the engine dies at about prompt 430 of the 900 prompts of `mmmu_val`, in the measure phase as well as in a plain lm-eval run:

```text
AssertionError: Expected number of multimodal embeddings to match number of input items: 1, but got len(mm_embeddings)=0 instead.
```

This is an issue of the HPU image path, not of the calibration. Until it is fixed, calibrate the language model on text tasks. Pass the vision tower names explicitly, because the text modality does not add them to the blocklist:

```bash
vllm-gaudi-calibrate run Qwen/Qwen2.5-VL-3B-Instruct -o ./calibration_output \
    --modality text --tasks pile_10k gsm8k --blocklist lm_head visual
```

`pile_10k` needs the fix described in [Engine Crash on Loglikelihood Tasks](#engine-crash-on-loglikelihood-tasks). Without it, pass `--tasks gsm8k`.

## Unknown lm-eval Task

```text
Unknown lm-eval tasks [...]; run 'vllm-gaudi-calibrate list-tasks' to see the available ones, or pass --include-path for custom task configs
```

The task names are validated in the phase process before the model is loaded. Run `vllm-gaudi-calibrate list-tasks` to list the tasks, groups, and tags lm-eval knows, or `vllm-gaudi-calibrate list-tasks --include-path <task_dir>` to include your own task YAML files. For custom tasks, pass the same `--include-path` to `run`. If lm-eval itself is missing, install the calibration dependencies with `pip install -r calibration/requirements.txt`.

## Missing Module ray

```text
ModuleNotFoundError: No module named 'ray'
```

lm-eval's multimodal vLLM model imports `ray`, so a multimodal run fails in the measure phase without it, as does `--multi-node`. vLLM does not install `ray` on HPU. Install the calibration dependencies with `pip install -r calibration/requirements.txt`.

## Model Without a Chat Template

If the tokenizer (or, for multimodal models, the processor) has no chat template, the tool logs a warning and runs the tasks on raw prompts:

```text
<model> has no chat template; running the tasks on raw prompts
```

This is expected for base models. If the template exists but cannot be detected, the tool keeps applying it, and lm-eval fails if it cannot. In that case, pass `--no-chat-template`. To keep the chat template but put few-shot examples in a single user turn, pass `--no-fewshot-as-multiturn`.

## Environment Variables on Ray

With `--multi-node`, the plugin forwards to the Ray workers only the variables whose names contain `HPU`, `RAY`, or `VLLM`, plus `GLOO_SOCKET_IFNAME`, `HCCL_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, and `QUANT_CONFIG`. Symptoms of a missing variable on the workers are missing measurement files from the ranks on other nodes, or workers that cannot reach the Hugging Face Hub or the model cache.

- Set other variables, such as `HF_HOME` or `HF_TOKEN`, on every node before `ray start`.
- If your setup does not forward `QUANT_CONFIG`, use `--quant-config-buffer` with a file on the shared file system, and export `QUANT_CONFIG` to that file on every node before `ray start`. See [Shared Config Buffer](advanced.md#shared-config-buffer).

## Multi-Node Run Hangs at Startup

If every rank logs its `world_size` with `backend=hccl` and the measure phase then makes no progress, HCCL cannot connect the cards of different nodes. `HCCL_SOCKET_IFNAME` selects only the interface for connection setup. The data goes over the Intel® Gaudi® scale-out ports, or over host NICs with `HCCL_OVER_OFI=1` and libfabric. A host or container network alone is not enough. Check the NIC ports as described in [Multi-Node Calibration with Ray](advanced.md#multi-node-calibration-with-ray).

## Other Errors

| Error | Cause and fix |
|-------|---------------|
| `<model> is an encoder-decoder model; INC calibration supports decoder-only models` | Encoder-decoder models are not supported. |
| `--expand-to-ep needs a MoE model; <model> has no routed experts` | `--expand-to-ep` applies only to MoE models. Use `--unify-to-tp` for dense models. |
| `--unify-to-tp N must be smaller than --tp M and divide it` | Choose a target that divides the measured tensor parallel size. |
| `Measurements of several world sizes [...] found` | The `unify` subcommand found module lists of several calibration runs in one directory. Run `unify` on a directory that holds one calibration run. |
| `<N> experts do not split evenly across <M> ranks` | Choose an expert parallel size that divides the number of experts. |
| `Unsupported HPU device ...` | Calibration supports Intel® Gaudi® 2 and Intel® Gaudi® 3. |
| `Cannot detect the HPU (...); pass --device g2 or --device g3` | The device is detected from the HPU by default. On a host without an HPU, pass `--device` to `print-config` or to `run --dry-run`. |
