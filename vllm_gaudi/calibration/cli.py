# SPDX-License-Identifier: Apache-2.0
"""Command line interface of ``vllm-gaudi-calibrate``.

Subcommands:
    run           Calibrate a model end to end.
    unify         Merge per-rank measurements to a smaller world size.
    expand        Split a world size 1 MoE measurement across expert parallel ranks.
    postprocess   Fix the KV cache inputs of attention matmuls in measurements.
    detect        Print the device type and the model facts the presets use.
    print-config  Print the INC configs ``run`` would write, without running anything.
    list-tasks    List the lm-eval tasks.
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

from vllm_gaudi.calibration import lmeval
from vllm_gaudi.calibration.args import (MODALITIES, QUANTIZE_EVAL_MODES, CalibrationArgs, parse_batch_size,
                                         parse_kv_list, parse_phases)
from vllm_gaudi.calibration.inc_config import QuantOptions
from vllm_gaudi.calibration.layout import DEFAULT_OBSERVER, SUPPORTED_DEVICES

logger = logging.getLogger(__name__)

PROG = "vllm-gaudi-calibrate"


def _argtype(parse: Any, name: str) -> Any:

    def convert(value: str) -> Any:
        try:
            return parse(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    convert.__name__ = name
    return convert


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    """Options shared by ``run`` and ``print-config``: what decides the INC configs."""
    inc = parser.add_argument_group("INC configuration")
    inc.add_argument("--device", choices=SUPPORTED_DEVICES, help="Device type; detected from the HPU by default.")
    inc.add_argument("--modality", choices=MODALITIES, help="Override the detected modality.")
    inc.add_argument("--trust-remote-code", action="store_true", help="Trust custom model code.")
    inc.add_argument("--scale-method", help="INC scale_method (default maxabs_hw).")
    inc.add_argument("--scale-format", help="INC scale_format; overrides the preset, e.g. CONST or scalar.")
    inc.add_argument("--blocklist",
                     nargs="+",
                     default=[],
                     metavar="NAME",
                     help="Module name patterns to keep in high precision, added to the preset's.")
    inc.add_argument("--allowlist", nargs="+", default=[], metavar="NAME", help="INC allowlist module names.")
    inc.add_argument("--quantize-vision-tower",
                     action="store_true",
                     help="Quantize the vision tower of multimodal models; it stays in BF16 by default.")
    inc.add_argument("--input-backoff", type=float, help="INC scale_params input_backoff.")
    inc.add_argument("--weight-backoff", type=float, help="INC scale_params weight_backoff.")
    inc.add_argument("--device-for-scales", help="INC device_for_scales, e.g. GAUDI2 for Gaudi 2 compatible scales.")
    inc.add_argument("--measure-exclude", help="INC measure_exclude, e.g. NONE or OUTPUT.")
    inc.add_argument("--dynamic-quantization", action="store_true", default=None, help="INC dynamic_quantization.")
    inc.add_argument("--measure-config", help="Use this INC measure JSON instead of generating one.")
    inc.add_argument("--quant-config", help="Use this INC quant JSON instead of generating one.")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Local model directory or Hugging Face model ID.")
    parser.add_argument("-o", "--output-dir", required=True, help="Root output directory.")
    _add_model_options(parser)

    ev = parser.add_argument_group("lm-eval")
    ev.add_argument("--tasks",
                    nargs="+",
                    metavar="TASK",
                    help="lm-eval tasks, groups or tags fed to the model (default: pile_10k gsm8k for text models, "
                    "mmmu_val for multimodal ones).")
    ev.add_argument("--limit", type=int, default=lmeval.DEFAULT_LIMIT, help="Samples per task (default %(default)s).")
    ev.add_argument("--num-fewshot", type=int, help="Few-shot examples; each task's default if unset.")
    ev.add_argument("--max-gen-toks",
                    type=int,
                    default=lmeval.DEFAULT_MAX_GEN_TOKS,
                    help="Generation budget for tasks that set none (default %(default)s).")
    ev.add_argument("--include-path", help="Directory with custom lm-eval task YAML files.")
    ev.add_argument("--no-chat-template",
                    dest="apply_chat_template",
                    action="store_false",
                    help="Do not format prompts with the chat template.")
    ev.add_argument("--no-fewshot-as-multiturn",
                    dest="fewshot_as_multiturn",
                    action="store_false",
                    help="Put few-shot examples in one user turn.")

    eng = parser.add_argument_group("engine")
    eng.add_argument("--tp", type=int, default=1, help="Tensor parallel size (default %(default)s).")
    eng.add_argument("--expert-parallel",
                     action="store_true",
                     default=None,
                     help="Enable expert parallelism; also selects the EP rule in --unify-to-tp.")
    eng.add_argument("--batch-size",
                     type=_argtype(parse_batch_size, "batch size"),
                     default="auto",
                     help="lm-eval batch size, auto or N (default %(default)s).")
    eng.add_argument("--max-num-seqs", type=int, help="vLLM max_num_seqs (preset default 32).")
    eng.add_argument("--max-model-len", type=int, help="vLLM max_model_len (preset default 4096, 8192 multimodal).")
    eng.add_argument("--enforce-eager", action="store_true", help="Run without HPU graphs.")
    eng.add_argument("--gpu-memory-utilization", type=float, help="vLLM gpu_memory_utilization.")
    eng.add_argument("--dtype", default="bfloat16", help="Model dtype (default %(default)s).")
    eng.add_argument("--max-images", type=int, default=1, help="Images per prompt, multimodal only (default 1).")
    eng.add_argument("--image-max-side",
                     type=int,
                     default=lmeval.DEFAULT_IMAGE_MAX_SIDE,
                     metavar="PIXELS",
                     help="Resize images so the longest side is at most PIXELS, multimodal only; 0 keeps the "
                     "original size (default %(default)s).")
    eng.add_argument("--engine-arg",
                     dest="engine_args",
                     action="append",
                     default=[],
                     metavar="KEY=VALUE",
                     help="Extra vllm.LLM argument; VALUE is parsed as JSON when possible. Repeatable.")
    eng.add_argument("--env",
                     action="append",
                     default=[],
                     metavar="KEY=VALUE",
                     help="Environment variable for the engine phases. Repeatable.")

    flow = parser.add_argument_group("flow")
    flow.add_argument("--phases",
                      type=_argtype(parse_phases, "phases"),
                      default=("measure", "quantize"),
                      help="Phases to run, a subset of measure,quantize (default both).")
    flow.add_argument("--quantize-eval",
                      choices=QUANTIZE_EVAL_MODES,
                      default="smoke",
                      help="Evaluation in the quantize phase: none, a short smoke run of the first task, or the full "
                      "task set (default %(default)s).")
    flow.add_argument("--smoke-limit",
                      type=int,
                      default=lmeval.DEFAULT_SMOKE_LIMIT,
                      help="Samples of the smoke evaluation (default %(default)s).")
    flow.add_argument("--no-postprocess",
                      dest="postprocess",
                      action="store_false",
                      help="Skip the KV cache input fix of the measurements.")
    flow.add_argument("--unify-to-tp", type=int, metavar="N", help="Unify the measurements to world size N.")
    flow.add_argument("--expand-to-ep",
                      type=int,
                      metavar="N",
                      help="Expand a MoE measurement to N expert parallel ranks.")

    env = parser.add_argument_group("environment")
    env.add_argument("--multi-node", action="store_true", help="Run the engine on a Ray cluster.")
    env.add_argument("--quant-config-buffer",
                     help="Shared file that every node's QUANT_CONFIG points to, for clusters that do not forward it.")
    env.add_argument("--dry-run",
                     action="store_true",
                     help="Write the configs and the manifest without loading the model.")
    env.add_argument("--keep-logs", action="store_true", help="Keep the phase logs after a successful run.")


def build_parser() -> argparse.ArgumentParser:
    """Returns the argument parser of every subcommand."""
    parser = argparse.ArgumentParser(prog=PROG, description="FP8 calibration of vLLM models on Intel Gaudi.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = sub.add_parser("run", help="Calibrate a model end to end.")
    _add_run_options(run)

    unify = sub.add_parser("unify", help="Merge per-rank measurements to a smaller world size.")
    unify.add_argument("-m", "--measurements", required=True, help="Directory with the measurement files.")
    unify.add_argument("-r", "--rank", type=int, required=True, help="Target world size.")
    unify.add_argument("-o", "--out", help="Output directory (default: in place).")
    unify.add_argument("--ep", action="store_true", help="Measurements were taken with expert parallelism.")
    unify.add_argument("--skip-scales", action="store_true", help="Unify the measure files only.")

    expand = sub.add_parser("expand", help="Expand a world size 1 MoE measurement to expert parallel ranks.")
    expand.add_argument("-m", "--measurements", required=True, help="Directory with the world size 1 measurement.")
    expand.add_argument("-w", "--world-size", type=int, required=True, help="Target expert parallel world size.")
    expand.add_argument("-o", "--out", help="Output directory (default: in place).")

    post = sub.add_parser("postprocess", help="Fix the KV cache inputs of attention matmuls in measurements.")
    post.add_argument("-m", "--measurements", required=True, help="Directory with the measurement files.")
    post.add_argument("-o", "--out", help="Output directory (default: in place).")

    detect = sub.add_parser("detect", help="Print the device type and the detected model facts.")
    detect.add_argument("model", help="Local model directory or Hugging Face model ID.")
    detect.add_argument("--trust-remote-code", action="store_true", help="Trust custom model code.")
    detect.add_argument("--no-device", action="store_true", help="Skip the HPU device query.")
    detect.add_argument("--json", action="store_true", help="Print JSON.")

    cfg = sub.add_parser("print-config", help="Print the INC configs run would write.")
    cfg.add_argument("model", help="Local model directory or Hugging Face model ID.")
    cfg.add_argument("-o", "--output-dir", default=".", help="Root output directory the paths refer to.")
    _add_model_options(cfg)

    tasks = sub.add_parser("list-tasks", help="List lm-eval tasks.")
    tasks.add_argument("--modality",
                       choices=(*MODALITIES, "all"),
                       default="all",
                       help="Show the default tasks of one modality instead of every task.")
    tasks.add_argument("--include-path", help="Directory with custom lm-eval task YAML files.")

    phase = sub.add_parser("_phase")
    phase.add_argument("name")
    phase.add_argument("--spec", required=True)
    # Keeps the internal subcommand out of the help listing.
    sub._choices_actions = [a for a in sub._choices_actions if a.dest != "_phase"]
    return parser


def _quant_options(ns: argparse.Namespace) -> QuantOptions:
    return QuantOptions(input_backoff=ns.input_backoff,
                        weight_backoff=ns.weight_backoff,
                        device_for_scales=ns.device_for_scales,
                        measure_exclude=ns.measure_exclude,
                        dynamic_quantization=ns.dynamic_quantization)


def calibration_args(ns: argparse.Namespace) -> CalibrationArgs:
    """Converts parsed ``run`` options to :class:`CalibrationArgs`.

    Raises:
        ValueError: On a malformed ``--engine-arg`` or ``--env``.
    """
    return CalibrationArgs(model=ns.model,
                           output_dir=ns.output_dir,
                           tasks=ns.tasks,
                           limit=ns.limit,
                           num_fewshot=ns.num_fewshot,
                           max_gen_toks=ns.max_gen_toks,
                           include_path=ns.include_path,
                           apply_chat_template=ns.apply_chat_template,
                           fewshot_as_multiturn=ns.fewshot_as_multiturn,
                           tp=ns.tp,
                           expert_parallel=ns.expert_parallel,
                           batch_size=ns.batch_size,
                           max_num_seqs=ns.max_num_seqs,
                           max_model_len=ns.max_model_len,
                           enforce_eager=ns.enforce_eager,
                           gpu_memory_utilization=ns.gpu_memory_utilization,
                           trust_remote_code=ns.trust_remote_code,
                           dtype=ns.dtype,
                           max_images=ns.max_images,
                           image_max_side=ns.image_max_side,
                           engine_args=parse_kv_list(ns.engine_args, parse_values=True),
                           env=parse_kv_list(ns.env, parse_values=False),
                           phases=ns.phases,
                           quantize_eval=ns.quantize_eval,
                           smoke_limit=ns.smoke_limit,
                           postprocess=ns.postprocess,
                           unify_to_tp=ns.unify_to_tp,
                           expand_to_ep=ns.expand_to_ep,
                           scale_method=ns.scale_method,
                           scale_format=ns.scale_format,
                           blocklist=ns.blocklist,
                           allowlist=ns.allowlist,
                           quantize_vision_tower=ns.quantize_vision_tower,
                           quant_options=_quant_options(ns),
                           measure_config=ns.measure_config,
                           quant_config=ns.quant_config,
                           multi_node=ns.multi_node,
                           quant_config_buffer=ns.quant_config_buffer,
                           device=ns.device,
                           modality=ns.modality,
                           dry_run=ns.dry_run,
                           keep_logs=ns.keep_logs)


def _emit(data: Any) -> None:
    sys.stdout.write(json.dumps(data, indent=2) + "\n")


def _cmd_run(ns: argparse.Namespace) -> int:
    from vllm_gaudi.calibration.orchestrator import run_calibration

    manifest = run_calibration(calibration_args(ns))
    if manifest.get("serve_command"):
        sys.stdout.write(manifest["serve_command"] + "\n")
    return 0


def _cmd_unify(ns: argparse.Namespace) -> int:
    from vllm_gaudi.calibration.unify import unify_dir

    written = unify_dir(ns.measurements, ns.rank, ns.out, use_ep=ns.ep, skip_scales=ns.skip_scales)
    logger.info("Wrote %d files", len(written))
    return 0


def _cmd_expand(ns: argparse.Namespace) -> int:
    from vllm_gaudi.calibration.expand import expand_dir

    written = expand_dir(ns.measurements, ns.world_size, ns.out)
    logger.info("Wrote %d files", len(written))
    return 0


def _cmd_postprocess(ns: argparse.Namespace) -> int:
    from vllm_gaudi.calibration.postprocess import postprocess_dir

    changes = postprocess_dir(ns.measurements, ns.out, observer=DEFAULT_OBSERVER)
    logger.info("Fixed %d KV cache inputs in %d files", sum(changes.values()), len(changes))
    return 0


def _cmd_detect(ns: argparse.Namespace) -> int:
    from vllm_gaudi.calibration.phases import phase_detect

    result = phase_detect({
        "model": ns.model,
        "trust_remote_code": ns.trust_remote_code,
        "detect_device": not ns.no_device
    })
    if ns.json:
        _emit(result)
    else:
        lines = [f"device: {result['device']}"] if "device" in result else []
        lines += [f"{key}: {value}" for key, value in result["model"].items()]
        sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _cmd_print_config(ns: argparse.Namespace) -> int:
    from vllm_gaudi.calibration.detect import ModelInfo, detect_device, detect_model
    from vllm_gaudi.calibration.inc_config import resolve_configs
    from vllm_gaudi.calibration.layout import OutputLayout
    from vllm_gaudi.calibration.presets import resolve_preset

    device = ns.device or detect_device()
    try:
        info = detect_model(ns.model, ns.trust_remote_code)
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("Model detection failed (%s); selecting the preset by model name", exc)
        info = ModelInfo()
    if ns.modality is not None:
        info = dataclasses.replace(info, is_multimodal=ns.modality == "multimodal")
    layout = OutputLayout.create(ns.output_dir, ns.model, device)
    preset = resolve_preset(info,
                            layout.model_name,
                            extra_blocklist=ns.blocklist,
                            allowlist=ns.allowlist,
                            scale_method=ns.scale_method,
                            scale_format=ns.scale_format,
                            quantize_vision_tower=ns.quantize_vision_tower)
    measure, quant = resolve_configs(preset,
                                     layout,
                                     _quant_options(ns),
                                     measure_config=ns.measure_config,
                                     quant_config=ns.quant_config,
                                     base_dir=os.getcwd())
    _emit({
        "model": info.to_dict(),
        "preset": preset.to_dict(),
        str(layout.measure_config): measure,
        str(layout.quant_config): quant
    })
    return 0


def _cmd_list_tasks(ns: argparse.Namespace) -> int:
    if ns.modality != "all":
        sys.stdout.write("\n".join(lmeval.default_tasks(ns.modality)) + "\n")
        return 0
    sys.stdout.write("\n".join(sorted(lmeval.available_tasks(ns.include_path))) + "\n")
    return 0


def _cmd_phase(ns: argparse.Namespace) -> int:
    from vllm_gaudi.calibration.phases import run_phase

    return run_phase(ns.name, ns.spec)


COMMANDS = {
    "run": _cmd_run,
    "unify": _cmd_unify,
    "expand": _cmd_expand,
    "postprocess": _cmd_postprocess,
    "detect": _cmd_detect,
    "print-config": _cmd_print_config,
    "list-tasks": _cmd_list_tasks,
    "_phase": _cmd_phase,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Runs ``vllm-gaudi-calibrate`` and returns the process exit code."""
    ns = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if ns.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from vllm_gaudi.calibration.orchestrator import CalibrationError

    try:
        return COMMANDS[ns.command](ns)
    except (CalibrationError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
