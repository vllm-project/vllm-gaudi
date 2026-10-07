# SPDX-License-Identifier: Apache-2.0
"""The ``run`` flow: detect, configure, measure, postprocess, quantize, unify, expand.

This process never imports vllm, torch or habana_frameworks. Every step that needs them
runs in a child interpreter (see :mod:`vllm_gaudi.calibration.phases`), because
``QUANT_CONFIG`` is read when the HPU model runner is imported and INC writes its
measurement files only when the engine shuts down.
"""

import dataclasses
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from vllm_gaudi.calibration import lmeval
from vllm_gaudi.calibration.args import CalibrationArgs
from vllm_gaudi.calibration.detect import MODEL_LOADING_ARGS, ModelInfo
from vllm_gaudi.calibration.expand import expand_dir
from vllm_gaudi.calibration.inc_config import resolve_configs, write_json_atomic
from vllm_gaudi.calibration.layout import DEFAULT_OBSERVER, OutputLayout
from vllm_gaudi.calibration.manifest import build_manifest, inventory, redact_args, redact_env
from vllm_gaudi.calibration.measurements import SCALES, list_measurement_files
from vllm_gaudi.calibration.postprocess import postprocess_dir
from vllm_gaudi.calibration.presets import ResolvedPreset, resolve_preset
from vllm_gaudi.calibration.unify import unify_dir

logger = logging.getLogger(__name__)

# Environment every engine phase gets unless --env overrides it. Warmup would feed the
# observers synthetic activations, and weight sharing breaks INC's module patching.
PHASE_ENV_DEFAULTS = {"VLLM_SKIP_WARMUP": "true", "PT_HPU_WEIGHT_SHARING": "0"}
TP_ENV_DEFAULTS = {"PT_HPU_ENABLE_LAZY_COLLECTIVES": "true"}
TERMINATE_TIMEOUT_S = 30


class CalibrationError(RuntimeError):
    """A calibration step failed; the message says which and where to look."""


class PhaseRunner(Protocol):

    def __call__(self, name: str, spec: dict[str, Any], env: dict[str, str], log_path: Path | None) -> dict[str, Any]:
        ...


def build_phase_env(base_env: Mapping[str, str], *, quant_config: str | None, tp: int, preset_env: Mapping[str, str],
                    user_env: Mapping[str, str]) -> dict[str, str]:
    """Returns the environment of a phase child.

    Precedence, lowest first: the inherited environment, the tool defaults, the preset,
    then ``--env``. The defaults and the preset apply only to the INC phases; ``--env`` also
    applies to detection. ``QUANT_CONFIG`` is always the phase's own config, or absent.
    """
    env = dict(base_env)
    env.pop("QUANT_CONFIG", None)
    if quant_config is not None:
        env.update(PHASE_ENV_DEFAULTS)
        if tp > 1:
            env.update(TP_ENV_DEFAULTS)
        env.update(preset_env)
    env.update(user_env)
    if quant_config is not None:
        env["QUANT_CONFIG"] = quant_config
    return env


def phase_env_overrides(env: Mapping[str, str], base_env: Mapping[str, str]) -> dict[str, str]:
    """Returns the variables a phase environment adds or changes relative to ``base_env``."""
    return {key: value for key, value in env.items() if base_env.get(key) != value}


def subprocess_runner(name: str, spec: dict[str, Any], env: dict[str, str], log_path: Path | None) -> dict[str, Any]:
    """Runs a phase in a child interpreter, mirroring its output to the console and ``log_path``.

    The child's cwd is a temporary directory, so INC's ``nc_workspace`` and other scratch
    files never land in the caller's directory.

    Raises:
        CalibrationError: If the child exits with a non-zero code.
    """
    with tempfile.TemporaryDirectory(prefix=f"vllm-gaudi-calibrate-{name}-") as workdir:
        spec = {**spec, "result_path": os.path.join(workdir, "result.json")}
        spec_path = os.path.join(workdir, "spec.json")
        Path(spec_path).write_text(json.dumps(spec), encoding="utf-8")
        # -u and a line-buffered log, so a tail -f of the log follows the phase as it runs.
        cmd = [sys.executable, "-u", "-m", "vllm_gaudi.calibration", "_phase", name, "--spec", spec_path]
        logger.info("Starting %s phase%s", name, f", log: {log_path}" if log_path else "")
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "w", encoding="utf-8", buffering=1) if log_path is not None else None  # noqa: SIM115
        try:
            with subprocess.Popen(cmd,
                                  cwd=workdir,
                                  env=env,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True,
                                  errors="replace",
                                  bufsize=1) as proc:
                try:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                        if log is not None:
                            log.write(line)
                    returncode = proc.wait()
                except KeyboardInterrupt:
                    proc.terminate()
                    try:
                        proc.wait(TERMINATE_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise
        finally:
            if log is not None:
                log.close()
        if returncode != 0:
            where = f"; see {log_path}" if log_path else ""
            raise CalibrationError(f"The {name} phase failed with exit code {returncode}{where}")
        return json.loads(Path(spec["result_path"]).read_text(encoding="utf-8"))


def _measurement_names(dump: str, observer: str, tp: int) -> list[Path]:
    names = []
    for rank in range(tp):
        stem = f"{dump}_hooks_{observer}_{rank}_{tp}"
        names += [Path(f"{stem}.json"), Path(f"{stem}.npz"), Path(f"{stem}_mod_list.json")]
    return names


def remove_previous_outputs(dump: str, observer: str, *, scales_only: bool) -> None:
    """Removes the INC files an earlier run left at ``dump``.

    INC QUANTIZE reuses existing scale files and computes scales only for the modules missing
    from them, so a rerun would keep the old scales. New measurements make every derived file
    stale, so before MEASURE all files go, before QUANTIZE only the scales.
    """
    directory = Path(dump).parent
    if not directory.is_dir():
        return
    prefix = f"{Path(dump).name}_hooks_"
    old = [
        f.path for f in list_measurement_files(directory, observer)
        if f.prefix.startswith(prefix) and (f.kind == SCALES or not scales_only)
    ]
    if old:
        logger.info("Removing %d files of an earlier run from %s", len(old), directory)
        for path in old:
            path.unlink()


def check_measurements(dump: str, observer: str, tp: int) -> None:
    """Verifies that the MEASURE phase wrote a measurement for every rank.

    Raises:
        CalibrationError: If a file is missing.
    """
    missing = [p for p in _measurement_names(dump, observer, tp) if not p.is_file()]
    if missing:
        directory = Path(dump).parent
        present = sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []
        raise CalibrationError(
            f"The measure phase did not write {[p.name for p in missing]} in {directory}. INC writes them from "
            "finalize_calibration when the engine shuts down cleanly; check the measure log for an engine crash. "
            f"Directory contents: {present}")


def check_scales(dump: str, observer: str, tp: int) -> None:
    """Warns when the QUANTIZE phase left no scale files; INC can still compute them when serving.

    Any scale method counts: INC names the files after the method it resolved, which can come from
    its own defaults or a per-op config, so the name is not predicted here.
    """
    directory = Path(dump).parent
    prefix = f"{Path(dump).name}_hooks_{observer}_"
    found = {(f.rank, f.ext)
             for f in list_measurement_files(directory, observer)
             if f.kind == SCALES and f.world == tp and f.prefix.startswith(prefix)} if directory.is_dir() else set()
    missing = [f"rank {rank} .{ext}" for rank in range(tp) for ext in ("json", "npz") if (rank, ext) not in found]
    if missing:
        logger.warning(
            "The quantize phase did not write scale files for %s in %s; INC will compute the scales when "
            "the model is served", missing, directory)


def _engine_args(args: CalibrationArgs, preset: ResolvedPreset) -> dict[str, Any]:
    engine: dict[str, Any] = dict(preset.engine_args)
    explicit = {
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_expert_parallel": args.expert_parallel,
    }
    engine.update({key: value for key, value in explicit.items() if value is not None})
    if args.enforce_eager:
        engine["enforce_eager"] = True
    return engine


def _expert_parallel(args: CalibrationArgs, preset: ResolvedPreset) -> bool:
    """Returns whether the engine runs with expert parallelism, ``--engine-arg`` included."""
    engine = {**_engine_args(args, preset), **args.engine_args}
    return bool(engine.get("enable_expert_parallel", False))


def _quantization(info: ModelInfo) -> str | None:
    """Returns the vLLM ``quantization`` argument; an already quantized checkpoint keeps its own method."""
    return None if info.quant_method else "inc"


def _model_args(args: CalibrationArgs, preset: ResolvedPreset, info: ModelInfo, phase: str) -> dict[str, Any]:
    engine = _engine_args(args, preset)
    if phase == "quantize":
        engine.update(preset.quantize_engine_args)
    return lmeval.build_model_args(model=args.model,
                                   phase=phase,
                                   engine_args=engine,
                                   tensor_parallel_size=args.tp,
                                   dtype=args.dtype,
                                   batch_size=args.batch_size,
                                   max_gen_toks=args.max_gen_toks,
                                   trust_remote_code=args.trust_remote_code,
                                   multimodal=info.is_multimodal,
                                   max_images=args.max_images,
                                   image_max_side=args.image_max_side,
                                   multi_node=args.multi_node,
                                   quantization=_quantization(info),
                                   user_engine_args=args.engine_args)


def _eval_spec(args: CalibrationArgs, tasks: list[str], limit: int) -> dict[str, Any]:
    return {
        "tasks": tasks,
        "limit": limit,
        "num_fewshot": args.num_fewshot,
        "apply_chat_template": args.apply_chat_template,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "include_path": args.include_path,
    }


def detect_model_info(args: CalibrationArgs, runner: PhaseRunner) -> tuple[str, ModelInfo]:
    """Runs the detect phase and applies ``--device`` and ``--modality``.

    The output directory does not exist yet, so the phase output goes to the console only.
    In ``--dry-run`` with ``--device`` a failed detection falls back to the model name.
    """
    spec = {
        "model": args.model,
        "trust_remote_code": args.trust_remote_code,
        "detect_device": args.device is None,
        "loading_args": {
            k: v
            for k, v in args.engine_args.items() if k in MODEL_LOADING_ARGS
        },
    }
    # --env applies here too, for example HF_TOKEN for a gated model; the preset is not known yet.
    env = build_phase_env(os.environ, quant_config=None, tp=1, preset_env={}, user_env=args.env)
    try:
        result = runner("detect", spec, env, None)
        info = ModelInfo.from_dict(result["model"])
        device = args.device or result["device"]
    except CalibrationError:
        if not (args.dry_run and args.device):
            raise
        logger.warning("Model detection failed; the dry run selects presets by model name only")
        info, device = ModelInfo(source="none"), args.device
    if args.modality is not None:
        info = dataclasses.replace(info, is_multimodal=args.modality == "multimodal")
    return device, info


def _resolve_local_paths(args: CalibrationArgs) -> None:
    # The phases run in a scratch cwd, so local paths are made absolute against the caller's cwd.
    if os.path.exists(args.model):
        args.model = os.path.abspath(args.model)
    if args.include_path is not None:
        args.include_path = os.path.abspath(args.include_path)
    tokenizer = args.engine_args.get("tokenizer")
    if isinstance(tokenizer, str) and os.path.exists(tokenizer):
        args.engine_args["tokenizer"] = os.path.abspath(tokenizer)
    # A relative download_dir would land in the scratch cwd, which is deleted after each phase.
    if isinstance(args.engine_args.get("download_dir"), str):
        args.engine_args["download_dir"] = os.path.abspath(args.engine_args["download_dir"])


def _resolve_chat_template(args: CalibrationArgs, info: ModelInfo) -> None:
    if args.apply_chat_template and info.has_chat_template is False:
        logger.warning("%s has no chat template; running the tasks on raw prompts", args.model)
        args.apply_chat_template = False
        args.fewshot_as_multiturn = False
    elif not args.apply_chat_template:
        args.fewshot_as_multiturn = False


def _write_configs(args: CalibrationArgs, preset: ResolvedPreset, layout: OutputLayout) -> tuple[dict, dict]:
    measure, quant = resolve_configs(preset,
                                     layout,
                                     args.quant_options,
                                     measure_config=args.measure_config,
                                     quant_config=args.quant_config,
                                     base_dir=os.getcwd())
    write_json_atomic(layout.measure_config, measure)
    write_json_atomic(layout.quant_config, quant)
    Path(measure["dump_stats_path"]).parent.mkdir(parents=True, exist_ok=True)
    return measure, quant


def _phase_config_path(args: CalibrationArgs, config: Mapping[str, Any], path: Path) -> str:
    """Returns the ``QUANT_CONFIG`` value for a phase, staging it in the shared buffer if one is set."""
    if args.quant_config_buffer is None:
        return str(path)
    buffer = Path(os.path.abspath(args.quant_config_buffer))
    write_json_atomic(buffer, config, fsync=True)
    if json.loads(buffer.read_text(encoding="utf-8")) != config:
        raise CalibrationError(f"{buffer} does not read back the config just written")
    return str(buffer)


def serve_command(args: CalibrationArgs,
                  layout: OutputLayout,
                  serve_world: int,
                  *,
                  quantization: str | None = "inc",
                  expert_parallel: bool = False) -> str:
    """Returns the command that serves the calibrated model.

    The measurement files of an expert parallel run hold the experts of each rank, so the
    model must be served with expert parallelism as well, unless it was unified to one card.
    """
    parts = [f"QUANT_CONFIG={shlex.quote(str(layout.quant_config))}", "vllm", "serve", shlex.quote(args.model)]
    if quantization is not None:
        parts += ["--quantization", quantization]
    parts += ["--kv-cache-dtype", "fp8_inc", "--tensor-parallel-size", str(serve_world)]
    if args.expand_to_ep is not None or (expert_parallel and serve_world > 1):
        parts.append("--enable-expert-parallel")
    if args.trust_remote_code:
        parts.append("--trust-remote-code")
    return " ".join(parts)


def run_calibration(args: CalibrationArgs, runner: PhaseRunner = subprocess_runner) -> dict[str, Any]:
    """Calibrates one model and returns the manifest that was written.

    Raises:
        CalibrationError: If a phase fails or produces no output.
        ValueError: If the options are inconsistent with the detected model.
    """
    args.validate()
    started = time.time()
    _resolve_local_paths(args)
    device, info = detect_model_info(args, runner)
    if info.is_encoder_decoder:
        raise ValueError(f"{args.model} is an encoder-decoder model; INC calibration supports decoder-only models")
    if args.expand_to_ep is not None and info.source != "none" and not info.is_moe:
        raise ValueError(f"--expand-to-ep needs a MoE model; {args.model} has no routed experts")
    if info.is_moe and args.tp > 1 and args.unify_to_tp is None:
        logger.info("MoE model measured with --tp %d; serving it with another world size needs --unify-to-tp", args.tp)
    _resolve_chat_template(args, info)

    layout = OutputLayout.create(args.output_dir, args.model, device)
    preset = resolve_preset(info,
                            layout.model_name,
                            extra_blocklist=args.blocklist,
                            allowlist=args.allowlist,
                            scale_method=args.scale_method,
                            scale_format=args.scale_format,
                            quantize_vision_tower=args.quantize_vision_tower)
    measure_cfg, quant_cfg = _write_configs(args, preset, layout)
    logger.info("Model %s: %s, preset %s, device %s", args.model, info.modality, preset.name, device)

    tasks = args.tasks or lmeval.default_tasks(info.modality)
    dump = measure_cfg["dump_stats_path"]
    observer = measure_cfg.get("observer", DEFAULT_OBSERVER)
    stats_dir = Path(dump).parent
    model_args = {phase: _model_args(args, preset, info, phase) for phase in args.phases}
    manifest_fields: dict[str, Any] = {
        "status": "dry-run" if args.dry_run else "ok",
        "model": args.model,
        "model_name": layout.model_name,
        "device": device,
        "model_info": info.to_dict(),
        "preset": preset.to_dict(),
        "tasks": tasks,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "apply_chat_template": args.apply_chat_template,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "tensor_parallel_size": args.tp,
        "stats_dir": str(stats_dir),
        "configs": {
            "measure": str(layout.measure_config),
            "quant": str(layout.quant_config)
        },
        "model_args": redact_args(model_args),
        "phases": {},
    }

    if args.dry_run:
        for phase in args.phases:
            cfg_path = layout.measure_config if phase == "measure" else layout.quant_config
            # Records the buffer path the phases would use, without overwriting the shared file.
            if args.quant_config_buffer is not None:
                cfg_path = Path(os.path.abspath(args.quant_config_buffer))
            env = build_phase_env(os.environ,
                                  quant_config=str(cfg_path),
                                  tp=args.tp,
                                  preset_env=preset.env,
                                  user_env=args.env)
            manifest_fields["phases"][phase] = {"env": redact_env(phase_env_overrides(env, os.environ))}
        return _finish(layout, manifest_fields, started, keep_logs=True)

    def spawn(phase: str, config: Mapping[str, Any], path: Path, evaluation: dict[str, Any] | None) -> None:
        env = build_phase_env(os.environ,
                              quant_config=_phase_config_path(args, config, path),
                              tp=args.tp,
                              preset_env=preset.env,
                              user_env=args.env)
        spec = {"model_args": model_args[phase], "multimodal": info.is_multimodal, "eval": evaluation}
        result = runner(phase, spec, env, layout.logs_dir / f"{phase}.log")
        manifest_fields["phases"][phase] = {
            "duration_s": result.get("duration_s"),
            "metrics": result.get("metrics"),
            "env": redact_env(phase_env_overrides(env, os.environ)),
        }

    if "measure" in args.phases:
        remove_previous_outputs(dump, observer, scales_only=False)
        spawn("measure", measure_cfg, layout.measure_config, _eval_spec(args, tasks, args.limit))
        check_measurements(dump, observer, args.tp)
        if args.postprocess:
            manifest_fields["postprocess"] = postprocess_dir(stats_dir, world=args.tp, observer=observer)

    if "quantize" in args.phases:
        evaluation = None
        if args.quantize_eval == "smoke":
            evaluation = _eval_spec(args, tasks[:1], min(args.smoke_limit, args.limit))
        elif args.quantize_eval == "full":
            evaluation = _eval_spec(args, tasks, args.limit)
        remove_previous_outputs(dump, observer, scales_only=True)
        spawn("quantize", quant_cfg, layout.quant_config, evaluation)
        check_scales(dump, observer, args.tp)

    serve_world = args.tp
    use_ep = _expert_parallel(args, preset)
    if args.unify_to_tp is not None:
        written = unify_dir(stats_dir, args.unify_to_tp, use_ep=use_ep, source_world=args.tp, observer=observer)
        manifest_fields["unify"] = {"world": args.unify_to_tp, "files": [p.name for p in written]}
        serve_world = args.unify_to_tp
    if args.expand_to_ep is not None:
        written = expand_dir(stats_dir, args.expand_to_ep, observer=observer)
        manifest_fields["expand"] = {"world": args.expand_to_ep, "files": [p.name for p in written]}
        serve_world = args.expand_to_ep

    manifest_fields["serve_command"] = serve_command(args,
                                                     layout,
                                                     serve_world,
                                                     quantization=_quantization(info),
                                                     expert_parallel=use_ep)
    manifest_fields["files"] = inventory(stats_dir)
    manifest = _finish(layout, manifest_fields, started, keep_logs=args.keep_logs)
    logger.info("Calibration finished. Serve the model with:\n  %s", manifest_fields["serve_command"])
    return manifest


def _finish(layout: OutputLayout, fields: dict[str, Any], started: float, *, keep_logs: bool) -> dict[str, Any]:
    fields["duration_s"] = round(time.time() - started, 3)
    manifest = build_manifest(**fields)
    write_json_atomic(layout.manifest, manifest)
    if not keep_logs:
        shutil.rmtree(layout.logs_dir, ignore_errors=True)
    logger.info("Wrote %s", layout.manifest)
    return manifest
