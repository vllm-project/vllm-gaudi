# SPDX-License-Identifier: Apache-2.0
import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest

from vllm_gaudi.calibration import orchestrator
from vllm_gaudi.calibration.args import CalibrationArgs
from vllm_gaudi.calibration.detect import ModelInfo
from vllm_gaudi.calibration.orchestrator import CalibrationError, build_phase_env, run_calibration

from .measurement_data import PREFIX, SCALES_PREFIX, measure_nodes, write_measure_run, write_rank

QWEN = ModelInfo(model_type="qwen2", architectures=("Qwen2ForCausalLM", ), has_chat_template=True, source="test")
MIXTRAL = ModelInfo(model_type="mixtral", is_moe=True, num_experts=8, has_chat_template=True, source="test")


class FakeRunner:
    """Plays the phase children: answers detect and writes the files INC would write."""

    def __init__(self, info: ModelInfo, *, write_measurements: bool = True):
        self.info = info
        self.write_measurements = write_measurements
        self.calls: list[tuple[str, dict, dict]] = []

    def __call__(self, name, spec, env, log_path):
        self.calls.append((name, spec, env))
        if name == "detect":
            return {"device": "g3", "model": self.info.to_dict()}
        config = json.loads(Path(env["QUANT_CONFIG"]).read_text())
        stats = Path(config["dump_stats_path"]).parent
        world = spec["model_args"]["tensor_parallel_size"]
        if name == "measure" and self.write_measurements:
            write_measure_run(stats, world)
        if name == "quantize":
            for rank in range(world):
                write_rank(stats, SCALES_PREFIX, rank, world, measure_nodes(rank))
        return {"metrics": {"gsm8k": {"exact_match": 0.5}} if spec["eval"] else None, "duration_s": 1.0}


def args(tmp_path, **kwargs) -> CalibrationArgs:
    return CalibrationArgs(model="Org/My-Model", output_dir=str(tmp_path), **kwargs)


def test_phase_env_sets_quant_config_and_precedence():
    base = {"QUANT_CONFIG": "/stale.json", "PATH": "/bin", "VLLM_SKIP_WARMUP": "false"}
    env = build_phase_env(base, quant_config="/m.json", tp=2, preset_env={"A": "preset"}, user_env={"A": "user"})
    assert env["QUANT_CONFIG"] == "/m.json"
    assert env["VLLM_SKIP_WARMUP"] == "true"
    assert env["PT_HPU_ENABLE_LAZY_COLLECTIVES"] == "true"
    assert env["A"] == "user"
    assert env["PATH"] == "/bin"
    assert "VLLM_HPU_FORCE_CHANNEL_FP8" not in env
    user = build_phase_env(base, quant_config="/m.json", tp=1, preset_env={}, user_env={"VLLM_SKIP_WARMUP": "false"})
    assert user["VLLM_SKIP_WARMUP"] == "false"
    assert "PT_HPU_ENABLE_LAZY_COLLECTIVES" not in user
    detect = build_phase_env(base, quant_config=None, tp=1, preset_env={}, user_env={})
    assert "QUANT_CONFIG" not in detect
    assert detect["VLLM_SKIP_WARMUP"] == "false"


def test_dry_run_writes_configs_and_manifest_without_engine(tmp_path):
    runner = FakeRunner(QWEN)
    manifest = run_calibration(args(tmp_path, device="g3", dry_run=True), runner)
    assert [name for name, _, _ in runner.calls] == ["detect"]
    assert runner.calls[0][1]["detect_device"] is False
    model_dir = tmp_path / "my-model"
    measure = json.loads((model_dir / "maxabs_measure_g3.json").read_text())
    assert measure["mode"] == "MEASURE"
    assert measure["dump_stats_path"] == str(model_dir / "g3" / "inc_output")
    assert (model_dir / "maxabs_quant_g3.json").is_file()
    assert json.loads((model_dir / "g3" / "calibration_manifest.json").read_text()) == manifest
    assert manifest["status"] == "dry-run"
    assert manifest["tasks"] == ["pile_10k", "gsm8k"]
    assert manifest["phases"]["measure"]["env"]["QUANT_CONFIG"] == str(model_dir / "maxabs_measure_g3.json")
    assert manifest["model_args"]["quantize"]["kv_cache_dtype"] == "fp8_inc"


def test_dry_run_survives_failed_detection(tmp_path):

    def failing(name, spec, env, log_path):
        raise CalibrationError("no vllm here")

    manifest = run_calibration(args(tmp_path, device="g2", dry_run=True), failing)
    assert manifest["model_info"]["source"] == "none"
    with pytest.raises(CalibrationError):
        run_calibration(args(tmp_path, dry_run=True), failing)


def test_full_flow_tp2_unify_to_1(tmp_path):
    runner = FakeRunner(QWEN)
    manifest = run_calibration(args(tmp_path, tp=2, unify_to_tp=1, tasks=["gsm8k", "wikitext"], limit=16), runner)
    names = [name for name, _, _ in runner.calls]
    assert names == ["detect", "measure", "quantize"]
    measure_spec, quantize_spec = runner.calls[1][1], runner.calls[2][1]
    assert measure_spec["eval"]["tasks"] == ["gsm8k", "wikitext"] and measure_spec["eval"]["limit"] == 16
    assert quantize_spec["eval"]["tasks"] == ["gsm8k"] and quantize_spec["eval"]["limit"] == 8
    stats = tmp_path / "my-model" / "g3"
    assert (stats / f"{PREFIX}_0_1.json").is_file()
    assert (stats / f"{SCALES_PREFIX}_0_1.json").is_file()
    assert manifest["status"] == "ok"
    assert manifest["unify"]["world"] == 1
    assert "--tensor-parallel-size 1" in manifest["serve_command"]
    assert manifest["phases"]["quantize"]["metrics"] == {"gsm8k": {"exact_match": 0.5}}
    assert f"{PREFIX}_1_2_mod_list.json" in manifest["files"]
    assert not (stats / "logs").exists()


def test_missing_measurements_is_a_hard_error(tmp_path):
    with pytest.raises(CalibrationError, match="finalize_calibration"):
        run_calibration(args(tmp_path), FakeRunner(QWEN, write_measurements=False))


def test_measurements_of_an_earlier_run_are_removed(tmp_path):
    stats = tmp_path / "my-model" / "g3"
    stats.mkdir(parents=True)
    write_measure_run(stats, 1)
    write_rank(stats, SCALES_PREFIX, 0, 1, measure_nodes(0))
    unrelated = stats / "notes.txt"
    unrelated.write_text("kept")
    with pytest.raises(CalibrationError, match="finalize_calibration"):
        run_calibration(args(tmp_path), FakeRunner(QWEN, write_measurements=False))
    assert [p.name for p in stats.iterdir()] == ["notes.txt"]


def test_quantize_only_run_removes_old_scales(tmp_path):
    stats = tmp_path / "my-model" / "g3"
    stats.mkdir(parents=True)
    write_measure_run(stats, 2)
    for world in (1, 2):
        for rank in range(world):
            write_rank(stats, SCALES_PREFIX, rank, world, {"stale": {}})
    run_calibration(args(tmp_path, tp=2, phases=("quantize", )), FakeRunner(QWEN))
    assert (stats / f"{PREFIX}_1_2.json").is_file()
    assert not (stats / f"{SCALES_PREFIX}_0_1.json").exists()
    scales = json.loads((stats / f"{SCALES_PREFIX}_0_2.json").read_text())
    assert "stale" not in scales["Nodes"]


def test_quantize_eval_none_and_full(tmp_path):
    runner = FakeRunner(QWEN)
    run_calibration(args(tmp_path, quantize_eval="none"), runner)
    assert runner.calls[-1][1]["eval"] is None
    runner = FakeRunner(QWEN)
    run_calibration(args(tmp_path, quantize_eval="full", limit=4), runner)
    assert runner.calls[-1][1]["eval"]["tasks"] == ["pile_10k", "gsm8k"]
    assert runner.calls[-1][1]["eval"]["limit"] == 4


def test_no_chat_template_disables_chat_formatting(tmp_path):
    runner = FakeRunner(ModelInfo(model_type="llama", has_chat_template=False, source="test"))
    run_calibration(args(tmp_path, phases=("measure", )), runner)
    evaluation = runner.calls[1][1]["eval"]
    assert evaluation["apply_chat_template"] is False and evaluation["fewshot_as_multiturn"] is False


def test_preset_and_user_env_reach_the_phase(tmp_path):
    runner = FakeRunner(MIXTRAL)
    manifest = run_calibration(args(tmp_path, env={"FOO": "bar"}, engine_args={"block_size": 256}), runner)
    _, spec, env = runner.calls[1]
    assert env["FOO"] == "bar"
    assert spec["model_args"]["block_size"] == 256
    assert manifest["preset"]["name"] == "mixtral"
    quant = json.loads((tmp_path / "my-model" / "maxabs_quant_g3.json").read_text())
    assert quant["scale_format"] == "CONST"


def test_engine_arg_expert_parallel_drives_unify_and_serve(tmp_path, monkeypatch):
    deepseek = ModelInfo(model_type="deepseek_v2", is_moe=True, num_experts=64, has_chat_template=True, source="test")
    seen = []
    monkeypatch.setattr(orchestrator, "unify_dir", lambda *a, **kw: seen.append(kw["use_ep"]) or [])
    manifest = run_calibration(args(tmp_path, tp=4, unify_to_tp=2), FakeRunner(deepseek))
    assert seen == [True]
    assert manifest["serve_command"].endswith("--tensor-parallel-size 2 --enable-expert-parallel")
    manifest = run_calibration(args(tmp_path, tp=4, unify_to_tp=2, engine_args={"enable_expert_parallel": False}),
                               FakeRunner(deepseek))
    assert seen == [True, False]
    assert "--enable-expert-parallel" not in manifest["serve_command"]
    manifest = run_calibration(args(tmp_path, tp=2, unify_to_tp=1), FakeRunner(deepseek))
    assert "--enable-expert-parallel" not in manifest["serve_command"]


def test_quantized_checkpoint_keeps_its_quantization(tmp_path):
    runner = FakeRunner(dataclasses.replace(QWEN, quant_method="fp8"))
    manifest = run_calibration(args(tmp_path), runner)
    assert all("quantization" not in spec["model_args"] for name, spec, _ in runner.calls if name != "detect")
    assert "--quantization" not in manifest["serve_command"]
    manifest = run_calibration(args(tmp_path), FakeRunner(QWEN))
    assert "--quantization inc" in manifest["serve_command"]


def test_user_config_dump_stats_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    custom = {"method": "HOOKS", "mode": "MEASURE", "observer": "maxabs", "dump_stats_path": "custom/inc_output"}
    measure = tmp_path / "measure.json"
    measure.write_text(json.dumps(custom))
    run_calibration(args(tmp_path / "out", device="g3", dry_run=True, measure_config=str(measure)), FakeRunner(QWEN))
    quant = json.loads((tmp_path / "out" / "my-model" / "maxabs_quant_g3.json").read_text())
    assert quant["mode"] == "QUANTIZE"
    assert quant["dump_stats_path"] == str(tmp_path / "custom" / "inc_output")
    manifest = json.loads((tmp_path / "out" / "my-model" / "g3" / "calibration_manifest.json").read_text())
    assert manifest["stats_dir"] == str(tmp_path / "custom")
    other = tmp_path / "quant.json"
    other.write_text(json.dumps({**custom, "mode": "QUANTIZE", "dump_stats_path": "/elsewhere/inc_output"}))
    with pytest.raises(ValueError, match="dump_stats_path"):
        run_calibration(
            args(tmp_path / "out", device="g3", dry_run=True, measure_config=str(measure), quant_config=str(other)),
            FakeRunner(QWEN))


def test_local_paths_are_resolved_against_the_caller_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "My-Model").mkdir(parents=True)
    runner = FakeRunner(QWEN)
    run_calibration(
        CalibrationArgs(model="models/My-Model", output_dir="out", include_path="tasks", phases=("measure", )), runner)
    model = str(tmp_path / "models" / "My-Model")
    detect, measure = runner.calls[0][1], runner.calls[1][1]
    assert detect["model"] == model
    assert measure["model_args"]["pretrained"] == model
    assert measure["eval"]["include_path"] == str(tmp_path / "tasks")
    assert (tmp_path / "out" / "my-model" / "maxabs_measure_g3.json").is_file()
    hub = FakeRunner(QWEN)
    run_calibration(args(tmp_path / "hub", phases=("measure", )), hub)
    assert hub.calls[0][1]["model"] == "Org/My-Model"


def test_model_rejections(tmp_path):
    with pytest.raises(ValueError, match="encoder-decoder"):
        run_calibration(args(tmp_path), FakeRunner(ModelInfo(is_encoder_decoder=True, source="test")))
    with pytest.raises(ValueError, match="MoE"):
        run_calibration(args(tmp_path, expand_to_ep=4), FakeRunner(QWEN))


def test_dry_run_records_the_quant_config_buffer(tmp_path):
    buffer = tmp_path / "shared" / "quant_config.json"
    manifest = run_calibration(
        args(tmp_path / "out", device="g3", dry_run=True, multi_node=True, quant_config_buffer=str(buffer)),
        FakeRunner(QWEN))
    for phase in ("measure", "quantize"):
        assert manifest["phases"][phase]["env"]["QUANT_CONFIG"] == str(buffer)
    assert not buffer.exists()


def test_quant_config_buffer_is_written_and_used(tmp_path):
    buffer = tmp_path / "shared" / "quant_config.json"
    buffer.parent.mkdir()
    runner = FakeRunner(QWEN)
    run_calibration(args(tmp_path / "out", multi_node=True, quant_config_buffer=str(buffer)), runner)
    assert runner.calls[1][2]["QUANT_CONFIG"] == str(buffer)
    assert runner.calls[1][1]["model_args"]["distributed_executor_backend"] == "ray"
    assert json.loads(buffer.read_text())["mode"] == "QUANTIZE"


def test_subprocess_runner_reports_child_failure(tmp_path):
    log = tmp_path / "logs" / "detect.log"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    with pytest.raises(CalibrationError, match="exit code"):
        orchestrator.subprocess_runner("bogus", {}, env, log)
    assert "Unknown phase" in log.read_text()
