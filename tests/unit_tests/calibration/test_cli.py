# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from vllm_gaudi.calibration import cli
from vllm_gaudi.calibration.args import CalibrationArgs


def parse(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def test_run_defaults_match_calibration_args():
    args = cli.calibration_args(parse("run", "org/Model", "-o", "out"))
    assert args == CalibrationArgs(model="org/Model", output_dir="out")


def test_run_flags_round_trip():
    ns = parse("run", "m", "-o", "out", "--tasks", "gsm8k", "wikitext", "--limit", "16", "--num-fewshot", "0",
               "--max-gen-toks", "64", "--include-path", "tasks", "--no-chat-template", "--no-fewshot-as-multiturn",
               "--tp", "4", "--expert-parallel", "--batch-size", "8", "--max-num-seqs", "16", "--max-model-len", "2048",
               "--enforce-eager", "--gpu-memory-utilization", "0.5", "--trust-remote-code", "--dtype", "float16",
               "--max-images", "2", "--image-max-side", "0", "--engine-arg", "block_size=256", "--engine-arg",
               'hf_overrides={"a": 1}', "--env", "VLLM_SKIP_WARMUP=false", "--phases", "quantize", "--quantize-eval",
               "full", "--smoke-limit", "4", "--no-postprocess", "--unify-to-tp", "2", "--scale-method", "maxabs_pow2",
               "--scale-format", "CONST", "--blocklist", "lm_head", "mlp", "--allowlist", "proj", "--input-backoff",
               "0.25", "--weight-backoff", "0.5", "--device-for-scales", "GAUDI2", "--measure-exclude", "NONE",
               "--dynamic-quantization", "--quantize-vision-tower", "--measure-config", "m.json", "--quant-config",
               "q.json", "--multi-node", "--quant-config-buffer", "buf.json", "--device", "g2", "--modality", "text",
               "--dry-run", "--keep-logs")
    args = cli.calibration_args(ns)
    assert args.tasks == ["gsm8k", "wikitext"]
    assert (args.limit, args.num_fewshot, args.max_gen_toks, args.include_path) == (16, 0, 64, "tasks")
    assert not args.apply_chat_template and not args.fewshot_as_multiturn
    assert (args.tp, args.expert_parallel, args.batch_size, args.max_num_seqs, args.max_model_len) == (4, True, 8, 16,
                                                                                                       2048)
    assert args.enforce_eager and args.trust_remote_code and args.multi_node and args.dry_run and args.keep_logs
    assert (args.gpu_memory_utilization, args.dtype, args.max_images, args.image_max_side) == (0.5, "float16", 2, 0)
    assert args.engine_args == {"block_size": 256, "hf_overrides": {"a": 1}}
    assert args.env == {"VLLM_SKIP_WARMUP": "false"}
    assert (args.phases, args.quantize_eval, args.smoke_limit, args.postprocess) == (("quantize", ), "full", 4, False)
    assert (args.unify_to_tp, args.scale_method, args.scale_format) == (2, "maxabs_pow2", "CONST")
    assert (args.blocklist, args.allowlist, args.quantize_vision_tower) == (["lm_head", "mlp"], ["proj"], True)
    opts = args.quant_options
    assert (opts.input_backoff, opts.weight_backoff, opts.device_for_scales, opts.measure_exclude,
            opts.dynamic_quantization) == (0.25, 0.5, "GAUDI2", "NONE", True)
    assert (args.measure_config, args.quant_config, args.quant_config_buffer) == ("m.json", "q.json", "buf.json")
    assert (args.device, args.modality) == ("g2", "text")
    args.validate()


@pytest.mark.parametrize("argv", [
    ("run", ),
    ("run", "m"),
    ("run", "m", "-o", "out", "--device", "g1"),
    ("run", "m", "-o", "out", "--batch-size", "0"),
    ("run", "m", "-o", "out", "--phases", "warmup"),
    ("run", "m", "-o", "out", "--limit", "many"),
    ("unify", "-m", "dir"),
    ("expand", "-w", "4"),
    (),
])
def test_invalid_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        parse(*argv)
    assert exc.value.code == 2


def test_help_exits_0_and_hides_internal_phase(capsys):
    with pytest.raises(SystemExit) as exc:
        parse("--help")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("run", "unify", "expand", "postprocess", "detect", "print-config", "list-tasks"):
        assert command in out
    assert "_phase" not in out


def test_malformed_engine_arg_is_an_error(capsys):
    assert cli.main(["run", "m", "-o", "out", "--engine-arg", "no_equals_sign"]) == 1


def test_validation_error_exits_1(tmp_path):
    assert cli.main(["run", "m", "-o", str(tmp_path), "--tp", "2", "--unify-to-tp", "3"]) == 1


def test_list_tasks_defaults(capsys):
    assert cli.main(["list-tasks", "--modality", "multimodal"]) == 0
    assert capsys.readouterr().out.split() == ["mmmu_val"]


def test_unify_subcommand(tmp_path):
    from .measurement_data import PREFIX, write_measure_run

    write_measure_run(tmp_path, world=2)
    assert cli.main(["unify", "-m", str(tmp_path), "-r", "1"]) == 0
    unified = json.loads((tmp_path / f"{PREFIX}_0_1.json").read_text())
    assert unified["LocalRank"] == -1


def test_print_config_aligns_user_config_like_run(tmp_path, monkeypatch, capsys):
    from vllm_gaudi.calibration import detect
    from vllm_gaudi.calibration.detect import ModelInfo

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(detect, "detect_model", lambda model, trust_remote_code: ModelInfo(model_type="llama"))
    (tmp_path / "measure.json").write_text(json.dumps({"mode": "MEASURE", "dump_stats_path": "custom/inc_output"}))
    assert cli.main(["print-config", "Org/My-Model", "-o", "out", "--device", "g3", "--measure-config",
                     "measure.json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    quant = printed[str(tmp_path / "out" / "my-model" / "maxabs_quant_g3.json")]
    assert quant["mode"] == "QUANTIZE"
    assert quant["dump_stats_path"] == str(tmp_path / "custom" / "inc_output")
