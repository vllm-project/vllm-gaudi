# SPDX-License-Identifier: Apache-2.0
import json
from typing import Any

import pytest

from vllm_gaudi.calibration.detect import ModelInfo
from vllm_gaudi.calibration.inc_config import (QuantOptions, build_configs, load_user_config, resolve_configs,
                                               write_json_atomic)
from vllm_gaudi.calibration.layout import OutputLayout
from vllm_gaudi.calibration.presets import resolve_preset

DUMP = "/out/m/g3/inc_output"

# Golden configs copied from the legacy calibrate_model.sh, with "$1/$2/$3" replaced by DUMP's parent.
LEGACY_MEASURE_BLOCKLISTS = {
    "mixtral": ["self_attn", "lm_head"],
    "deepseek_v3": ["lm_head", "mlp\\.gate\\b"],
    "granitemoehybrid": ["mamba", "self_attn"],
    "llama": [],
}
LEGACY_QUANT_SCALE_FORMAT = {"mixtral": "CONST", "deepseek_v3": "scalar", "granitemoehybrid": None, "llama": None}


def _configs(model_type, options=None, **kwargs):
    preset = resolve_preset(ModelInfo(model_type=model_type), "m", **kwargs)
    layout = OutputLayout.create("/out", "m", "g3")
    return build_configs(preset, layout, options)


@pytest.mark.parametrize("model_type", sorted(LEGACY_MEASURE_BLOCKLISTS))
def test_measure_config_matches_legacy(model_type):
    measure, _ = _configs(model_type)
    expected = {
        "method": "HOOKS",
        "mode": "MEASURE",
        "observer": "maxabs",
        "allowlist": {
            "types": [],
            "names": []
        },
        "blocklist": {
            "types": [],
            "names": LEGACY_MEASURE_BLOCKLISTS[model_type]
        },
        "quantize_weight": False,
        "dump_stats_path": DUMP,
        "calibration_sample_interval": 1,
    }
    assert json.dumps(measure) == json.dumps(expected)


@pytest.mark.parametrize("model_type", sorted(LEGACY_QUANT_SCALE_FORMAT))
def test_quant_config_matches_legacy(model_type):
    _, quant = _configs(model_type)
    expected: dict[str, Any] = {"mode": "QUANTIZE", "observer": "maxabs", "scale_method": "maxabs_hw"}
    if LEGACY_QUANT_SCALE_FORMAT[model_type] is not None:
        expected["scale_format"] = LEGACY_QUANT_SCALE_FORMAT[model_type]
    expected["allowlist"] = {"types": [], "names": []}
    expected["blocklist"] = {"types": [], "names": LEGACY_MEASURE_BLOCKLISTS[model_type]}
    expected["dump_stats_path"] = DUMP
    assert json.dumps(quant) == json.dumps(expected)


def test_deepseek_blocklist_serializes_like_legacy_bash():
    measure, _ = _configs("deepseek_v3")
    assert '"mlp\\\\.gate\\\\b"' in json.dumps(measure)


def test_quant_options_written_only_when_set():
    options = QuantOptions(input_backoff=0.25, device_for_scales="GAUDI2", dynamic_quantization=False)
    _, quant = _configs("llama", options)
    assert quant["scale_params"] == {"input_backoff": 0.25}
    assert quant["device_for_scales"] == "GAUDI2"
    assert quant["dynamic_quantization"] is False
    assert "measure_exclude" not in quant
    assert list(quant)[-3:] == ["scale_params", "device_for_scales", "dynamic_quantization"]


def test_write_json_atomic(tmp_path):
    target = tmp_path / "a" / "cfg.json"
    write_json_atomic(target, {"mode": "MEASURE"}, fsync=True)
    assert json.loads(target.read_text()) == {"mode": "MEASURE"}
    assert [p.name for p in target.parent.iterdir()] == ["cfg.json"]


def test_load_user_config_resolves_relative_dump(tmp_path):
    cfg = tmp_path / "user.json"
    cfg.write_text(json.dumps({"mode": "MEASURE", "dump_stats_path": "./stats/inc_output"}))
    loaded = load_user_config(cfg, base_dir=tmp_path)
    assert loaded["dump_stats_path"] == str(tmp_path / "stats" / "inc_output")


@pytest.mark.parametrize("content", ['[]', '{"mode": "MEASURE"}'])
def test_load_user_config_rejects_invalid(tmp_path, content):
    cfg = tmp_path / "user.json"
    cfg.write_text(content)
    with pytest.raises(ValueError):
        load_user_config(cfg, base_dir=tmp_path)


def test_resolve_configs_aligns_dump_stats_path(tmp_path):
    preset = resolve_preset(ModelInfo(model_type="llama"), "m")
    layout = OutputLayout.create("/out", "m", "g3")
    user = tmp_path / "measure.json"
    user.write_text(json.dumps({"mode": "MEASURE", "dump_stats_path": "custom/inc_output"}))
    measure, quant = resolve_configs(preset, layout, measure_config=str(user), base_dir=tmp_path)
    assert quant["mode"] == "QUANTIZE"
    assert measure["dump_stats_path"] == quant["dump_stats_path"] == str(tmp_path / "custom" / "inc_output")
    generated = resolve_configs(preset, layout, base_dir=tmp_path)
    assert generated == build_configs(preset, layout)
    other = tmp_path / "quant.json"
    other.write_text(json.dumps({"mode": "QUANTIZE", "dump_stats_path": "/elsewhere/inc_output"}))
    with pytest.raises(ValueError, match="dump_stats_path"):
        resolve_configs(preset, layout, measure_config=str(user), quant_config=str(other), base_dir=tmp_path)
