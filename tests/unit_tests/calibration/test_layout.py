# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from vllm_gaudi.calibration.layout import OutputLayout, model_dir_name


@pytest.mark.parametrize(("model", "expected"), [
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5-0.5b-instruct"),
    ("/models/Llama-3.1-8B/", "llama-3.1-8b"),
    ("mixtral", "mixtral"),
])
def test_model_dir_name(model, expected):
    assert model_dir_name(model) == expected


def test_model_dir_name_rejects_empty():
    with pytest.raises(ValueError):
        model_dir_name("/")


def test_layout_matches_legacy_paths(tmp_path):
    layout = OutputLayout.create(tmp_path, "Qwen/Qwen2.5-0.5B-Instruct", "g3")
    root = tmp_path / "qwen2.5-0.5b-instruct"
    assert layout.measure_config == root / "maxabs_measure_g3.json"
    assert layout.quant_config == root / "maxabs_quant_g3.json"
    assert layout.stats_dir == root / "g3"
    assert layout.dump_stats_path == str(root / "g3" / "inc_output")
    assert layout.manifest == root / "g3" / "calibration_manifest.json"
    assert layout.logs_dir == root / "g3" / "logs"


def test_layout_resolves_relative_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    layout = OutputLayout.create("out", "m", "g2")
    assert layout.output_dir == tmp_path / "out"


def test_layout_validation():
    with pytest.raises(ValueError, match="device"):
        OutputLayout(Path("/abs"), "m", "g1")
    with pytest.raises(ValueError, match="absolute"):
        OutputLayout(Path("rel"), "m", "g3")
