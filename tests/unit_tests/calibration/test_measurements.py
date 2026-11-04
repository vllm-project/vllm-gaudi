# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from vllm_gaudi.calibration.measurements import (MEASURE, MOD_LIST, SCALES, fused_moe_ops, list_measurement_files,
                                                 load_npz, local_expert_num, parse_measurement_filename,
                                                 split_expert_name, write_measurement)

from .measurement_data import FUSED, GATE, measure_nodes


@pytest.mark.parametrize(("name", "kind", "rank", "world", "method"), [
    ("inc_output_hooks_maxabs_0_1.json", MEASURE, 0, 1, None),
    ("inc_output_hooks_maxabs_3_8.npz", MEASURE, 3, 8, None),
    ("inc_output_hooks_maxabs_1_2_mod_list.json", MOD_LIST, 1, 2, None),
    ("inc_output_hooks_maxabs_MAXABS_HW_0_1.json", SCALES, 0, 1, "MAXABS_HW"),
    ("my_dump_hooks_maxabs_MAXABS_POW2_7_16.npz", SCALES, 7, 16, "MAXABS_POW2"),
])
def test_parse_measurement_filename(name, kind, rank, world, method):
    parsed = parse_measurement_filename(name)
    assert parsed is not None
    assert (parsed.kind, parsed.rank, parsed.world, parsed.scale_method) == (kind, rank, world, method)


@pytest.mark.parametrize("name", [
    "calibration_manifest.json", "inc_output_hooks_maxabs.json", "inc_output_hooks_maxabs_0_1_mod_list.npz",
    "inc_output_hooks_minmax_0_1.json", "inc_output_hooks_maxabs_lower_0_1.json", "maxabs_quant_g3.json"
])
def test_parse_ignores_unrelated_files(name):
    assert parse_measurement_filename(name) is None


def test_list_measurement_files_sorted(tmp_path):
    for name in ("inc_output_hooks_maxabs_1_2.json", "inc_output_hooks_maxabs_0_2.json", "notes.txt",
                 "inc_output_hooks_maxabs_0_2_mod_list.json"):
        (tmp_path / name).write_text("{}")
    names = [f.path.name for f in list_measurement_files(tmp_path)]
    assert names == [
        "inc_output_hooks_maxabs_0_2.json", "inc_output_hooks_maxabs_1_2.json",
        "inc_output_hooks_maxabs_0_2_mod_list.json"
    ]


def test_write_measurement_round_trip(tmp_path):
    data = {"GlobalRank": 5, "LocalRank": 1, "Mode": "DynamicRange", "Nodes": measure_nodes(0)}
    target = tmp_path / "out" / "inc_output_hooks_maxabs_1_2.json"
    write_measurement(target, data, keep_global_rank=True)
    assert json.loads(target.read_text()) == data
    npz = load_npz(target.with_suffix(".npz"))
    assert (npz["GlobalRank"], npz["LocalRank"], npz["Mode"]) == (5, 1, "DynamicRange")
    assert npz["Nodes"][GATE]["outputs"][0].tolist() == [1.5]
    assert "params" not in npz["Nodes"][GATE]
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name, target.with_suffix(".npz").name]

    write_measurement(target, data)
    assert load_npz(target.with_suffix(".npz"))["GlobalRank"] is None


def test_moe_helpers():
    nodes = measure_nodes(0, local_experts=3)
    assert fused_moe_ops(nodes) == {FUSED}
    assert local_expert_num(nodes) == 3
    assert local_expert_num({GATE: {}}) == 0
    assert split_expert_name(f"{FUSED}.w2_list.12") == (f"{FUSED}.w2_list", 12)
    with pytest.raises(ValueError):
        split_expert_name(f"{FUSED}.w2_list")
