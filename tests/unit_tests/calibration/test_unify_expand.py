# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from vllm_gaudi.calibration.expand import expand_dir, expand_nodes
from vllm_gaudi.calibration.measurements import list_measurement_files, load_npz
from vllm_gaudi.calibration.unify import build_groups, unify_dir, unify_nodes

from .measurement_data import (FUSED, GATE, LINEAR, PREFIX, SCALES_PREFIX, measure_nodes, write_measure_run, write_rank)


def test_build_groups():
    assert build_groups(8, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert build_groups(2, 1) == [[0, 1]]
    for source, target in ((2, 2), (4, 3), (2, 0), (1, 1)):
        with pytest.raises(ValueError):
            build_groups(source, target)


def test_unify_measure_takes_per_channel_max():
    unified = unify_nodes([measure_nodes(0), measure_nodes(1)], scales=False)
    assert unified[LINEAR]["inputs"] == [[[2.0], [9.0]]]
    assert unified[LINEAR]["outputs"] == [[4.0]]
    assert unified[LINEAR]["params"]["weight"] == [[6.0], [8.0]]
    # Without expert parallelism the experts are max-merged like any other node.
    assert unified[f"{FUSED}.w13_list.1"]["inputs"] == [[[101]]]
    assert len(unified[FUSED]["outputs"]) == 3


def test_unify_measure_expert_parallel_concatenates_experts():
    ranks = [measure_nodes(0), measure_nodes(1)]
    unified = unify_nodes(ranks, scales=False, use_ep=True)
    assert unified[FUSED]["outputs"] == [[20.0], [0], [1], [100], [101]]
    assert unified[FUSED]["inputs"] == [[[2.0]]]
    assert [unified[f"{FUSED}.w13_list.{e}"]["inputs"] for e in range(4)] == [[[[0]]], [[[1]]], [[[100]]], [[[101]]]]
    # The router gate has "moe" in its name but no experts; it is merged like a plain layer.
    assert unified[GATE]["outputs"] == [[2.5]]
    assert ranks[0] == measure_nodes(0)


def test_unify_scales_uses_whole_value_max():
    ranks = [{
        LINEAR: {
            "inputs": [1.0, 5.0],
            "outputs": [2.0],
            "params": {
                "weight": [3.0, 1.0]
            }
        }
    }, {
        LINEAR: {
            "inputs": [4.0, 0.5],
            "outputs": [1.0],
            "params": {
                "weight": [2.0, 9.0]
            }
        }
    }]
    unified = unify_nodes(ranks, scales=True)
    assert unified[LINEAR] == {"inputs": [4.0, 5.0], "outputs": [2.0], "params": {"weight": [3.0, 1.0]}}


def test_unify_scales_expert_parallel_extends_fused_inputs():
    expert = f"{FUSED}.w13_list.0"
    ranks = [{
        FUSED: {
            "inputs": [1.0, 7.0]
        },
        expert: {
            "inputs": [7.0]
        }
    }, {
        FUSED: {
            "inputs": [3.0, 8.0]
        },
        expert: {
            "inputs": [8.0]
        }
    }]
    unified = unify_nodes(ranks, scales=True, use_ep=True)
    assert unified[FUSED]["inputs"] == [3.0, 7.0, 8.0]
    assert unified[f"{FUSED}.w13_list.1"] == {"inputs": [8.0]}


def test_unify_dir_tp2_to_tp1(tmp_path):
    write_measure_run(tmp_path, world=2)
    # INC scale files hold a list of floats per node and a scalar weight scale.
    write_rank(tmp_path, SCALES_PREFIX, 0, 2, {LINEAR: {"inputs": [1.0], "outputs": [0.5], "params": {"weight": 0.25}}})
    write_rank(tmp_path, SCALES_PREFIX, 1, 2, {LINEAR: {"inputs": [2.0], "outputs": [0.25], "params": {"weight": 0.5}}})
    written = unify_dir(tmp_path, 1, use_ep=True)
    assert [p.name for p in written] == [f"{PREFIX}_0_1.json", f"{SCALES_PREFIX}_0_1.json"]
    data = json.loads(written[0].read_text())
    assert data["LocalRank"] == -1
    assert len(data["Nodes"][FUSED]["outputs"]) == 5
    scales = load_npz(written[1].with_suffix(".npz"))["Nodes"][LINEAR]
    assert scales["inputs"][0].tolist() == 2.0
    assert scales["outputs"][0].tolist() == 0.5
    assert scales["params"]["weight"].tolist() == 0.5


def test_unify_dir_counts_only_the_source_world(tmp_path):
    # A previous unify left _0_1 files next to the TP4 run; the legacy script miscounted them.
    write_measure_run(tmp_path, world=4)
    write_rank(tmp_path, PREFIX, 0, 1, measure_nodes(0))
    written = unify_dir(tmp_path, 2, skip_scales=True)
    assert [p.name for p in written] == [f"{PREFIX}_0_2.json", f"{PREFIX}_1_2.json"]
    assert json.loads(written[1].read_text())["LocalRank"] == 1


def test_unify_dir_errors(tmp_path):
    with pytest.raises(ValueError, match="mod_list"):
        unify_dir(tmp_path, 1)
    write_measure_run(tmp_path, world=2)
    (tmp_path / f"{PREFIX}_1_2.json").unlink()
    with pytest.raises(ValueError, match=r"ranks \[1\]"):
        unify_dir(tmp_path, 1)


def test_expand_nodes_slices_experts():
    unified = unify_nodes([measure_nodes(0), measure_nodes(1)], scales=False, use_ep=True)
    assert expand_nodes(unified, 1, 2)[FUSED]["outputs"] == [[20.0], [100], [101]]
    assert expand_nodes(unified, 3, 4)[FUSED]["outputs"] == [[20.0], [101]]
    assert len(unified[FUSED]["outputs"]) == 5
    with pytest.raises(ValueError, match="evenly"):
        expand_nodes(unified, 0, 3)
    with pytest.raises(ValueError, match="no MoE experts"):
        expand_nodes({LINEAR: measure_nodes(0)[LINEAR]}, 0, 2)


def test_expand_dir(tmp_path):
    write_measure_run(tmp_path, world=2)
    unify_dir(tmp_path, 1, use_ep=True)
    written = expand_dir(tmp_path, 4)
    assert [p.name for p in written] == [f"{PREFIX}_{r}_4.json" for r in range(4)]
    for rank, path in enumerate(written):
        data = json.loads(path.read_text())
        assert data["LocalRank"] == rank
        assert len(data["Nodes"][FUSED]["outputs"]) == 2
        assert load_npz(path.with_suffix(".npz"))["LocalRank"] == rank
    assert {f.world for f in list_measurement_files(tmp_path)} == {1, 2, 4}


def test_expand_dir_removes_stale_target_scales(tmp_path):
    write_measure_run(tmp_path, world=2)
    unify_dir(tmp_path, 1, use_ep=True)
    stale = write_rank(tmp_path, SCALES_PREFIX, 3, 4, {LINEAR: {"inputs": [9.0]}})
    other_world = write_rank(tmp_path, SCALES_PREFIX, 0, 2, {LINEAR: {"inputs": [1.0]}})
    expand_dir(tmp_path, 4)
    assert not stale.exists() and not stale.with_suffix(".npz").exists()
    assert other_world.exists()


@pytest.mark.parametrize("skip_scales", [False, True])
def test_unify_dir_removes_stale_target_scales(tmp_path, skip_scales):
    # No source scales (or --skip-scales): old target scales must not survive next to new measurements.
    write_measure_run(tmp_path, world=2)
    stale = write_rank(tmp_path, SCALES_PREFIX, 0, 1, {LINEAR: {"inputs": [9.0]}})
    unify_dir(tmp_path, 1, skip_scales=skip_scales)
    assert not stale.exists() and not stale.with_suffix(".npz").exists()
    assert (tmp_path / f"{PREFIX}_0_1.json").is_file()


def test_expand_dir_errors(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        expand_dir(tmp_path, 1)
    with pytest.raises(ValueError, match="unify"):
        expand_dir(tmp_path, 2)
