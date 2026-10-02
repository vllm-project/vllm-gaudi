# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from vllm_gaudi.calibration.measurements import load_npz
from vllm_gaudi.calibration.postprocess import fix_kv_cache_inputs, postprocess_dir

from .measurement_data import PREFIX, SCALES_PREFIX, write_rank


def attention(prefix: str, *, mla: bool = False) -> dict:
    impl = f"{prefix}.impl"
    nodes = {
        f"{impl}.matmul_qk": {
            "inputs": [[[1.0]], [[2.0]]]
        },
        f"{impl}.matmul_av": {
            "inputs": [[[3.0]], [[4.0]]]
        },
    }
    if mla:
        nodes[f"{impl}.latent_cache_k"] = {"inputs": [[[9.0]]]}
    else:
        nodes[f"{impl}.k_cache"] = {"inputs": [[[7.0]]]}
        nodes[f"{impl}.v_cache"] = {"inputs": [[[8.0]]]}
    return nodes


@pytest.mark.parametrize("prefix", ["model.layers.0.self_attn.attn", "language_model.model.layers.11.self_attn.attn"])
def test_fix_standard_attention(prefix):
    nodes = attention(prefix)
    assert fix_kv_cache_inputs(nodes) == 2
    assert nodes[f"{prefix}.impl.matmul_qk"]["inputs"] == [[[1.0]], [[7.0]]]
    assert nodes[f"{prefix}.impl.matmul_av"]["inputs"] == [[[3.0]], [[8.0]]]
    assert fix_kv_cache_inputs(nodes) == 0


def test_fix_mla_uses_latent_cache():
    prefix = "model.layers.3.self_attn.mla_attn.mla_attn"
    nodes = attention(prefix, mla=True)
    assert fix_kv_cache_inputs(nodes) == 2
    assert nodes[f"{prefix}.impl.matmul_qk"]["inputs"][1] == [[9.0]]
    assert nodes[f"{prefix}.impl.matmul_av"]["inputs"][1] == [[9.0]]


def test_fix_skips_matmul_without_cache():
    nodes = {"model.layers.0.self_attn.attn.impl.matmul_av": {"inputs": [[[3.0]], [[4.0]]]}}
    assert fix_kv_cache_inputs(nodes) == 0
    assert nodes["model.layers.0.self_attn.attn.impl.matmul_av"]["inputs"][1] == [[4.0]]


def test_fix_copies_instead_of_aliasing():
    prefix = "model.layers.0.self_attn.attn"
    nodes = attention(prefix)
    fix_kv_cache_inputs(nodes)
    nodes[f"{prefix}.impl.matmul_qk"]["inputs"][1][0][0] = -1.0
    assert nodes[f"{prefix}.impl.k_cache"]["inputs"][0] == [[7.0]]


def test_postprocess_dir(tmp_path):
    prefix = "model.layers.0.self_attn.attn"
    write_rank(tmp_path, PREFIX, 0, 1, attention(prefix))
    write_rank(tmp_path, SCALES_PREFIX, 0, 1, attention(prefix))
    write_rank(tmp_path, PREFIX, 0, 2, attention(prefix))
    out = tmp_path / "out"
    results = postprocess_dir(tmp_path, out, world=1)
    assert results == {f"{PREFIX}_0_1.json": 2, f"{SCALES_PREFIX}_0_1.json": 2}
    fixed = json.loads((out / f"{PREFIX}_0_1.json").read_text())
    assert fixed["Nodes"][f"{prefix}.impl.matmul_av"]["inputs"][1] == [[8.0]]
    npz = load_npz(out / f"{PREFIX}_0_1.npz")
    assert npz["LocalRank"] == 0
    assert npz["Nodes"][f"{prefix}.impl.matmul_qk"]["inputs"][1].tolist() == [[7.0]]


def test_postprocess_in_place_leaves_unchanged_files(tmp_path):
    path = write_rank(tmp_path, PREFIX, 0, 1, {"model.layers.0.mlp.down_proj": {"inputs": [[[1.0]]]}})
    before = path.read_text()
    assert postprocess_dir(tmp_path) == {path.name: 0}
    assert path.read_text() == before
