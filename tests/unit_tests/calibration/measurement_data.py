# SPDX-License-Identifier: Apache-2.0
"""Synthetic INC measurement files shaped like real maxabs outputs."""

import json
from pathlib import Path

PREFIX = "inc_output_hooks_maxabs"
SCALES_PREFIX = "inc_output_hooks_maxabs_MAXABS_HW"
FUSED = "model.layers.1.mlp.experts.moe_op"
GATE = "model.layers.1.block_sparse_moe.gate"
LINEAR = "model.layers.0.mlp.down_proj"


def measure_nodes(rank: int, local_experts: int = 2) -> dict:
    """Per-channel layout: inputs[i][j][0], outputs[i][0], weight[i][0]."""
    base = float(rank + 1)
    nodes = {
        LINEAR: {
            "inputs": [[[base], [10 - base]]],
            "outputs": [[base * 2]],
            "params": {
                "weight": [[base * 3], [9 - base]]
            }
        },
        GATE: {
            "inputs": [[[base]]],
            "outputs": [[base + 0.5]]
        },
        FUSED: {
            "inputs": [[[base]]],
            "outputs": [[base * 10], *[[100 * rank + e] for e in range(local_experts)]]
        },
    }
    for e in range(local_experts):
        nodes[f"{FUSED}.w13_list.{e}"] = {"inputs": [[[100 * rank + e]]]}
    return nodes


def write_rank(directory: Path, prefix: str, rank: int, world: int, nodes: dict) -> Path:
    path = directory / f"{prefix}_{rank}_{world}.json"
    path.write_text(json.dumps({"GlobalRank": None, "LocalRank": rank, "Mode": "DynamicRange", "Nodes": nodes}))
    (directory / f"{prefix}_{rank}_{world}.npz").write_bytes(b"")
    return path


def write_measure_run(directory: Path, world: int, local_experts: int = 2) -> None:
    for rank in range(world):
        write_rank(directory, PREFIX, rank, world, measure_nodes(rank, local_experts))
        (directory / f"{PREFIX}_{rank}_{world}_mod_list.json").write_text(json.dumps([LINEAR, GATE, FUSED]))
