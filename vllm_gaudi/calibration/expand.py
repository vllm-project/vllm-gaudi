# SPDX-License-Identifier: Apache-2.0
"""Expansion of a unified MoE measurement to more expert parallel ranks.

A MoE model measured once (or unified to world size 1) can be served with expert
parallelism M: each rank gets a copy of the measurement whose fused MoE op keeps only the
intermediate maxima of the experts that rank owns.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any

from vllm_gaudi.calibration.layout import DEFAULT_OBSERVER
from vllm_gaudi.calibration.measurements import (MEASURE, fused_moe_ops, list_measurement_files, load_json,
                                                 local_expert_num, select, write_measurement)

logger = logging.getLogger(__name__)


def expand_nodes(nodes: dict[str, Any], ep_rank: int, world: int) -> dict[str, Any]:
    """Returns the nodes of one expert parallel rank; the input is not modified.

    Raises:
        ValueError: If the nodes have no experts, the experts do not split evenly, or a fused
            MoE op does not hold one intermediate maximum per expert.
    """
    total_experts = local_expert_num(nodes)
    if total_experts == 0:
        raise ValueError("The measurement has no MoE experts to expand")
    if total_experts % world:
        raise ValueError(f"{total_experts} experts do not split evenly across {world} ranks")
    local = total_experts // world
    start = ep_rank * local

    expanded = copy.deepcopy(nodes)
    for name in sorted(fused_moe_ops(expanded)):
        node = expanded[name]
        outputs = node.get("outputs")
        if outputs is None or len(outputs) - 1 != total_experts:
            found = None if outputs is None else len(outputs) - 1
            raise ValueError(f"{name}: expected {total_experts} expert intermediate maxima, found {found}")
        node["outputs"] = [outputs[0], *outputs[1 + start:1 + start + local]]
    return expanded


def expand_dir(measurements_dir: str | os.PathLike[str],
               target_world: int,
               out_dir: str | os.PathLike[str] | None = None,
               *,
               observer: str = DEFAULT_OBSERVER) -> list[Path]:
    """Expands the unified world size 1 measurement in a directory to ``target_world`` ranks.

    Args:
        measurements_dir: Directory holding ``<prefix>_0_1.json``.
        target_world: Number of expert parallel ranks.
        out_dir: Output directory; defaults to ``measurements_dir``.
        observer: INC observer name used in the file names.

    Returns:
        The JSON files written; each has an ``.npz`` twin.
    """
    if target_world < 2:
        raise ValueError(f"The target world size must be at least 2, got {target_world}")
    source = Path(measurements_dir)
    target = Path(out_dir) if out_dir is not None else source
    sources = [f for f in select(list_measurement_files(source, observer), kind=MEASURE, world=1) if f.rank == 0]
    if len(sources) != 1:
        raise ValueError(f"Expected exactly one world size 1 measurement in {source}, found "
                         f"{[f.path.name for f in sources]}; unify the measurements to world size 1 first")
    data = load_json(sources[0].path)

    written = []
    for ep_rank in range(target_world):
        rank_data = dict(data)
        rank_data["LocalRank"] = ep_rank
        rank_data["Nodes"] = expand_nodes(data["Nodes"], ep_rank, target_world)
        path = target / f"{sources[0].prefix}_{ep_rank}_{target_world}.json"
        write_measurement(path, rank_data)
        written.append(path)
    logger.info("Expanded %s to %d expert parallel ranks", sources[0].path.name, target_world)
    return written
