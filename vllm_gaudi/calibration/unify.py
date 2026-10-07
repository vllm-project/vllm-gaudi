# SPDX-License-Identifier: Apache-2.0
"""Unification of per-rank measurements to a smaller world size.

A model measured with tensor parallelism N can be served with any M that divides N:
the measurements of each group of N/M consecutive ranks are merged by taking the
maximum. With expert parallelism the experts of the group are concatenated instead,
so expert ``e`` of rank ``r`` within a group becomes expert ``local_experts * r + e``.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any

from vllm_gaudi.calibration.layout import DEFAULT_OBSERVER
from vllm_gaudi.calibration.measurements import (MEASURE, MOD_LIST, SCALES, MeasurementFile, fused_moe_ops,
                                                 remove_scales, is_moe_experts, list_measurement_files, load_json,
                                                 local_expert_num, select, split_expert_name, write_measurement)

logger = logging.getLogger(__name__)


def build_groups(source_world: int, target_world: int) -> list[list[int]]:
    """Returns the source ranks merged into each target rank.

    Raises:
        ValueError: If the target is not smaller than the source or does not divide it.
    """
    if not 1 <= target_world < source_world or source_world % target_world:
        raise ValueError(f"Cannot unify world size {source_world} to {target_world}: the target must be smaller "
                         "than the measured world size and divide it")
    size = source_world // target_world
    return [[i * size + j for j in range(size)] for i in range(target_world)]


def detect_source_world(files: list[MeasurementFile]) -> int:
    """Returns the world size of the measurement run, read from the ``*_mod_list.json`` files.

    Raises:
        ValueError: If there is no module list or the directory mixes several measurement runs.
    """
    worlds = sorted({f.world for f in files if f.kind == MOD_LIST})
    if not worlds:
        raise ValueError("No *_mod_list.json file found; is this an INC measurement directory?")
    if len(worlds) > 1:
        raise ValueError(f"Measurements of several world sizes {worlds} found; keep one calibration run per directory")
    return worlds[0]


def _merge_expert(experts: dict[str, Any], node_name: str, node: dict[str, Any], idx: int, expert_num: int) -> None:
    if node_name not in experts:
        experts[node_name] = node
        return
    prefix, local_id = split_expert_name(node_name)
    new_name = f"{prefix}.{expert_num * idx + local_id}"
    if new_name in experts:
        raise ValueError(f"Expert {new_name} appears in more than one rank")
    experts[new_name] = node


def unify_nodes(rank_nodes: list[dict[str, Any]], *, scales: bool, use_ep: bool = False) -> dict[str, Any]:
    """Merges the ``Nodes`` of one group of ranks.

    Args:
        rank_nodes: ``Nodes`` of each source rank in the group, in rank order.
        scales: Whether the nodes come from scales files rather than measurement files.
        use_ep: Apply the expert parallelism rules to MoE nodes.

    Returns:
        The merged nodes; the inputs are not modified.
    """
    unified = copy.deepcopy(rank_nodes[0])
    experts: dict[str, Any] = {}
    expert_num = local_expert_num(unified) if use_ep else -1
    fused_ops = fused_moe_ops(unified) if use_ep else set()

    for node_name, node in unified.items():
        max_inputs = node["inputs"]
        max_outputs = node.get("outputs")
        params = node.get("params")
        max_weight = params.get("weight") if params is not None else None
        fused_moe = node_name in fused_ops

        for idx, nodes in enumerate(rank_nodes):
            other = nodes[node_name]
            if use_ep and is_moe_experts(node_name):
                _merge_expert(experts, node_name, node if idx == 0 else other, idx, expert_num)
                continue
            if scales:
                if fused_moe and idx > 0:
                    # Input 0 is the hidden states; the rest are per-expert intermediate maxima.
                    max_inputs[0] = max(other["inputs"][0], max_inputs[0])
                    max_inputs.extend(other["inputs"][1:])
                else:
                    for i in range(len(max_inputs)):
                        max_inputs[i] = max(other["inputs"][i], max_inputs[i])
                if max_outputs is not None:
                    max_outputs = max(other["outputs"], max_outputs)
                if max_weight is not None:
                    max_weight = max(other["params"]["weight"], max_weight)
            else:
                for i in range(len(max_inputs)):
                    for j in range(len(max_inputs[i])):
                        max_inputs[i][j][0] = max(other["inputs"][i][j][0], max_inputs[i][j][0])
                if max_outputs is not None:
                    if fused_moe and idx > 0:
                        max_outputs[0][0] = max(other["outputs"][0][0], max_outputs[0][0])
                        max_outputs.extend(other["outputs"][1:])
                    else:
                        for i in range(len(max_outputs)):
                            max_outputs[i][0] = max(other["outputs"][i][0], max_outputs[i][0])
                if max_weight is not None:
                    for i in range(len(max_weight)):
                        max_weight[i][0] = max(other["params"]["weight"][i][0], max_weight[i][0])

        if max_outputs is not None:
            node["outputs"] = max_outputs
        if max_weight is not None:
            node["params"]["weight"] = max_weight

    if use_ep:
        unified.update(experts)
    return unified


def _unify_kind(files: list[MeasurementFile], groups: list[list[int]], source_world: int, out_dir: Path, *, kind: str,
                use_ep: bool) -> list[Path]:
    written = []
    candidates = select(files, kind=kind, world=source_world)
    for prefix in sorted({f.prefix for f in candidates}):
        by_rank = {f.rank: f for f in candidates if f.prefix == prefix}
        missing = sorted(set(range(source_world)) - set(by_rank))
        if missing:
            raise ValueError(f"{prefix}: missing {kind} files for ranks {missing} of world size {source_world}")
        target_world = len(groups)
        for group_index, group in enumerate(groups):
            sources = [load_json(by_rank[rank].path) for rank in group]
            data = copy.deepcopy(sources[0])
            data["LocalRank"] = group_index if target_world != 1 else -1
            data["Nodes"] = unify_nodes([s["Nodes"] for s in sources], scales=kind == SCALES, use_ep=use_ep)
            target = out_dir / f"{prefix}_{group_index}_{target_world}.json"
            write_measurement(target, data)
            written.append(target)
            logger.info("Unified ranks %s into %s", group, target.name)
    return written


def unify_dir(measurements_dir: str | os.PathLike[str],
              target_world: int,
              out_dir: str | os.PathLike[str] | None = None,
              *,
              use_ep: bool = False,
              skip_scales: bool = False,
              source_world: int | None = None,
              observer: str = DEFAULT_OBSERVER) -> list[Path]:
    """Unifies the measurements in a directory to a smaller world size.

    Args:
        measurements_dir: Directory with the per-rank INC outputs.
        target_world: World size to unify to.
        out_dir: Output directory; defaults to ``measurements_dir``.
        use_ep: Apply the expert parallelism rules to MoE nodes.
        skip_scales: Unify only the measurements, not the scales.
        source_world: World size of the measurement run; read from the module lists when None.
        observer: INC observer name used in the file names.

    Returns:
        The JSON files written; each has an ``.npz`` twin.
    """
    source = Path(measurements_dir)
    target = Path(out_dir) if out_dir is not None else source
    files = list_measurement_files(source, observer)
    world = source_world if source_world is not None else detect_source_world(files)
    groups = build_groups(world, target_world)
    logger.info("Unifying world size %d to %d, rank groups %s", world, target_world, groups)

    # The target world's scales are written again below or, without source scales, left to INC.
    remove_scales(target, {f.prefix for f in select(files, kind=MEASURE, world=world)}, target_world, observer)
    written = _unify_kind(files, groups, world, target, kind=MEASURE, use_ep=use_ep)
    if not written:
        raise ValueError(f"No measurement files of world size {world} found in {source}")
    if not skip_scales:
        scales = _unify_kind(files, groups, world, target, kind=SCALES, use_ep=use_ep)
        if not scales:
            logger.warning("No scale files of world size %d found in %s; unified only the measurements", world, source)
        written.extend(scales)
    return written
