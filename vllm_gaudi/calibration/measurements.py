# SPDX-License-Identifier: Apache-2.0
"""Discovery, loading and writing of INC measurement files.

INC names its outputs after ``dump_stats_path``::

    <dump>_hooks_<observer>_<rank>_<world>.{json,npz}            measurements
    <dump>_hooks_<observer>_<rank>_<world>_mod_list.json         measured module list
    <dump>_hooks_<observer>_<SCALE_METHOD>_<rank>_<world>.{json,npz}  scales

Every JSON file has a binary twin (``.npz``) holding the same nodes as numpy arrays.
"""

import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vllm_gaudi.calibration.layout import DEFAULT_OBSERVER

MEASURE = "measure"
SCALES = "scales"
MOD_LIST = "mod_list"

logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<rank>\d+)_(?P<world>\d+)(?P<mod_list>_mod_list)?\.(?P<ext>json|npz)$")


@dataclass(frozen=True)
class MeasurementFile:
    """One parsed INC output file.

    Attributes:
        path: File path.
        prefix: Name without the rank, world and extension, for example ``inc_output_hooks_maxabs``.
        rank: Rank that wrote the file.
        world: World size of the run that wrote the file.
        kind: :data:`MEASURE`, :data:`SCALES` or :data:`MOD_LIST`.
        ext: ``json`` or ``npz``.
        scale_method: Upper-case scale method of a scales file, for example ``MAXABS_HW``.
    """

    path: Path
    prefix: str
    rank: int
    world: int
    kind: str
    ext: str
    scale_method: str | None = None


def parse_measurement_filename(path: str | os.PathLike[str],
                               observer: str = DEFAULT_OBSERVER) -> MeasurementFile | None:
    """Parses an INC output file name, returning None for unrelated files."""
    path = Path(path)
    match = _FILENAME_RE.match(path.name)
    if match is None:
        return None
    prefix, ext = match["prefix"], match["ext"]
    rank, world = int(match["rank"]), int(match["world"])
    hooks = f"_hooks_{observer}"
    if match["mod_list"]:
        if ext != "json" or not prefix.endswith(hooks):
            return None
        return MeasurementFile(path, prefix, rank, world, MOD_LIST, ext)
    if prefix.endswith(hooks):
        return MeasurementFile(path, prefix, rank, world, MEASURE, ext)
    scales = re.match(rf"^.+{re.escape(hooks)}_(?P<method>[A-Z][A-Z0-9_]*)$", prefix)
    if scales is not None:
        return MeasurementFile(path, prefix, rank, world, SCALES, ext, scales["method"])
    return None


def list_measurement_files(directory: str | os.PathLike[str],
                           observer: str = DEFAULT_OBSERVER) -> list[MeasurementFile]:
    """Returns the INC output files in a directory, sorted by kind, prefix, world, rank and extension."""
    files = []
    for entry in sorted(Path(directory).iterdir()):
        if entry.is_file():
            parsed = parse_measurement_filename(entry, observer)
            if parsed is not None:
                files.append(parsed)
    return sorted(files, key=lambda f: (f.kind, f.prefix, f.world, f.rank, f.ext))


def select(files: Iterable[MeasurementFile],
           *,
           kind: str | None = None,
           world: int | None = None,
           ext: str | None = "json") -> list[MeasurementFile]:
    """Filters parsed files by kind, world size and extension; None means any."""
    return [
        f for f in files
        if (kind is None or f.kind == kind) and (world is None or f.world == world) and (ext is None or f.ext == ext)
    ]


def remove_scales(directory: str | os.PathLike[str],
                  measure_prefixes: Iterable[str],
                  world: int,
                  observer: str = DEFAULT_OBSERVER) -> list[Path]:
    """Removes the scale files of one world size derived from the given measurements.

    INC reuses existing scale files and computes scales only for the modules missing from them,
    so scales left from earlier measurements must go when the measurements of that world size
    are rewritten.

    Returns:
        The removed files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    prefixes = tuple(f"{prefix}_" for prefix in measure_prefixes)
    stale = [
        f.path for f in select(list_measurement_files(directory, observer), kind=SCALES, world=world, ext=None)
        if f.prefix.startswith(prefixes)
    ]
    for path in stale:
        path.unlink()
    if stale:
        logger.info("Removed %d scale files of world size %d from %s", len(stale), world, directory)
    return stale


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _nodes_to_arrays(nodes: dict[str, Any]) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for name, node in nodes.items():
        layer: dict[str, Any] = {"inputs": [np.array(x) for x in node["inputs"]]}
        if node.get("outputs") is not None:
            layer["outputs"] = [np.array(x) for x in node["outputs"]]
        params = node.get("params")
        if params is not None and params.get("weight") is not None:
            layer["params"] = {"weight": np.array(params["weight"])}
        layers[name] = layer
    return layers


def _replace_atomic(path: Path, write: Any, mode: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, mode) as handle:
            write(handle)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_measurement(json_path: Path,
                      data: dict[str, Any],
                      *,
                      indent: int | None = 4,
                      keep_global_rank: bool = False) -> None:
    """Writes a measurement JSON and its ``.npz`` twin, each through an atomic rename.

    Args:
        json_path: Destination ``.json`` path; the ``.npz`` goes next to it.
        data: Measurement with ``GlobalRank``, ``LocalRank``, ``Mode`` and ``Nodes``.
        indent: JSON indentation; None writes compact JSON.
        keep_global_rank: Copy ``GlobalRank`` into the npz instead of None.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    _replace_atomic(json_path, lambda handle: json.dump(data, handle, indent=indent), "w")
    npz_data = {
        "GlobalRank": data.get("GlobalRank") if keep_global_rank else None,
        "LocalRank": data["LocalRank"],
        "Mode": data["Mode"],
        "Nodes": _nodes_to_arrays(data["Nodes"]),
    }
    _replace_atomic(json_path.with_suffix(".npz"), lambda handle: np.savez(handle, npz_data), "wb")


def load_npz(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Loads a measurement ``.npz`` written by INC or :func:`write_measurement`."""
    with np.load(path, allow_pickle=True) as archive:
        return archive["arr_0"].item()


# MoE node naming, for example model.layers.3.mlp.experts.moe_op (the fused op) and
# model.layers.3.mlp.experts.moe_op.w13_list.0 (one expert).
def is_moe_experts(node_name: str) -> bool:
    return "moe" in node_name.lower() and (".w13_list" in node_name or ".w2_list" in node_name)


def is_fused_moe_op(node_name: str) -> bool:
    return "moe" in node_name.lower() and ".w13_list" not in node_name and ".w2_list" not in node_name


def fused_moe_ops(nodes: dict[str, Any]) -> set[str]:
    """Returns the fused MoE op nodes, those with per-expert child nodes.

    A name test alone would also match plain layers such as Mixtral's ``block_sparse_moe.gate``.
    """
    experts = [name for name in nodes if is_moe_experts(name)]
    return {name for name in nodes if is_fused_moe_op(name) and any(e.startswith(name + ".") for e in experts)}


def split_expert_name(node_name: str) -> tuple[str, int]:
    """Splits an expert node name into its prefix and expert id."""
    prefix, _, expert_id = node_name.rpartition(".")
    if not expert_id.isdigit():
        raise ValueError(f"Expert node {node_name!r} does not end with an expert id")
    return prefix, int(expert_id)


def local_expert_num(nodes: dict[str, Any]) -> int:
    """Returns the highest expert id in the nodes plus one, or 0 for dense models."""
    return max((split_expert_name(name)[1] for name in nodes if is_moe_experts(name)), default=-1) + 1
