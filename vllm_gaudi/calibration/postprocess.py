# SPDX-License-Identifier: Apache-2.0
"""KV cache input fix-up for measurement files.

INC observes the attention matmuls with the freshly computed key and value, not with the
cached tensors the matmuls read during decode. The fix copies the cache module's input
range into the second input of ``matmul_qk`` and ``matmul_av``. Nodes are matched by
name suffix, so any module prefix works (``model.layers.*``,
``language_model.model.layers.*``) and MLA (``latent_cache_k``) needs no flag.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any

from vllm_gaudi.calibration.layout import DEFAULT_OBSERVER
from vllm_gaudi.calibration.measurements import (MEASURE, SCALES, list_measurement_files, load_json, select,
                                                 write_measurement)

logger = logging.getLogger(__name__)

# matmul suffix -> cache module suffixes, in order of preference.
KV_CACHE_SOURCES = {
    ".matmul_qk": (".k_cache", ".latent_cache_k"),
    ".matmul_av": (".v_cache", ".latent_cache_k"),
}


def fix_kv_cache_inputs(nodes: dict[str, Any]) -> int:
    """Copies cache input ranges into the attention matmul nodes, in place.

    Args:
        nodes: The ``Nodes`` mapping of one measurement or scales file.

    Returns:
        The number of matmul inputs changed.
    """
    changes = 0
    for name, node in nodes.items():
        for suffix, cache_suffixes in KV_CACHE_SOURCES.items():
            if not name.endswith(suffix):
                continue
            base = name[:-len(suffix)]
            cache = next((nodes[base + s] for s in cache_suffixes if base + s in nodes), None)
            inputs = node.get("inputs")
            if cache is None or not cache.get("inputs") or not inputs or len(inputs) < 2:
                continue
            if inputs[1] != cache["inputs"][0]:
                inputs[1] = copy.deepcopy(cache["inputs"][0])
                changes += 1
    return changes


def postprocess_file(json_path: Path, out_dir: Path | None = None) -> int:
    """Fixes one measurement or scales JSON and rewrites it with its ``.npz`` twin.

    The files are rewritten only when something changed or the output directory differs.

    Returns:
        The number of matmul inputs changed.
    """
    data = load_json(json_path)
    changes = fix_kv_cache_inputs(data["Nodes"])
    target = (out_dir or json_path.parent) / json_path.name
    if changes or target != json_path:
        write_measurement(target, data, indent=None, keep_global_rank=True)
    return changes


def postprocess_dir(measurements_dir: str | os.PathLike[str],
                    out_dir: str | os.PathLike[str] | None = None,
                    *,
                    world: int | None = None,
                    observer: str = DEFAULT_OBSERVER) -> dict[str, int]:
    """Fixes every measurement and scales JSON in a directory.

    Args:
        measurements_dir: Directory with INC outputs.
        out_dir: Output directory; defaults to rewriting in place.
        world: Process only files of this world size.
        observer: INC observer name used in the file names.

    Returns:
        Number of changes per file name.
    """
    source = Path(measurements_dir)
    target = Path(out_dir) if out_dir is not None else None
    files = list_measurement_files(source, observer)
    results = {}
    for kind in (MEASURE, SCALES):
        for parsed in select(files, kind=kind, world=world):
            results[parsed.path.name] = postprocess_file(parsed.path, target)
            logger.info("Postprocessed %s: %d KV cache inputs fixed", parsed.path.name, results[parsed.path.name])
    return results
