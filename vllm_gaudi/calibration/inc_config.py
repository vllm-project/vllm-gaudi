# SPDX-License-Identifier: Apache-2.0
"""Generation of the Intel Neural Compressor ``QUANT_CONFIG`` files.

The emitted keys and their order match what the legacy ``calibrate_model.sh`` wrote, so
configs produced by either tool are interchangeable.
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_gaudi.calibration.layout import DEFAULT_OBSERVER, OutputLayout
from vllm_gaudi.calibration.presets import ResolvedPreset


@dataclass(frozen=True)
class QuantOptions:
    """Optional QUANTIZE settings that are written only when set.

    Attributes:
        input_backoff: ``scale_params.input_backoff``.
        weight_backoff: ``scale_params.weight_backoff``.
        device_for_scales: ``device_for_scales``, for example ``GAUDI2``.
        measure_exclude: ``measure_exclude``, for example ``OUTPUT``.
        dynamic_quantization: ``dynamic_quantization``.
    """

    input_backoff: float | None = None
    weight_backoff: float | None = None
    device_for_scales: str | None = None
    measure_exclude: str | None = None
    dynamic_quantization: bool | None = None


def _name_list(names: tuple[str, ...]) -> dict[str, list[str]]:
    return {"types": [], "names": list(names)}


def build_measure_config(preset: ResolvedPreset, dump_stats_path: str) -> dict[str, Any]:
    """Returns the MEASURE config dictionary.

    Args:
        preset: Resolved model settings.
        dump_stats_path: Prefix INC uses for the measurement files.
    """
    return {
        "method": "HOOKS",
        "mode": "MEASURE",
        "observer": DEFAULT_OBSERVER,
        "allowlist": _name_list(preset.allowlist_names),
        "blocklist": _name_list(preset.blocklist_names),
        "quantize_weight": False,
        "dump_stats_path": dump_stats_path,
        "calibration_sample_interval": 1,
    }


def build_quant_config(preset: ResolvedPreset,
                       dump_stats_path: str,
                       options: QuantOptions | None = None) -> dict[str, Any]:
    """Returns the QUANTIZE config dictionary.

    Args:
        preset: Resolved model settings.
        dump_stats_path: Prefix of the measurement files written by MEASURE.
        options: Optional settings, written only when set.
    """
    options = options or QuantOptions()
    config: dict[str, Any] = {"mode": "QUANTIZE", "observer": DEFAULT_OBSERVER, "scale_method": preset.scale_method}
    if preset.scale_format is not None:
        config["scale_format"] = preset.scale_format
    config["allowlist"] = _name_list(preset.allowlist_names)
    config["blocklist"] = _name_list(preset.blocklist_names)
    config["dump_stats_path"] = dump_stats_path

    scale_params = {
        key: value
        for key, value in (("input_backoff", options.input_backoff), ("weight_backoff", options.weight_backoff))
        if value is not None
    }
    if scale_params:
        config["scale_params"] = scale_params
    if options.device_for_scales is not None:
        config["device_for_scales"] = options.device_for_scales
    if options.measure_exclude is not None:
        config["measure_exclude"] = options.measure_exclude
    if options.dynamic_quantization is not None:
        config["dynamic_quantization"] = options.dynamic_quantization
    return config


def build_configs(preset: ResolvedPreset,
                  layout: OutputLayout,
                  options: QuantOptions | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns the ``(measure, quant)`` config dictionaries for a layout."""
    dump = layout.dump_stats_path
    return build_measure_config(preset, dump), build_quant_config(preset, dump, options)


def write_json_atomic(path: Path, data: Mapping[str, Any], *, fsync: bool = False) -> None:
    """Writes JSON through a sibling temporary file and an atomic rename.

    Args:
        path: Destination file; parent directories are created.
        data: JSON-serializable mapping.
        fsync: Flush the file to stable storage before the rename, for shared file systems.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")
        if fsync:
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_user_config(path: str | os.PathLike[str], *, base_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Loads a user-supplied INC config and makes its ``dump_stats_path`` absolute.

    Args:
        path: The JSON file given with ``--measure-config`` or ``--quant-config``.
        base_dir: Directory a relative ``dump_stats_path`` is resolved against, normally the invoking cwd.

    Raises:
        ValueError: If the file is not a JSON object or has no ``dump_stats_path``.
    """
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: an INC config must be a JSON object")
    dump = config.get("dump_stats_path")
    if not isinstance(dump, str) or not dump:
        raise ValueError(f"{path}: dump_stats_path is required")
    if not os.path.isabs(dump):
        config["dump_stats_path"] = os.path.normpath(os.path.join(os.fspath(base_dir), dump))
    return config


def resolve_configs(preset: ResolvedPreset,
                    layout: OutputLayout,
                    options: QuantOptions | None = None,
                    *,
                    measure_config: str | None = None,
                    quant_config: str | None = None,
                    base_dir: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns the ``(measure, quant)`` configs of a run, with user configs replacing the generated ones.

    QUANTIZE finds the measurements by a file name built from its own ``dump_stats_path`` and
    ``observer``, so a generated config takes both from a user config given for the other phase.

    Args:
        preset: The resolved model family preset.
        layout: The output layout of the run.
        options: Optional QUANTIZE settings.
        measure_config: Path of the ``--measure-config`` file, if any.
        quant_config: Path of the ``--quant-config`` file, if any.
        base_dir: Directory a relative ``dump_stats_path`` is resolved against, normally the invoking cwd.

    Raises:
        ValueError: If a user config is invalid, or two user configs differ in ``dump_stats_path`` or ``observer``.
    """
    measure, quant = build_configs(preset, layout, options)
    if measure_config:
        measure = load_user_config(measure_config, base_dir=base_dir)
    if quant_config:
        quant = load_user_config(quant_config, base_dir=base_dir)
    # dump_stats_path is required in a user config; INC defaults the observer to maxabs.
    for key, default, normalize in (("dump_stats_path", None, os.path.normpath), ("observer", DEFAULT_OBSERVER, str)):
        measure_value, quant_value = measure.get(key, default), quant.get(key, default)
        if measure_config and not quant_config:
            quant[key] = measure_value
        elif quant_config and not measure_config:
            measure[key] = quant_value
        elif normalize(measure_value) != normalize(quant_value):
            raise ValueError(f"--measure-config and --quant-config have different {key} values "
                             f"({measure_value} and {quant_value}); QUANTIZE would not find the measurements")
    return measure, quant
