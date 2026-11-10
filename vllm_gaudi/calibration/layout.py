# SPDX-License-Identifier: Apache-2.0
"""On-disk layout of calibration outputs.

The layout is unchanged from the legacy ``calibrate_model.sh`` so that existing
``QUANT_CONFIG`` paths keep working::

    <output_dir>/<model_name>/maxabs_measure_<device>.json
    <output_dir>/<model_name>/maxabs_quant_<device>.json
    <output_dir>/<model_name>/<device>/inc_output_hooks_maxabs_<rank>_<world>.{json,npz}
    <output_dir>/<model_name>/<device>/inc_output_hooks_maxabs_<rank>_<world>_mod_list.json
    <output_dir>/<model_name>/<device>/inc_output_hooks_maxabs_MAXABS_HW_<rank>_<world>.{json,npz}
    <output_dir>/<model_name>/<device>/calibration_manifest.json
"""

import os
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_DEVICES = ("g2", "g3")
DUMP_STATS_BASENAME = "inc_output"
DEFAULT_OBSERVER = "maxabs"
MANIFEST_NAME = "calibration_manifest.json"


def model_dir_name(model: str) -> str:
    """Returns the output directory name for a model path or Hugging Face ID.

    Args:
        model: Local model directory or Hugging Face model ID.

    Returns:
        The lower-cased last path component, for example ``qwen2.5-0.5b-instruct``.
    """
    name = os.path.basename(model.rstrip("/"))
    if not name:
        raise ValueError(f"Cannot derive a model name from {model!r}")
    return name.lower()


@dataclass(frozen=True)
class OutputLayout:
    """Paths of every artifact produced for one model on one device type.

    Attributes:
        output_dir: Absolute root output directory given with ``-o``.
        model_name: Directory name of the model, see :func:`model_dir_name`.
        device: Device type, ``g2`` or ``g3``.
    """

    output_dir: Path
    model_name: str
    device: str

    def __post_init__(self) -> None:
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported device {self.device!r}, expected one of {SUPPORTED_DEVICES}")
        if not self.output_dir.is_absolute():
            raise ValueError(f"output_dir must be absolute, got {self.output_dir}")

    @classmethod
    def create(cls, output_dir: str | os.PathLike[str], model: str, device: str) -> "OutputLayout":
        """Builds a layout from user input, resolving the output directory."""
        return cls(Path(os.path.abspath(os.fspath(output_dir))), model_dir_name(model), device)

    @property
    def model_dir(self) -> Path:
        return self.output_dir / self.model_name

    @property
    def measure_config(self) -> Path:
        return self.model_dir / f"maxabs_measure_{self.device}.json"

    @property
    def quant_config(self) -> Path:
        return self.model_dir / f"maxabs_quant_{self.device}.json"

    @property
    def stats_dir(self) -> Path:
        return self.model_dir / self.device

    @property
    def dump_stats_path(self) -> str:
        """Value of ``dump_stats_path`` in the INC configs; INC appends its own suffixes."""
        return str(self.stats_dir / DUMP_STATS_BASENAME)

    @property
    def manifest(self) -> Path:
        return self.stats_dir / MANIFEST_NAME

    @property
    def logs_dir(self) -> Path:
        return self.stats_dir / "logs"
