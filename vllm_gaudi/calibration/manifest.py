# SPDX-License-Identifier: Apache-2.0
"""The calibration manifest: what was calibrated, how, and with which software.

The manifest is written to the device directory of the output layout and records the directory
of the measurement files, so a set of scales can always be traced back to the model, tasks,
presets and library versions that produced it. The two directories differ only when a custom
config sets another ``dump_stats_path``.
"""

import datetime
import re
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from vllm_gaudi.calibration.layout import MANIFEST_NAME

MANIFEST_VERSION = 1
# Distributions recorded in the manifest; the INC wheel has been published under several names.
TRACKED_DISTRIBUTIONS = ("vllm", "vllm_gaudi", "lm_eval", "neural_compressor_pt", "neural_compressor_3x_pt",
                         "neural_compressor", "torch", "habana_torch_plugin", "transformers")
_SECRET_RE = re.compile(r"TOKEN|SECRET|PASSWORD|KEY|CREDENTIAL", re.IGNORECASE)


def collect_versions(distributions: tuple[str, ...] = TRACKED_DISTRIBUTIONS) -> dict[str, str]:
    """Returns the installed versions of the tracked distributions, without importing them."""
    versions = {}
    for name in distributions:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def redact_env(env: Mapping[str, str]) -> dict[str, str]:
    """Masks values of variables whose names look like credentials."""
    return {key: "<redacted>" if _SECRET_RE.search(key) else value for key, value in env.items()}


def inventory(directory: Path) -> list[str]:
    """Returns the sorted names of the INC output files in a directory."""
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir()
                  if p.is_file() and p.suffix in (".json", ".npz") and p.name != MANIFEST_NAME)


def build_manifest(**fields: Any) -> dict[str, Any]:
    """Returns a manifest with the version header, the creation time and ``fields``."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "versions": collect_versions(),
        **fields,
    }
