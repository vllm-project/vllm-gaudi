"""mkdocs-macros hook – automatically sets {{ VLLM_VERSION }}.

The macros plugin loads this file and calls define_env() at startup.
VLLM_VERSION is derived from the latest git tag (e.g. v0.19.0 → 0.19.0).
Falls back to the branch name if it matches releases/vX.Y.Z.
"""

import re
import subprocess


def _get_vllm_version() -> str | None:
    """Extract version from the most recent vX.Y.Z git tag or branch name."""
    try:
        # Try the nearest release tag first
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        m = re.match(r"v?(\d+\.\d+\.\d+)", tag)
        if m:
            return m.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: parse from branch name (e.g. releases/v0.19.0)
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        m = re.match(r"releases/v?(\d+\.\d+\.\d+)", branch)
        if m:
            return m.group(1)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None


def define_env(env):
    version = _get_vllm_version()
    if version:
        env.variables["VLLM_VERSION"] = version
