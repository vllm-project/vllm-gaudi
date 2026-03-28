#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Set up and validate two vllm-gaudi code-base checkpoints for bucketing benchmarks.

Checkpoint 1 (baseline): latest vllm-gaudi main — supports exponential and
    linear bucketing strategies.
Checkpoint 2 (linear-with-limits): baseline + PR #762 cherry-picked on top —
    adds the linear-with-limits bucketing strategy.

The script:
  1. Resolves the exact commit SHAs for both checkpoints.
  2. Creates isolated virtualenvs (or verifies existing ones) under the
     specified ``--env-dir``.
  3. Installs vllm-gaudi from each checkout into its virtualenv in
     editable (``-e``) mode.
  4. Runs a quick import smoke-test to confirm each install is functional.
  5. Captures full environment details (driver version, Python version,
     PyTorch version, Habana software stack version, etc.) and writes
     them to ``benchmark_env.json`` for reproducibility.

Usage:
    python tools/benchmark/setup_benchmark_env.py \\
        --repo-dir /path/to/vllm-gaudi \\
        --env-dir  /tmp/bench_envs \\
        --baseline-ref main \\
        --pr762-ref pr762

Alternatively, when running on a machine that already has a single
virtualenv, pass ``--in-place`` to skip virtualenv creation and only
generate the environment manifest.
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

BASELINE_REF_DEFAULT = "main"
PR762_REF_DEFAULT = "pr762"

# The PR #762 branch name patterns that might exist in the remote.
PR762_CANDIDATE_REFS = [
    "pr762",
    "origin/pr762",
    "refs/pull/762/head",
]


def _git(args: list[str], cwd: str | Path | None = None) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def resolve_sha(ref: str, repo_dir: str | Path) -> str:
    """Resolve *ref* (branch, tag, SHA prefix, …) to a full SHA."""
    return _git(["rev-parse", ref], cwd=repo_dir)


def ref_exists(ref: str, repo_dir: str | Path) -> bool:
    """Return True if *ref* can be resolved in *repo_dir*."""
    try:
        _git(["rev-parse", "--verify", ref], cwd=repo_dir)
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_worktree(repo_dir: str | Path, dest: str | Path, ref: str) -> str:
    """Create (or reuse) a git work-tree at *dest* checked-out at *ref*.

    Returns the full commit SHA of the checked-out revision.
    """
    dest = Path(dest)
    if dest.exists():
        # Already exists — just make sure it's on the right SHA.
        current = resolve_sha("HEAD", dest)
        desired = resolve_sha(ref, repo_dir)
        if current != desired:
            _git(["checkout", ref], cwd=dest)
        return resolve_sha("HEAD", dest)

    _git(["worktree", "add", str(dest), ref], cwd=repo_dir)
    return resolve_sha("HEAD", dest)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def _run_quiet(cmd: list[str]) -> str:
    """Run *cmd* and return stdout, or ``"N/A"`` on failure."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"


def detect_python_version() -> str:
    return platform.python_version()


def detect_pytorch_version() -> str:
    try:
        import torch  # noqa: F401
        return torch.__version__
    except ImportError:
        return "N/A"


def detect_habana_sw_version() -> str:
    """Return the Habana software stack version from hl-smi or the runtime."""
    ver = _run_quiet(["hl-smi", "-v"])
    if ver and ver != "N/A":
        return ver
    # Fallback: try the Python runtime
    try:
        import habana_frameworks  # type: ignore[import-untyped]
        return getattr(habana_frameworks, "__version__", "N/A")
    except ImportError:
        return "N/A"


def detect_driver_version() -> str:
    """Return the Habana kernel driver version."""
    hl_smi_out = _run_quiet(["hl-smi"])
    if hl_smi_out and hl_smi_out != "N/A":
        for line in hl_smi_out.splitlines():
            if "Driver Version" in line:
                parts = line.split("Driver Version")
                if len(parts) > 1:
                    return parts[1].strip().strip(":").strip()
    # Fallback: kernel module
    modinfo = _run_quiet(["modinfo", "-F", "version", "habanalabs"])
    return modinfo if modinfo else "N/A"


def detect_gaudi_device() -> str:
    """Return the Gaudi device type (e.g. 'Gaudi2', 'Gaudi3')."""
    hl_smi_out = _run_quiet(["hl-smi"])
    if hl_smi_out and hl_smi_out != "N/A":
        for line in hl_smi_out.splitlines():
            for candidate in ("Gaudi3", "Gaudi2", "Gaudi"):
                if candidate in line:
                    return candidate
    return "N/A"


def detect_os_info() -> str:
    return f"{platform.system()} {platform.release()} ({platform.version()})"


def collect_environment(repo_dir: str | Path, baseline_sha: str, pr762_sha: str) -> dict:
    """Build a JSON-serialisable environment manifest."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {
            "baseline": {
                "description": "Latest vllm-gaudi main (exponential + linear bucketing)",
                "commit_sha": baseline_sha,
            },
            "linear_with_limits": {
                "description": "Baseline + PR #762 (linear-with-limits bucketing)",
                "commit_sha": pr762_sha,
            },
        },
        "hardware": {
            "device": detect_gaudi_device(),
            "driver_version": detect_driver_version(),
        },
        "software": {
            "os": detect_os_info(),
            "python_version": detect_python_version(),
            "pytorch_version": detect_pytorch_version(),
            "habana_sw_version": detect_habana_sw_version(),
        },
        "repo": {
            "remote_url": _git(["remote", "get-url", "origin"], cwd=repo_dir),
            "main_sha": resolve_sha("main", repo_dir),
        },
    }


# ---------------------------------------------------------------------------
# Virtualenv helpers
# ---------------------------------------------------------------------------


def _create_venv(venv_path: Path, python: str = sys.executable) -> None:
    """Create a virtualenv at *venv_path* if it does not already exist."""
    if venv_path.exists():
        logger.info("Virtualenv already exists: %s", venv_path)
        return
    logger.info("Creating virtualenv at %s", venv_path)
    subprocess.run([python, "-m", "venv", str(venv_path)], check=True)


def _pip_install_editable(venv_path: Path, project_dir: Path) -> None:
    """Run ``pip install -e .`` inside *venv_path* for *project_dir*."""
    pip = venv_path / "bin" / "pip"
    logger.info("Installing vllm-gaudi (editable) from %s", project_dir)
    subprocess.run(
        [str(pip), "install", "-e", str(project_dir)],
        check=True,
    )


def _smoke_test(venv_path: Path) -> bool:
    """Return True if ``import vllm_gaudi`` succeeds in the virtualenv."""
    python = venv_path / "bin" / "python"
    result = subprocess.run(
        [str(python), "-c", "import vllm_gaudi; print(vllm_gaudi.__name__)"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Smoke test passed: %s", result.stdout.strip())
        return True
    logger.error("Smoke test failed:\n%s", result.stderr)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare two vllm-gaudi checkpoints for bucketing benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples
            --------
            # Full setup with isolated virtualenvs
            python tools/benchmark/setup_benchmark_env.py \\
                --repo-dir . --env-dir /tmp/bench_envs

            # In-place mode (no virtualenvs, just generate manifest)
            python tools/benchmark/setup_benchmark_env.py --in-place
        """),
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default=".",
        help="Path to the vllm-gaudi git repository (default: current directory).",
    )
    parser.add_argument(
        "--env-dir",
        type=str,
        default=None,
        help="Directory for virtualenvs and worktrees. Required unless --in-place is set.",
    )
    parser.add_argument(
        "--baseline-ref",
        type=str,
        default=BASELINE_REF_DEFAULT,
        help=f"Git ref for the baseline checkpoint (default: {BASELINE_REF_DEFAULT}).",
    )
    parser.add_argument(
        "--pr762-ref",
        type=str,
        default=PR762_REF_DEFAULT,
        help=f"Git ref for the PR #762 checkpoint (default: {PR762_REF_DEFAULT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tools/benchmark/benchmark_env.json",
        help="Path for the environment manifest JSON (default: tools/benchmark/benchmark_env.json).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Skip virtualenv/worktree creation; only generate the environment manifest.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def resolve_pr762_ref(repo_dir: str | Path, user_ref: str) -> str:
    """Resolve the PR #762 git ref, trying several fallback names.

    The user may specify an explicit ref via ``--pr762-ref``.  If that ref
    does not exist, we try a list of common candidate ref names.  Returns
    the first ref that resolves, or raises ``SystemExit`` if none do.
    """
    if ref_exists(user_ref, repo_dir):
        return user_ref
    for candidate in PR762_CANDIDATE_REFS:
        if ref_exists(candidate, repo_dir):
            logger.info("Resolved PR #762 ref via fallback: %s", candidate)
            return candidate
    logger.warning(
        "PR #762 ref '%s' not found locally.  The manifest will record the "
        "baseline SHA for both checkpoints.  To set up the linear-with-limits "
        "checkpoint, fetch PR #762 and re-run:\n"
        "  git fetch origin pull/762/head:pr762\n"
        "  python tools/benchmark/setup_benchmark_env.py --pr762-ref pr762",
        user_ref,
    )
    return ""


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    repo_dir = Path(args.repo_dir).resolve()
    if not (repo_dir / ".git").exists():
        logger.error("Not a git repository: %s", repo_dir)
        sys.exit(1)

    # Resolve baseline SHA
    baseline_sha = resolve_sha(args.baseline_ref, repo_dir)
    logger.info("Baseline ref '%s' -> %s", args.baseline_ref, baseline_sha)

    # Resolve PR #762 SHA
    pr762_ref = resolve_pr762_ref(repo_dir, args.pr762_ref)
    if pr762_ref:
        pr762_sha = resolve_sha(pr762_ref, repo_dir)
        logger.info("PR #762 ref '%s' -> %s", pr762_ref, pr762_sha)
    else:
        pr762_sha = baseline_sha
        logger.info("PR #762 ref unavailable; using baseline SHA as placeholder: %s", pr762_sha)

    if not args.in_place:
        if args.env_dir is None:
            logger.error("--env-dir is required unless --in-place is set.")
            sys.exit(1)

        env_dir = Path(args.env_dir).resolve()
        env_dir.mkdir(parents=True, exist_ok=True)

        # --- Checkpoint 1: baseline worktree + venv ---
        baseline_wt = env_dir / "worktree_baseline"
        baseline_venv = env_dir / "venv_baseline"
        logger.info("Setting up baseline worktree at %s", baseline_wt)
        actual_baseline_sha = ensure_worktree(repo_dir, baseline_wt, args.baseline_ref)
        assert actual_baseline_sha == baseline_sha

        _create_venv(baseline_venv)
        _pip_install_editable(baseline_venv, baseline_wt)
        if not _smoke_test(baseline_venv):
            logger.error("Baseline smoke test failed. Check installation logs.")
            sys.exit(1)

        # --- Checkpoint 2: PR #762 worktree + venv ---
        if pr762_ref:
            pr762_wt = env_dir / "worktree_linear_with_limits"
            pr762_venv = env_dir / "venv_linear_with_limits"
            logger.info("Setting up PR #762 worktree at %s", pr762_wt)
            actual_pr762_sha = ensure_worktree(repo_dir, pr762_wt, pr762_ref)
            assert actual_pr762_sha == pr762_sha

            _create_venv(pr762_venv)
            _pip_install_editable(pr762_venv, pr762_wt)
            if not _smoke_test(pr762_venv):
                logger.error("PR #762 smoke test failed. Check installation logs.")
                sys.exit(1)
        else:
            logger.warning(
                "Skipping linear-with-limits virtualenv (PR #762 ref not available)."
            )

    # --- Write environment manifest ---
    manifest = collect_environment(repo_dir, baseline_sha, pr762_sha)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Environment manifest written to %s", output_path)

    # Pretty-print summary
    print("\n" + "=" * 72)
    print("Benchmark Environment Summary")
    print("=" * 72)
    print(f"  Baseline commit (exponential + linear) : {baseline_sha}")
    print(f"  PR #762 commit  (linear-with-limits)   : {pr762_sha}")
    print(f"  Device          : {manifest['hardware']['device']}")
    print(f"  Driver version  : {manifest['hardware']['driver_version']}")
    print(f"  Python version  : {manifest['software']['python_version']}")
    print(f"  PyTorch version : {manifest['software']['pytorch_version']}")
    print(f"  Habana SW stack : {manifest['software']['habana_sw_version']}")
    print(f"  OS              : {manifest['software']['os']}")
    print(f"  Manifest        : {output_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
