#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Regenerate (or check) vllm_gaudi/v1/_vendored/ from an upstream vLLM checkout.

Upstream vLLM removed Model Runner V1 in aac1b2a4f3 (vllm-project/vllm#63373). The HPU model runner still
uses its sampler, spec-decode and input-batch modules, so they are vendored from the last commit that has them.

The vendored tree is a pure function of (vLLM rev, manifest.txt, patches/):

1. ``git show <rev>:<path>`` for every path in ``manifest.txt``.
2. Replace the upstream SPDX lines with a provenance header.
3. Rewrite import statements only: intra-set ``vllm.v1.X`` becomes ``vllm_gaudi.v1._vendored.X``, and the
   helpers removed from still-existing upstream modules come from ``vllm_gaudi.v1._vendored._compat``.
4. Generate ``_compat.py`` and the package ``__init__.py`` files.
5. Apply ``patches/NNNN-Pn-*.patch`` in order with ``git apply`` (paths relative to the vendored root).

``--check`` rebuilds the tree in a temporary directory and fails if it differs from the committed one, so the
committed code is always "upstream at rev + the listed patches". Only git objects are read: no torch or vllm
install is needed.

Usage::

    python3 -P tools/vendor_v1/vendor_v1.py --vllm-repo ../vllm --rev b46907cce0f702eec703bbdac5c5f984cbd2be40 \
        --out vllm_gaudi/v1/_vendored [--check]
"""

from __future__ import annotations

import argparse
import ast
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.txt"
PATCH_DIR = HERE / "patches"

DEST_PKG = "vllm_gaudi.v1._vendored"
COMPAT_MOD = f"{DEST_PKG}._compat"
REMOVAL = "aac1b2a4f3, vllm-project/vllm#63373"

# Helpers removed in the V1 removal from modules that still exist upstream: (module, name, source path).
COMPAT_HELPERS = [
    ("vllm.utils.collection_utils", "swap_dict_values", "vllm/utils/collection_utils.py"),
    ("vllm.v1.utils", "copy_slice", "vllm/v1/utils.py"),
    ("vllm.utils.torch_utils", "make_tensor_with_pad", "vllm/utils/torch_utils.py"),
]
# Files that still exist upstream after the removal, in a reduced form.
REDUCED_UPSTREAM = {"vllm/v1/sample/ops/topk_topp_sampler.py"}

PATCH_NAME = re.compile(r"^\d{4}-(P\d+)-[\w.-]+\.patch$")


def git_show(repo: Path, rev: str, path: str) -> str:
    """Return the content of ``path`` at ``rev`` in ``repo``."""
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=repo, check=True, capture_output=True, text=True).stdout


def resolve_rev(repo: Path, rev: str) -> str:
    """Return the full commit SHA for ``rev``."""
    return subprocess.run(["git", "rev-parse", "--verify", f"{rev}^{{commit}}"],
                          cwd=repo,
                          check=True,
                          capture_output=True,
                          text=True).stdout.strip()


def read_manifest() -> list[str]:
    """Return the upstream paths listed in manifest.txt."""
    paths = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            if not (line.startswith("vllm/v1/") and line.endswith(".py")):
                raise SystemExit(f"manifest entry outside vllm/v1/: {line}")
            paths.append(line)
    return paths


def module_of(path: str) -> str:
    """Return the dotted module name of a repo-relative .py path."""
    mod = path[:-len(".py")].replace("/", ".")
    return mod[:-len(".__init__")] if mod.endswith(".__init__") else mod


def dest_module(mod: str) -> str:
    """Map an upstream ``vllm.v1.X`` module to its vendored name."""
    assert mod.startswith("vllm.v1."), mod
    return f"{DEST_PKG}.{mod[len('vllm.v1.'):]}"


def absolute_module(path: str, node: ast.ImportFrom) -> str:
    """Return the absolute module of an ImportFrom node in the file at ``path``."""
    if not node.level:
        assert node.module is not None
        return node.module
    parts = module_of(path).split(".")
    # A package __init__ is its own package; a plain module drops its last component.
    keep = len(parts) - (node.level - 1 if path.endswith("/__init__.py") else node.level)
    return ".".join(parts[:keep] + ([node.module] if node.module else []))


def rewrite_imports(path: str, src: str, vendored: set[str]) -> str:
    """Retarget the import statements of one vendored file; everything else is left byte-identical."""
    compat = {(mod, name) for mod, name, _ in COMPAT_HELPERS}
    lines = src.split("\n")
    edits = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            mod = absolute_module(path, node)
            keep, sub, helpers = [], [], []
            for alias in node.names:
                if f"{mod}.{alias.name}" in vendored:
                    sub.append(alias)  # `from vllm.v1.sample import sampler`
                elif (mod, alias.name) in compat:
                    helpers.append(alias)
                else:
                    keep.append(alias)
            if not (mod in vendored or sub or helpers or node.level):
                continue
            stmts: list[ast.stmt] = []
            if keep:
                stmts.append(ast.ImportFrom(module=dest_module(mod) if mod in vendored else mod, names=keep, level=0))
            if sub:
                stmts.append(ast.ImportFrom(module=dest_module(mod), names=sub, level=0))
            if helpers:
                stmts.append(ast.ImportFrom(module=COMPAT_MOD, names=helpers, level=0))
        elif isinstance(node, ast.Import):
            if not any(alias.name in vendored for alias in node.names):
                continue
            # Keep the bound name the upstream code uses.
            stmts = [
                ast.Import(names=[
                    ast.alias(name=dest_module(a.name), asname=a.asname or a.name) if a.name in vendored else a
                    for a in node.names
                ])
            ]
        else:
            continue
        indent = lines[node.lineno - 1][:node.col_offset]
        assert node.end_lineno is not None
        edits.append((node.lineno, node.end_lineno, "\n".join(indent + ast.unparse(s) for s in stmts)))
    for first, last, text in sorted(edits, reverse=True):
        lines[first - 1:last] = [text]
    return "\n".join(lines)


def strip_spdx(src: str) -> str:
    """Drop the upstream SPDX lines at the top of a file."""
    lines = src.split("\n")
    head = [line for line in lines[:3] if not line.startswith("# SPDX")]
    return "\n".join(head + lines[3:]).lstrip("\n")


def patch_targets() -> dict[str, list[str]]:
    """Map each vendored file (relative to the vendored root) to the patch labels that touch it."""
    targets: dict[str, list[str]] = {}
    for patch in sorted(PATCH_DIR.glob("*.patch")):
        match = PATCH_NAME.match(patch.name)
        if match is None:
            raise SystemExit(f"bad patch file name (want NNNN-Pn-desc.patch): {patch.name}")
        for line in patch.read_text().splitlines():
            if line.startswith("+++ b/"):
                labels = targets.setdefault(line[len("+++ b/"):], [])
                if match.group(1) not in labels:
                    labels.append(match.group(1))
    return targets


def header(short_rev: str, path: str, labels: list[str]) -> str:
    """Return the provenance header of a vendored file."""
    what = "reduced" if path in REDUCED_UPSTREAM else "removed"
    changes = "imports retargeted" + (f"; {', '.join(labels)} (see tools/vendor_v1/patches/)" if labels else "")
    return ("# SPDX-License-Identifier: Apache-2.0\n"
            "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
            f"# Vendored from https://github.com/vllm-project/vllm/blob/{short_rev}/{path}\n"
            f"# ({what} upstream in {REMOVAL}). Local changes: {changes}.\n")


def compat_source(repo: Path, rev: str, short_rev: str, labels: list[str]) -> str:
    """Build _compat.py from the upstream definitions of the removed helpers, verbatim."""
    parts = []
    for _, name, path in COMPAT_HELPERS:
        src = git_show(repo, rev, path)
        defs = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(defs) != 1:
            raise SystemExit(f"{name} not found once in {path}@{short_rev}")
        segment = ast.get_source_segment(src, defs[0])
        assert segment is not None
        parts.append(f"# Vendored from https://github.com/vllm-project/vllm/blob/{short_rev}/{path}\n{segment}")
    changes = "; ".join(["copied verbatim"] +
                        ([", ".join(labels) + " (see tools/vendor_v1/patches/)"] if labels else []))
    return ("# SPDX-License-Identifier: Apache-2.0\n"
            "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
            f"# Helpers removed from still-existing vLLM modules in {REMOVAL}.\n"
            f"# Local changes: {changes}.\n"
            '"""V1 helpers that upstream vLLM removed together with Model Runner V1."""\n\n'
            "from typing import TypeVar\n\n"
            "import torch\n\n"
            "from vllm.utils.torch_utils import TORCH_DTYPE_TO_NUMPY_DTYPE, make_ndarray_with_pad\n\n"
            '_K = TypeVar("_K")\n_V = TypeVar("_V")\nT = TypeVar("T")\n\n\n' + "\n\n\n".join(parts) + "\n")


ROOT_INIT = ('"""Model Runner V1 modules vendored from upstream vLLM.\n\n'
             "Generated by tools/vendor_v1/vendor_v1.py; do not edit by hand. Every local change is a numbered patch\n"
             'in tools/vendor_v1/patches/.\n"""\n')


def build(repo: Path, rev: str, out: Path) -> None:
    """Generate the vendored tree into ``out`` (which must not exist)."""
    short_rev = rev[:10]
    paths = read_manifest()
    vendored = {module_of(p) for p in paths}
    targets = patch_targets()
    out.mkdir(parents=True)
    for path in paths:
        rel = path[len("vllm/v1/"):]
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        body = strip_spdx(rewrite_imports(path, git_show(repo, rev, path), vendored))
        dst.write_text(header(short_rev, path, targets.get(rel, [])) + body)
    (out / "_compat.py").write_text(compat_source(repo, rev, short_rev, targets.get("_compat.py", [])))
    for pkg in sorted({p.parent for p in out.rglob("*.py")}):
        init = pkg / "__init__.py"
        if not init.exists():
            init.write_text(ROOT_INIT if pkg == out else "")
    unknown = sorted(set(targets) - {str(p.relative_to(out)) for p in out.rglob("*.py")})
    if unknown:
        raise SystemExit(f"patches target files that are not vendored: {unknown}")
    for patch in sorted(PATCH_DIR.glob("*.patch")):
        apply_patch(patch, out)


def apply_patch(patch: Path, root: Path) -> None:
    """Apply one patch to the tree at ``root`` and fail unless it changed every file it targets."""
    files = [line[len("+++ b/"):] for line in patch.read_text().splitlines() if line.startswith("+++ b/")]
    before = {f: (root / f).read_bytes() for f in files}
    # Inside a git work tree, git apply silently skips paths it considers outside the index; stop the repository
    # discovery at root's parent so the patch always applies to the plain files under root.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(root.parent))
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    res = subprocess.run(["git", "apply", "--verbose", "--whitespace=nowarn",
                          str(patch)],
                         cwd=root,
                         env=env,
                         capture_output=True,
                         text=True)
    if res.returncode or "Skipped patch" in res.stderr:
        raise SystemExit(f"{patch.name} does not apply:\n{res.stderr}")
    unchanged = [f for f in files if (root / f).read_bytes() == before[f]]
    if unchanged:
        raise SystemExit(f"{patch.name} left {unchanged} unchanged:\n{res.stderr}")


def tree_files(root: Path) -> set[str]:
    """Return the files under ``root``, relative, ignoring bytecode caches."""
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def check(repo: Path, rev: str, out: Path) -> int:
    """Rebuild into a temporary directory and diff against ``out``. Return a process exit code."""
    with tempfile.TemporaryDirectory(prefix=".vendor_v1_check-", dir=out.parent) as tmp:
        fresh = Path(tmp) / out.name
        build(repo, rev, fresh)
        want, have = tree_files(fresh), tree_files(out)
        bad = [f"missing: {f}" for f in sorted(want - have)] + [f"unexpected: {f}" for f in sorted(have - want)]
        bad += [f"differs: {f}" for f in sorted(want & have) if not filecmp.cmp(fresh / f, out / f, shallow=False)]
        for f in [b.split(": ", 1)[1] for b in bad if b.startswith("differs: ")]:
            diff = subprocess.run(["git", "diff", "--no-index", "--",
                                   str(fresh / f), str(out / f)],
                                  capture_output=True,
                                  text=True).stdout
            print(diff, end="")
    if bad:
        print("\n".join(bad))
        print(f"{out} is not upstream {rev[:10]} + tools/vendor_v1/patches/. Regenerate it with vendor_v1.py, and "
              "record any local change as a new numbered patch.")
        return 1
    print(f"{out}: OK ({len(want)} files match upstream {rev[:10]} + {len(list(PATCH_DIR.glob('*.patch')))} patches)")
    return 0


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vllm-repo", type=Path, required=True, help="path to a vLLM git checkout")
    parser.add_argument("--rev", required=True, help="vLLM commit to vendor from")
    parser.add_argument("--out", type=Path, required=True, help="vendored tree, e.g. vllm_gaudi/v1/_vendored")
    parser.add_argument("--check", action="store_true", help="verify --out instead of regenerating it")
    args = parser.parse_args()
    rev = resolve_rev(args.vllm_repo, args.rev)
    out = args.out.resolve()
    if args.check:
        return check(args.vllm_repo, rev, out)
    if out.exists():
        shutil.rmtree(out)
    build(args.vllm_repo, rev, out)
    print(f"{out}: regenerated from upstream {rev[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
