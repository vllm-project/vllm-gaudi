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

With ``--tests-out``, the upstream tests listed in ``manifest-tests.txt`` (``<upstream path> <file name>``) are
vendored the same way into one package, with their imports retargeted to the vendored code and to each other, and
``patches-tests/NNNN-Pn-*.patch`` applied relative to that package.

``--check`` rebuilds the tree (and the tests tree with ``--tests-out``) in a temporary directory and fails if it
differs from the committed one, so the committed code is always "upstream at rev + the listed patches". Only git
objects are read: no torch or vllm install is needed.

Usage::

    python3 -P tools/vendor_v1/vendor_v1.py --vllm-repo ../vllm --rev b46907cce0f702eec703bbdac5c5f984cbd2be40 \
        --out vllm_gaudi/v1/_vendored [--tests-out tests/unit_tests/vendored_v1/upstream] [--check]
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
# Upstream V1 tests vendored into one generated package; each line is "<upstream path> <file name>".
TESTS_MANIFEST = HERE / "manifest-tests.txt"
TESTS_PATCH_DIR = HERE / "patches-tests"
TESTS_PKG = "tests.unit_tests.vendored_v1.upstream"

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
REDUCED_UPSTREAM = {"vllm/v1/sample/ops/topk_topp_sampler.py", "tests/v1/sample/utils.py"}

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


def manifest_lines(manifest: Path) -> list[str]:
    """Return the non-comment lines of a manifest file."""
    return [line.strip() for line in manifest.read_text().splitlines() if line.strip() and not line.startswith("#")]


def read_manifest() -> list[str]:
    """Return the upstream paths listed in manifest.txt."""
    paths = manifest_lines(MANIFEST)
    for path in paths:
        if not (path.startswith("vllm/v1/") and path.endswith(".py")):
            raise SystemExit(f"manifest entry outside vllm/v1/: {path}")
    return paths


def read_tests_manifest() -> dict[str, str]:
    """Return {upstream test path: vendored file name} from manifest-tests.txt."""
    tests = {}
    for line in manifest_lines(TESTS_MANIFEST):
        src, name = line.split()
        if not (src.startswith("tests/") and src.endswith(".py") and name.endswith(".py") and "/" not in name):
            raise SystemExit(f"bad tests manifest entry: {line}")
        tests[src] = name
    return tests


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


def rewrite_imports(path: str, src: str, retarget: dict[str, str]) -> str:
    """Retarget the import statements of one vendored file; everything else is left byte-identical.

    Args:
        path: upstream path of the file, to resolve relative imports.
        src: upstream source.
        retarget: upstream module -> vendored module, for every vendored module.
    """
    vendored = set(retarget)
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
                stmts.append(ast.ImportFrom(module=retarget.get(mod, mod), names=keep, level=0))
            if sub:
                stmts.append(ast.ImportFrom(module=dest_module(mod), names=sub, level=0))
            if helpers:
                stmts.append(ast.ImportFrom(module=COMPAT_MOD, names=helpers, level=0))
        elif isinstance(node, ast.Import):
            if not any(alias.name in vendored for alias in node.names):
                continue
            # `import vllm.v1.x` binds `vllm`, which a retargeted import cannot keep, so refuse it.
            for alias in node.names:
                if alias.name in vendored and alias.asname is None:
                    raise SystemExit(f"{path}:{node.lineno}: unaliased `import {alias.name}` of a vendored "
                                     "module cannot be retargeted; extend rewrite_imports to handle it")
            # Keep the bound name the upstream code uses.
            stmts = [
                ast.Import(names=[
                    ast.alias(name=retarget[a.name], asname=a.asname) if a.name in vendored else a for a in node.names
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


def patch_targets(patch_dir: Path = PATCH_DIR) -> dict[str, list[str]]:
    """Map each vendored file (relative to the vendored root) to the patch labels that touch it."""
    targets: dict[str, list[str]] = {}
    for patch in sorted(patch_dir.glob("*.patch")):
        match = PATCH_NAME.match(patch.name)
        if match is None:
            raise SystemExit(f"bad patch file name (want NNNN-Pn-desc.patch): {patch.name}")
        for line in patch.read_text().splitlines():
            if line.startswith("+++ b/"):
                labels = targets.setdefault(line[len("+++ b/"):], [])
                if match.group(1) not in labels:
                    labels.append(match.group(1))
    return targets


def header(short_rev: str, path: str, labels: list[str], patch_dir: Path = PATCH_DIR) -> str:
    """Return the provenance header of a vendored file."""
    what = "reduced" if path in REDUCED_UPSTREAM else "removed"
    where = f"tools/vendor_v1/{patch_dir.name}/"
    changes = "imports retargeted" + (f"; {', '.join(labels)} (see {where})" if labels else "")
    return ("# SPDX-License-Identifier: Apache-2.0\n"
            "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
            f"{VENDORED_FROM}{short_rev}/{path}\n"
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
# Third line of every generated file's provenance header.
VENDORED_FROM = "# Vendored from https://github.com/vllm-project/vllm/blob/"
TESTS_INIT = ("# Upstream vLLM tests vendored by tools/vendor_v1/vendor_v1.py --tests-out; do not edit by hand.\n"
              "# Every local change is a numbered patch in tools/vendor_v1/patches-tests/.\n")


def build(repo: Path, rev: str, out: Path) -> None:
    """Generate the vendored tree into ``out`` (which must not exist)."""
    short_rev = rev[:10]
    paths = read_manifest()
    retarget = {module_of(p): dest_module(module_of(p)) for p in paths}
    targets = patch_targets()
    out.mkdir(parents=True)
    (out / "__init__.py").write_text(ROOT_INIT)  # first, so a failed run still leaves a deletable tree
    for path in paths:
        rel = path[len("vllm/v1/"):]
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        body = strip_spdx(rewrite_imports(path, git_show(repo, rev, path), retarget))
        ast.parse(body, filename=path)  # the rewrite must never emit code that does not parse
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


def _generated_test(path: Path) -> bool:
    """Whether ``path`` looks like a file build_tests() wrote: a .py with the provenance header."""
    if path.suffix != ".py":
        return False
    lines = path.read_text(errors="replace").splitlines()[:3]
    return len(lines) == 3 and lines[0].startswith("# SPDX-License-Identifier") and lines[2].startswith(VENDORED_FROM)


def build_tests(repo: Path, rev: str, out: Path) -> None:
    """Generate the vendored upstream tests into ``out`` (which must not exist)."""
    short_rev = rev[:10]
    tests = read_tests_manifest()
    retarget = {module_of(p): dest_module(module_of(p)) for p in read_manifest()}
    retarget.update({module_of(src): f"{TESTS_PKG}.{name[:-len('.py')]}" for src, name in tests.items()})
    targets = patch_targets(TESTS_PATCH_DIR)
    out.mkdir(parents=True)
    (out / "__init__.py").write_text(TESTS_INIT)  # the marker the delete guard requires
    for src, name in tests.items():
        body = strip_spdx(rewrite_imports(src, git_show(repo, rev, src), retarget))
        ast.parse(body, filename=src)
        (out / name).write_text(header(short_rev, src, targets.get(name, []), TESTS_PATCH_DIR) + body)
    unknown = sorted(set(targets) - set(tests.values()))
    if unknown:
        raise SystemExit(f"test patches target files that are not vendored: {unknown}")
    for patch in sorted(TESTS_PATCH_DIR.glob("*.patch")):
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


def compare(fresh: Path, committed: Path) -> list[str]:
    """Diff a freshly built tree against the committed one; return one line per problem."""
    want, have = tree_files(fresh), tree_files(committed)
    bad = [f"missing: {committed / f}" for f in sorted(want - have)]
    bad += [f"unexpected: {committed / f}" for f in sorted(have - want)]
    for f in sorted(want & have):
        if not filecmp.cmp(fresh / f, committed / f, shallow=False):
            bad.append(f"differs: {committed / f}")
            print(subprocess.run(["git", "diff", "--no-index", "--",
                                  str(fresh / f), str(committed / f)],
                                 capture_output=True,
                                 text=True).stdout,
                  end="")
    return bad


def check(repo: Path, rev: str, out: Path, tests_out: Path | None = None) -> int:
    """Rebuild into a temporary directory and diff against ``out`` (and ``tests_out``). Return an exit code."""
    with tempfile.TemporaryDirectory(prefix=".vendor_v1_check-", dir=out.parent) as tmp:
        build(repo, rev, Path(tmp) / out.name)
        bad = compare(Path(tmp) / out.name, out)
        if tests_out is not None:
            build_tests(repo, rev, Path(tmp) / "tests")
            bad += compare(Path(tmp) / "tests", tests_out)
    if bad:
        print("\n".join(bad))
        print(f"Not upstream {rev[:10]} + tools/vendor_v1/patches*/. Regenerate with vendor_v1.py, and record any "
              "local change as a new numbered patch.")
        return 1
    print(f"OK: {out}{' and ' + str(tests_out) if tests_out else ''} match upstream {rev[:10]} + the patches")
    return 0


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vllm-repo", type=Path, required=True, help="path to a vLLM git checkout")
    parser.add_argument("--rev", required=True, help="vLLM commit to vendor from")
    parser.add_argument("--out", type=Path, required=True, help="vendored tree, e.g. vllm_gaudi/v1/_vendored")
    parser.add_argument("--tests-out",
                        type=Path,
                        help="vendored upstream tests, e.g. tests/unit_tests/vendored_v1/upstream")
    parser.add_argument("--check",
                        action="store_true",
                        help="verify --out (and --tests-out) instead of regenerating them")
    args = parser.parse_args()
    rev = resolve_rev(args.vllm_repo, args.rev)
    out = args.out.resolve()
    tests_out = args.tests_out.resolve() if args.tests_out else None
    if tests_out is not None and (tests_out == out or tests_out in out.parents or out in tests_out.parents):
        # Regenerating one would delete the other.
        raise SystemExit(f"--out {out} and --tests-out {tests_out} must not be the same or nested")
    if args.check:
        return check(args.vllm_repo, rev, out, tests_out)
    # Only ever delete trees this script generated (a wrong or empty path would otherwise remove unrelated files).
    if out.exists():
        init = out / "__init__.py"
        if not (init.is_file() and init.read_text().startswith(ROOT_INIT)):
            raise SystemExit(f"refusing to delete {out}: it is not a tree generated by vendor_v1.py")
    if tests_out is not None and tests_out.exists():
        expected = {"__init__.py", *read_tests_manifest().values()}
        init = tests_out / "__init__.py"
        # A test dropped from the manifest is still recognised by its provenance header, so it can be removed.
        unknown = [
            f for f in tree_files(tests_out)
            if f not in expected and not ("/" not in f and _generated_test(tests_out / f))
        ]
        if not (init.is_file() and init.read_text().startswith(TESTS_INIT)) or unknown:
            raise SystemExit(f"refusing to delete {tests_out}: it is not a tests tree generated by vendor_v1.py")
    if out.exists():
        shutil.rmtree(out)
    build(args.vllm_repo, rev, out)
    if tests_out is not None:
        if tests_out.exists():
            shutil.rmtree(tests_out)
        build_tests(args.vllm_repo, rev, tests_out)
    print(f"regenerated {out}{' and ' + str(tests_out) if tests_out else ''} from upstream {rev[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
