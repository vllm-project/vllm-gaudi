#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Watch the upstream vLLM modules the vendored Model Runner V1 code still imports.

The vendored tree (vllm_gaudi/v1/_vendored/, see vendor_v1.py) is frozen at the vLLM commit it was copied from, but
it imports modules that upstream keeps changing. Between a base vLLM commit (normally VLLM_STABLE_COMMIT) and a
newer one (normally the vLLM main tip) this reports, from git objects only (no torch or vllm install):

* ``missing_modules``: imported ``vllm.*`` modules that no longer exist at the tip;
* ``missing_names``: names imported with ``from vllm.x import name`` that the tip module no longer defines;
* ``unverified``: names not found in tip modules whose names cannot all be known statically (a module-level
  ``__getattr__`` or an absolute star import);
* ``changed``: imported module files that differ between base and tip (``git diff --numstat``), to read first when
  the tip breaks something;
* ``vendor_check``: the result of ``vendor_v1.py --check`` (the committed tree is upstream + the listed patches).

Attributes read on imported vLLM modules (``envs.X``) are checked like imported names. Names bound only under
``if TYPE_CHECKING:`` do not count. Exit status 1 if anything is missing or the vendor check fails, 2 if a commit or
the repository cannot be resolved, else 0. The JSON summary goes to stdout.

Usage::

    python3 -P tools/vendor_v1/watch.py --vllm-repo ../vllm --base <VLLM_STABLE_COMMIT> --tip origin/main
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENDORED = REPO / "vllm_gaudi" / "v1" / "_vendored"
# The commit the vendored tree was copied from (vendor_v1.py --rev).
VENDOR_REV = "b46907cce0f702eec703bbdac5c5f984cbd2be40"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def _is_vllm(module: str | None) -> bool:
    return bool(module) and (module == "vllm" or module.startswith("vllm."))


def _bound_locally(func: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    """Names a function binds in its own scope (parameters, assignments, imports, loop and with targets)."""
    args = func.args
    names = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    if isinstance(func, ast.Lambda):
        return names
    for node in ast.walk(ast.Module(body=func.body, type_ignores=[])):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def _dotted(node: ast.expr) -> str | None:
    """``a.b.c`` for a chain of attribute reads rooted at a name, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _scan_file(tree: ast.Module, rel: str, found: dict[str, dict[str, set[str]]],
               attrs: dict[str, dict[str, set[str]]]) -> None:
    """Add one file's ``vllm.*`` imports to ``found`` and its attribute reads on vLLM modules to ``attrs``."""
    aliases: dict[str, str] = {}  # local name -> module it may be (``import vllm.x as y``, ``from vllm import x``)
    plain: set[str] = set()  # modules imported without ``as`` (``import vllm.x.y``), read as ``vllm.x.y.attr``
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_vllm(alias.name):
                    found.setdefault(alias.name, {}).setdefault("", set()).add(rel)
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                    else:
                        plain.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level and _is_vllm(node.module):
            for alias in node.names:
                found.setdefault(node.module, {}).setdefault(alias.name, set()).add(rel)
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def record(module: str, attr: str) -> None:
        attrs.setdefault(module, {}).setdefault(attr, set()).add(rel)

    def visit(node: ast.AST, shadowed: frozenset[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            shadowed = shadowed | _bound_locally(node)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            chain = _dotted(node)
            if chain is not None:
                head, _, rest = chain.partition(".")
                if head in aliases and head not in shadowed:
                    record(aliases[head], rest.split(".")[0])
                elif head == "vllm" and head not in shadowed:
                    # Longest imported module prefix of ``vllm.x.y.attr``; the next part is the attribute.
                    parts = chain.split(".")
                    for i in range(len(parts) - 1, 0, -1):
                        if ".".join(parts[:i]) in plain:
                            record(".".join(parts[:i]), parts[i])
                            break
                return  # the chain is handled as a whole
        for child in ast.iter_child_nodes(node):
            visit(child, shadowed)

    visit(tree, frozenset())


def vllm_imports(root: Path) -> tuple[dict[str, dict[str, set[str]]], dict[str, dict[str, set[str]]]]:
    """Every ``vllm.*`` import in the tree, {module: {imported name or "": {importing files}}}, and every attribute
    read on a name that may be a vLLM module, {candidate module: {attribute: {importing files}}}."""
    found: dict[str, dict[str, set[str]]] = {}
    attrs: dict[str, dict[str, set[str]]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        _scan_file(ast.parse(path.read_text(), filename=rel), rel, found, attrs)
    return found, attrs


def module_path(repo: Path, rev: str, module: str) -> str | None:
    """The file that defines ``module`` at ``rev``, or None if it does not exist."""
    base = module.replace(".", "/")
    for candidate in (f"{base}.py", f"{base}/__init__.py"):
        if _git(repo, "cat-file", "-e", f"{rev}:{candidate}", check=False).returncode == 0:
            return candidate
    return None


_TRY_NODES = (ast.Try, ast.TryStar) if hasattr(ast, "TryStar") else (ast.Try, )  # TryStar: Python 3.11+


def _target_names(target: ast.expr) -> set[str]:
    """Names an assignment target binds (not names used inside subscripts or attributes)."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(e) for e in target.elts))
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (isinstance(test, ast.Attribute)
                                                                           and test.attr == "TYPE_CHECKING")


def _literal_all(tree: ast.Module) -> set[str] | None:
    """A module's literal ``__all__``, with top-level ``+=`` and ``.extend(...)`` applied, or None."""
    names: set[str] | None = None
    try:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                names = set(ast.literal_eval(node.value))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
                names = (names or set()) | set(ast.literal_eval(node.value))
            elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and _dotted(node.value.func) == "__all__.extend" and node.value.args):
                names = (names or set()) | set(ast.literal_eval(node.value.args[0]))
    except (ValueError, TypeError):
        return None
    return names


def _env_var_names(tree: ast.Module) -> set[str]:
    """String keys of a module-level ``environment_variables`` dict (vllm/envs.py serves them via __getattr__)."""
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(target, ast.Name) and target.id == "environment_variables" and isinstance(value, ast.Dict):
            return {k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def defined_names(repo: Path, rev: str, path: str,
                  seen: frozenset[str] = frozenset()) -> tuple[set[str], bool, set[str], set[str] | None]:
    """For the module file at ``rev:path``: the module-level names it binds at runtime (following relative
    ``from .x import *``), whether some names cannot be known statically (a module-level ``__getattr__`` or an
    absolute star import), the files the names came from, and its literal ``__all__`` (or None)."""
    names: set[str] = set()
    files = {path}
    dynamic = False
    package = path.rsplit("/", 1)[0]  # the package a relative import resolves against
    tree = ast.parse(_git(repo, "show", f"{rev}:{path}").stdout)

    def visit(body: list[ast.stmt]) -> None:
        nonlocal dynamic
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
                dynamic |= node.name == "__getattr__"
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    names.update(_target_names(target))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                # A bare annotation (``x: int``) binds nothing.
                if not isinstance(node, ast.AnnAssign) or node.value is not None:
                    names.update(_target_names(node.target))
            elif isinstance(node, ast.ImportFrom) and node.names[0].name == "*":
                target = None
                if node.level == 1 and node.module:
                    target = module_path(repo, rev, f"{package.replace('/', '.')}.{node.module}")
                if target is None or target in seen:
                    dynamic |= target is None
                    continue
                star, star_dynamic, star_files, star_all = defined_names(repo, rev, target, seen | {path})
                names.update(star_all if star_all is not None else (n for n in star if not n.startswith("_")))
                dynamic |= star_dynamic
                files.update(star_files)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.If):
                # Any branch may bind (version guards); names bound only for type checkers do not exist at runtime.
                if not _is_type_checking(node.test):
                    visit(node.body)
                visit(node.orelse)
            elif isinstance(node, _TRY_NODES):
                visit(node.body)
                visit(node.orelse)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.finalbody)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        names.update(_target_names(item.optional_vars))
                visit(node.body)
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                if not isinstance(node, ast.While):
                    names.update(_target_names(node.target))
                visit(node.body)
                visit(node.orelse)
            elif isinstance(node, ast.Match):
                for case in node.cases:
                    visit(case.body)

    visit(tree.body)
    env_vars = _env_var_names(tree)
    if dynamic and env_vars:
        # vllm/envs.py: __getattr__ serves exactly these keys, so the module's names are known after all.
        names |= env_vars
        dynamic = False
    return names, dynamic, files, _literal_all(tree)


def watch(repo: Path, base: str, tip: str, root: Path = VENDORED) -> dict:
    """``base`` and ``tip`` should be full commit SHAs (see ``resolve``)."""
    imports, attrs = vllm_imports(root)
    missing_modules: list[dict] = []
    missing_names: list[dict] = []
    unverified: list[dict] = []
    paths: list[str] = []
    cache: dict[str, tuple[set[str], bool, set[str], set[str] | None]] = {}

    def check(module: str, path: str, wanted: dict[str, set[str]]) -> None:
        if path not in cache:
            cache[path] = defined_names(repo, tip, path)
        defined, dynamic, files, _ = cache[path]
        paths.extend(files)
        for name in sorted(wanted):
            # ``from vllm.x import y`` may name the submodule vllm/x/y.
            if name in defined or module_path(repo, tip, f"{module}.{name}") is not None:
                continue
            entry = {"module": module, "name": name, "imported_by": sorted(wanted[name])}
            (unverified if dynamic else missing_names).append(entry)

    for module, names in sorted(imports.items()):
        path = module_path(repo, tip, module)
        if path is None:
            missing_modules.append({"module": module, "imported_by": sorted(set().union(*names.values()))})
            continue
        paths.append(path)
        check(module, path, {n: f for n, f in names.items() if n})
    # Attributes read on imported modules (``envs.X``); a candidate that is not a module is a plain name, skip it.
    for module, used in sorted(attrs.items()):
        path = module_path(repo, tip, module)
        if path is not None:
            check(module, path, used)
    changed = []
    if paths:
        out = _git(repo, "diff", "--numstat", base, tip, "--", *sorted(set(paths))).stdout
        for line in out.splitlines():
            added, removed, path = line.split("\t", 2)
            changed.append({"path": path, "added": added, "removed": removed})
    return {
        "base": base,
        "tip": tip,
        "imported_modules": len(imports),
        "missing_modules": missing_modules,
        "missing_names": missing_names,
        "unverified": unverified,
        "changed": changed,
    }


def resolve(repo: Path, rev: str) -> str | None:
    out = _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def vendor_check(repo: Path) -> dict:
    cmd = [
        sys.executable, "-P",
        str(Path(__file__).with_name("vendor_v1.py")), "--vllm-repo",
        str(repo), "--rev", VENDOR_REV, "--out",
        str(VENDORED), "--tests-out",
        str(REPO / "tests/unit_tests/vendored_v1/upstream"), "--check"
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode == 0:
        return {"result": "ok"}
    return {"result": "failed", "output": (out.stdout + out.stderr).strip().splitlines()[-10:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vllm-repo", type=Path, required=True, help="a full vLLM clone")
    parser.add_argument("--base", required=True, help="vLLM commit the plugin is pinned to (VLLM_STABLE_COMMIT)")
    parser.add_argument("--tip", default="origin/main", help="newer vLLM commit to check (default: origin/main)")
    parser.add_argument("--skip-vendor-check", action="store_true", help="do not run vendor_v1.py --check")
    args = parser.parse_args()
    base, tip = resolve(args.vllm_repo, args.base), resolve(args.vllm_repo, args.tip)
    if base is None or tip is None:
        print(f"cannot resolve {args.base if base is None else args.tip} in {args.vllm_repo}", file=sys.stderr)
        return 2
    result = watch(args.vllm_repo, base, tip)
    result["vendor_check"] = {"result": "skipped"} if args.skip_vendor_check else vendor_check(args.vllm_repo)
    json.dump(result, sys.stdout, indent=2)
    print()
    broken = result["missing_modules"] or result["missing_names"] or result["vendor_check"]["result"] == "failed"
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
