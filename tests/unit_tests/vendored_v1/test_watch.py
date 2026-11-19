# SPDX-License-Identifier: Apache-2.0
"""tools/vendor_v1/watch.py against a throwaway vLLM-like git repository."""
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("vendor_v1_watch", REPO / "tools/vendor_v1/watch.py")
assert _spec is not None and _spec.loader is not None
watch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(watch)

# Like vllm/envs.py: the variables are served by a module __getattr__ from the environment_variables keys.
_ENVS = ("def f(): return 1\nenvironment_variables = {{{keys}}}\n"
         "def __getattr__(name):\n    return environment_variables[name]()\n")

_IMPL = "__all__ = ['g']\n__all__ += ['h']\ndef g(): pass\ndef h(): pass\ndef hidden(): pass\n"

_BASE: dict[str, str | None] = {
    "vllm/__init__.py": "",
    "vllm/a.py": "Kept = 1\nRemoved = 2\n",
    "vllm/b.py": "def f(): pass\n",
    "vllm/c.py": "X = 1\n",
    "vllm/envs.py": _ENVS.format(keys='"FLAG": f, "GONE": f'),
    "vllm/pkg/__init__.py": "from .impl import *\n",
    "vllm/pkg/impl.py": _IMPL,
    "vllm/pkg/sub.py": "",
}

_TIP: dict[str, str | None] = {
    # Removed now exists only for type checkers and Annotated has no value; the other names are bound in blocks.
    "vllm/a.py": ("from typing import TYPE_CHECKING\nfrom contextlib import suppress\nKept = 1\nAnnotated: int\n"
                  "if TYPE_CHECKING:\n    Removed = 2\n"
                  "with suppress(ImportError):\n    from os import path as Ctx\n"
                  "if Kept:\n    Else1 = 1\nelse:\n    Else2 = 2\n"
                  "try:\n    import nothing_here as Opt\nexcept ImportError:\n    Opt = None\n"
                  "for Loop in range(1):\n    pass\n"),
    "vllm/b.py":
    None,
    "vllm/c.py":
    "",
    "vllm/envs.py":
    _ENVS.format(keys='"FLAG": f'),
    "vllm/pkg/impl.py":
    _IMPL + "# changed\n",
}

_VENDORED = (
    "import vllm.envs as envs\nimport vllm.c\nimport vllm.b\n"
    "from vllm.a import Annotated, Ctx, Else1, Else2, Kept, Loop, Opt, Removed\n"
    "from vllm.pkg import g, h, hidden, sub\n\n"
    "def late():\n    return envs.FLAG + envs.GONE + vllm.c.X\n\n"
    # A parameter that shadows the alias, and an attribute write, are not reads of the module.
    "def shadow(envs):\n    return envs.anything\n\n"
    "def write():\n    envs.WRITTEN = 1\n")


def _commit(repo: Path, files: dict[str, str | None], message: str) -> None:
    for name, text in files.items():
        path = repo / name
        if text is None:
            path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message], check=True)


def _repos(tmp_path: Path) -> tuple[Path, Path]:
    vllm = tmp_path / "vllm-repo"
    subprocess.run(["git", "init", "-q", str(vllm)], check=True)
    _commit(vllm, _BASE, "base")
    _commit(vllm, _TIP, "tip")
    vendored = tmp_path / "vendored"
    vendored.mkdir()
    (vendored / "m.py").write_text(_VENDORED)
    return vllm, vendored


def test_watch_reports_what_the_tip_removed(tmp_path):
    vllm, vendored = _repos(tmp_path)
    base, tip = watch.resolve(vllm, "HEAD~1"), watch.resolve(vllm, "HEAD")

    result = watch.watch(vllm, base, tip, vendored)

    assert [m["module"] for m in result["missing_modules"]] == ["vllm.b"]
    assert {(m["module"], m["name"])
            for m in result["missing_names"]} == {
                ("vllm.a", "Annotated"),  # a bare annotation binds nothing
                ("vllm.a", "Removed"),  # bound only under TYPE_CHECKING
                ("vllm.c", "X"),  # attribute read through the dotted module name
                ("vllm.envs", "GONE"),  # attribute of an imported module, served by __getattr__ from its key list
                ("vllm.pkg", "hidden"),  # not in the star-imported module's __all__
            }
    # g and h come through the star import, sub is a submodule, the rest are bound in blocks: all resolve.
    assert result["unverified"] == []
    # The star-imported file is watched too.
    assert {c["path"] for c in result["changed"]} == {"vllm/a.py", "vllm/c.py", "vllm/envs.py", "vllm/pkg/impl.py"}
    # Nothing changed between a commit and itself.
    assert watch.watch(vllm, tip, tip, vendored)["changed"] == []


def test_watch_exit_codes(tmp_path, monkeypatch, capsys):
    vllm, _ = _repos(tmp_path)
    monkeypatch.setattr(sys, "argv", ["watch.py", "--vllm-repo", str(vllm), "--base", "HEAD", "--tip", "nope"])
    assert watch.main() == 2
    assert "cannot resolve nope" in capsys.readouterr().err
