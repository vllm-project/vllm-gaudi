# SPDX-License-Identifier: Apache-2.0
"""mark_step policy of the Kimi-K3 hooks (extension/graph_state.py, pure python loaded by path)."""

import importlib.util
import pathlib
import sys
import types

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "extension" / "graph_state.py"
_spec = importlib.util.spec_from_file_location("graph_state_under_test", _PATH)
gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gs)


@pytest.fixture
def fake_habana(monkeypatch):
    """habana_frameworks.torch stub: lazy mode on, mark_step recorded."""
    calls = []
    htorch = types.ModuleType("habana_frameworks.torch")
    htorch.core = types.SimpleNamespace(mark_step=lambda: calls.append(1))
    htorch.utils = types.SimpleNamespace(internal=types.SimpleNamespace(is_lazy=lambda: True))
    pkg = types.ModuleType("habana_frameworks")
    pkg.torch = htorch
    monkeypatch.setitem(sys.modules, "habana_frameworks", pkg)
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", htorch)
    return calls


def test_unknown_mode_never_marks_steps(fake_habana):
    assert gs.lazy_step_hook() is None


def test_forward_without_graphs_gets_mark_step_and_graph_forward_does_not(fake_habana):
    with gs.forward_graph_mode(use_graphs=False):
        hook = gs.lazy_step_hook()
        assert hook is not None
        hook()
        assert fake_habana == [1]
    with gs.forward_graph_mode(use_graphs=True):
        assert gs.lazy_step_hook() is None
    assert gs.lazy_step_hook() is None  # state restored after the forward


def test_nested_modes_restore_the_previous_one(fake_habana):
    with gs.forward_graph_mode(use_graphs=False):
        with gs.forward_graph_mode(use_graphs=True):
            assert not gs.forward_without_graphs()
        assert gs.forward_without_graphs()


def test_state_is_restored_when_the_forward_raises(fake_habana):
    with pytest.raises(RuntimeError):
        with gs.forward_graph_mode(use_graphs=False):
            raise RuntimeError
    assert not gs.forward_without_graphs()


def test_not_lazy_or_without_habana_gives_no_hook(monkeypatch, fake_habana):
    import habana_frameworks.torch as htorch
    monkeypatch.setattr(htorch.utils.internal, "is_lazy", lambda: False)
    with gs.forward_graph_mode(use_graphs=False):
        assert gs.lazy_step_hook() is None
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", None)  # import raises ImportError
    with gs.forward_graph_mode(use_graphs=False):
        assert gs.lazy_step_hook() is None


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_env_switch_disables_the_cuts(monkeypatch, fake_habana, value):
    monkeypatch.setenv("HPU_KIMI_SITU_MARK_STEP", value)
    with gs.forward_graph_mode(use_graphs=False):
        assert gs.lazy_step_hook() is None


def test_runner_and_both_hooks_use_the_shared_policy():
    root = _PATH.parents[1]
    assert "forward_graph_mode(use_graphs)" in (root / "v1/worker/hpu_model_runner.py").read_text()
    assert "lazy_step_hook()" in (root / "models/kimi_k3_kda.py").read_text()
    assert "lazy_step_hook" in (root / "ops/hpu_situ_moe.py").read_text()


def test_compile_mode_gives_no_hook_without_touching_habana(monkeypatch, fake_habana):
    monkeypatch.setenv("PT_HPU_LAZY_MODE", "0")
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", None)  # an import would raise
    with gs.forward_graph_mode(use_graphs=False):
        assert gs.lazy_step_hook() is None
