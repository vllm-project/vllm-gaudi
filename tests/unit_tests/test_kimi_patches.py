# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Kimi-K3 patch helpers in vllm_gaudi/patches.py against a stub of the upstream linear module."""

import importlib.util
import logging
import pathlib
import sys
import types

import pytest

_PATCHES = pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "patches.py"
LOGGER = "vllm.gaudi.kimi_patches"


@pytest.fixture
def patches(monkeypatch):
    # patches.py only needs ``from vllm import envs`` at import time; the rest is imported lazily by the helpers
    vllm = types.ModuleType("vllm")
    vllm.envs = types.ModuleType("vllm.envs")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", vllm.envs)
    spec = importlib.util.spec_from_file_location("hpu_patches_under_test", _PATCHES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_configure_situ_aliases", lambda: "exact")  # needs the whole plugin package
    return mod


def _stub_linear(monkeypatch):
    class KimiMoE:

        def __init__(self, *args, **kwargs):
            self.args = args

    class KimiDecoderLayer:

        def forward(self, x):
            return x

    original_forward = KimiDecoderLayer.forward
    mod = types.ModuleType("vllm.models.kimi_k3.amd.linear")
    mod.KimiMoE, mod.KimiDecoderLayer = KimiMoE, KimiDecoderLayer
    for name in ("vllm.models", "vllm.models.kimi_k3", "vllm.models.kimi_k3.amd"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.linear", mod)
    return mod, original_forward


def test_linear_weights_patch_installs_once_and_leaves_the_decoder_layer_alone(patches, monkeypatch, caplog):
    mod, original_forward = _stub_linear(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        patches._patch_kimi_linear_weights()
        patched_init = mod.KimiMoE.__init__
        patches._patch_kimi_linear_weights()
    assert mod.KimiMoE._patched_weights and mod.KimiMoE.__init__ is patched_init  # idempotent
    assert mod.KimiDecoderLayer.forward is original_forward  # the old pass-through wrapper is gone
    assert mod.KimiMoE(1, 2).args == (1, 2)
    assert patches._KIMI_PATCHES_INSTALLED.count("linear_weights") == 1
    assert caplog.records == []


def test_a_failing_patch_logs_a_warning_with_traceback(patches, monkeypatch, caplog):
    # no vllm.models.kimi_k3 at all: the import inside the helper fails
    monkeypatch.setitem(sys.modules, "vllm.models.kimi_k3.amd.linear", None)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        patches._patch_kimi_linear_weights()
    (record, ) = caplog.records
    assert record.name == LOGGER and record.levelno == logging.WARNING
    assert "linear_weights" in record.getMessage() and record.exc_info is not None
    assert "linear_weights" not in patches._KIMI_PATCHES_INSTALLED


def test_installed_patches_are_listed_in_one_info_line(patches, caplog):
    patches._kimi_patch_installed("a")
    patches._kimi_patch_installed("b")
    patches._kimi_patch_installed("a")
    with caplog.at_level(logging.INFO, logger=LOGGER):
        patches._kimi_patches_log_installed()
    assert [r.getMessage() for r in caplog.records] == ["Kimi-K3 patches installed: a, b"]
