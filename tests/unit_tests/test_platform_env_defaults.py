# SPDX-License-Identifier: Apache-2.0
"""Unit tests for HPU compile-related env var defaults.

These guard two regressions:

* GAUDISW-248809: eager-mode defaults (e.g. ``RUNTIME_SCALE_PATCHING=1``) must
  not be set at plugin-registration / import time, otherwise they leak into a
  lazy-mode ``vllm serve`` subprocess spawned by a parent that merely imported
  ``vllm_gaudi`` (e.g. a pytest collection process) and regress throughput.
* GAUDISW-249135: an env var the user set explicitly must never be overwritten
  or removed.

The eager-only defaults now live in ``set_compile_env_defaults()`` which is
called from ``check_and_update_config()`` (engine construction time), while
``set_torch_compile()`` (import time) only sets mode-agnostic / lazy-only vars.
"""
import os
from unittest import mock

import pytest
import torch

from vllm_gaudi.platform import HpuPlatform

_EAGER_ONLY_VARS = (
    "RUNTIME_SCALE_PATCHING",
    "FUSER_ENABLE_MULTI_THREADED_INVOCATIONS",
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Isolate the env vars and global torch state this test touches."""
    touched = (
        "PT_HPU_WEIGHT_SHARING",
        "PT_HPU_ENABLE_LAZY_COLLECTIVES",
        *_EAGER_ONLY_VARS,
    )
    # set_torch_compile() flips torch._dynamo.config.disable in lazy mode;
    # snapshot it so the change does not leak into other tests.
    saved_dynamo_disable = torch._dynamo.config.disable
    with mock.patch.dict(os.environ, clear=False):
        for var in touched:
            os.environ.pop(var, None)
        try:
            yield
        finally:
            torch._dynamo.config.disable = saved_dynamo_disable


def _set_lazy(is_lazy: bool):
    return mock.patch("vllm_gaudi.platform.htorch.utils.internal.is_lazy", return_value=is_lazy)


@pytest.mark.parametrize("is_lazy", [True, False])
def test_set_torch_compile_never_sets_eager_only_vars(is_lazy):
    """Import-time hook must not set eager-only vars in ANY mode (no leak)."""
    with _set_lazy(is_lazy):
        HpuPlatform.set_torch_compile()
    for var in _EAGER_ONLY_VARS:
        assert var not in os.environ, f"{var} leaked from set_torch_compile (is_lazy={is_lazy})"


def test_set_compile_env_defaults_eager_sets_defaults():
    with _set_lazy(False):
        HpuPlatform.set_compile_env_defaults()
    assert os.environ.get("RUNTIME_SCALE_PATCHING") == "1"
    assert os.environ.get("FUSER_ENABLE_MULTI_THREADED_INVOCATIONS") == "1"


def test_set_compile_env_defaults_lazy_is_noop():
    with _set_lazy(True):
        HpuPlatform.set_compile_env_defaults()
    for var in _EAGER_ONLY_VARS:
        assert var not in os.environ


def test_set_compile_env_defaults_respects_user_values():
    """User-set values must not be overwritten (GAUDISW-249135)."""
    os.environ["RUNTIME_SCALE_PATCHING"] = "0"
    os.environ["FUSER_ENABLE_MULTI_THREADED_INVOCATIONS"] = "0"
    with _set_lazy(False):
        HpuPlatform.set_compile_env_defaults()
    assert os.environ["RUNTIME_SCALE_PATCHING"] == "0"
    assert os.environ["FUSER_ENABLE_MULTI_THREADED_INVOCATIONS"] == "0"


def test_no_leak_from_eager_parent_into_lazy_child():
    """End-to-end of the GAUDISW-248809 scenario.

    An eager parent process (e.g. pytest collection) imports vllm_gaudi and runs
    set_torch_compile(). A lazy child then builds an engine. The eager-only var
    must not be present for the lazy engine.
    """
    # Eager parent merely imports / registers the plugin.
    with _set_lazy(False):
        HpuPlatform.set_torch_compile()
    assert "RUNTIME_SCALE_PATCHING" not in os.environ

    # Lazy child builds an engine.
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
        HpuPlatform.set_compile_env_defaults()
    assert "RUNTIME_SCALE_PATCHING" not in os.environ


def test_user_value_survives_lazy_engine_build():
    """A user-set eager-only var is preserved even in a lazy engine build."""
    os.environ["RUNTIME_SCALE_PATCHING"] = "1"
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
        HpuPlatform.set_compile_env_defaults()
    assert os.environ["RUNTIME_SCALE_PATCHING"] == "1"


def test_pt_hpu_weight_sharing_default_and_respect():
    # Default-set when absent.
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
    assert os.environ["PT_HPU_WEIGHT_SHARING"] == "0"

    # User value respected.
    os.environ["PT_HPU_WEIGHT_SHARING"] = "1"
    with _set_lazy(False):
        HpuPlatform.set_torch_compile()
    assert os.environ["PT_HPU_WEIGHT_SHARING"] == "1"


def test_pt_hpu_enable_lazy_collectives_default_and_respect():
    # Default-set in lazy mode when absent.
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
    assert os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] == "true"

    # User value respected in lazy mode (GAUDISW-249135).
    os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "false"
    with _set_lazy(True):
        HpuPlatform.set_torch_compile()
    assert os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] == "false"
