# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_gaudi.extension.runtime import clear_config, finalize_config, get_config

# Values that a real boot derives from the installed Habana stack (device,
# plugin build, engine version) are injected through get_config(**overrides) —
# the seam get_config already exposes — so the guard logic can be exercised
# on any machine. The flag under test, PT_HPU_EAGER_PIPELINE_ENABLE, flows
# through the real parse surface: it is read from os.environ at finalize
# time, exactly as at boot.

_HERMETIC_BOOT = {
    'hw': 'cpu',
    'build': '1.22.0.494',
    'engine_version': 'unknown',
}


@pytest.fixture(autouse=True)
def reset_runtime_config():
    clear_config()
    yield
    clear_config()


def _finalize_boot(monkeypatch, bridge_mode, **boot_env):
    for name, value in boot_env.items():
        monkeypatch.setenv(name, value)
    get_config(**_HERMETIC_BOOT, bridge_mode=bridge_mode)
    return finalize_config()


@pytest.mark.parametrize('value', ['0', 'false', 'off', 'no'])
def test_finalize_rejects_disabled_eager_pipeline_in_eager_mode(monkeypatch, value):
    # Disabling the Habana eager pipeline on the eager backend is the
    # silent-death combination from issue #1855: streamed requests hang
    # mid-generation with no server-side error. Startup must fail loudly.
    with pytest.raises(RuntimeError) as excinfo:
        _finalize_boot(monkeypatch, bridge_mode='eager', PT_HPU_EAGER_PIPELINE_ENABLE=value)
    message = str(excinfo.value)
    assert 'PT_HPU_EAGER_PIPELINE_ENABLE' in message
    assert 'issues/1855' in message


def test_finalize_allows_disabled_eager_pipeline_in_lazy_mode(monkeypatch):
    # The eager pipeline only applies to the eager backend; lazy mode with the
    # knob disabled must keep booting unharmed.
    config = _finalize_boot(monkeypatch, bridge_mode='lazy', PT_HPU_EAGER_PIPELINE_ENABLE='0')
    assert config is not None


@pytest.mark.parametrize('value', ['1', 'true', 'on'])
def test_finalize_allows_explicitly_enabled_eager_pipeline_in_eager_mode(monkeypatch, value):
    config = _finalize_boot(monkeypatch, bridge_mode='eager', PT_HPU_EAGER_PIPELINE_ENABLE=value)
    assert config is not None


def test_finalize_allows_default_eager_pipeline_in_eager_mode(monkeypatch):
    monkeypatch.delenv('PT_HPU_EAGER_PIPELINE_ENABLE', raising=False)
    config = _finalize_boot(monkeypatch, bridge_mode='eager')
    assert config is not None


def test_finalize_treats_empty_eager_pipeline_value_as_unset(monkeypatch):
    # An empty string (an env template that expanded to nothing) is not an
    # explicit disable; the guard is scoped to the reported off-setting.
    config = _finalize_boot(monkeypatch, bridge_mode='eager', PT_HPU_EAGER_PIPELINE_ENABLE='')
    assert config is not None
