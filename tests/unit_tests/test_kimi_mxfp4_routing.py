# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 routed experts must get the HPU MXFP4 method when vLLM maps the checkpoint to ``Mxfp4Config``.

On vLLM 28a5fba06 ``Mxfp4Config._make_moe_method`` returns ``Mxfp4MoEMethod``, whose ``__init__`` calls
``select_deepseek_v4_mxfp4_moe_backend`` and raises ``NotImplementedError`` on HPU. Only Kimi-K3 may be re-routed:
gpt-oss (``GptOssMxfp4Config``) and any other model keep the upstream method.

Runs without vllm or HPU: ``ops/hpu_mxfp4.py`` is loaded from its file against stubbed modules.
"""
import importlib.util
import pathlib
import sys
import types

import pytest

_SOURCE = pathlib.Path(__file__).resolve().parents[2] / "vllm_gaudi" / "ops" / "hpu_mxfp4.py"


class _Base:
    """Stands for FusedMoEMethodBase: records the moe config."""

    def __init__(self, moe):
        self.moe = moe


@pytest.fixture
def mxfp4(monkeypatch):
    state = {"kimi": False, "apply_calls": []}

    class Mxfp4MoEMethod(_Base):

        def __init__(self, moe, activation_key=None):
            raise NotImplementedError("No MXFP4 MoE backend supports the deployment configuration.")

    class GptOssMxfp4MoEMethod(Mxfp4MoEMethod):
        pass

    class Mxfp4Config:

        def _make_moe_method(self, moe):
            return Mxfp4MoEMethod(moe)

    class GptOssMxfp4Config(Mxfp4Config):

        def _make_moe_method(self, moe):
            return GptOssMxfp4MoEMethod(moe)

    GptOssMxfp4MoEMethod.__init__ = lambda self, moe, activation_key=None: _Base.__init__(self, moe)

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    q = module("vllm.model_executor.layers.quantization")
    mxfp4_mod = module("vllm.model_executor.layers.quantization.mxfp4",
                       Mxfp4Config=Mxfp4Config,
                       Mxfp4MoEMethod=Mxfp4MoEMethod,
                       GptOssMxfp4Config=GptOssMxfp4Config,
                       GptOssMxfp4MoEMethod=GptOssMxfp4MoEMethod)
    q.mxfp4 = mxfp4_mod
    module("vllm.config", get_current_vllm_config=lambda: None)
    module("vllm.model_executor.layers.fused_moe.routed_experts", RoutedExperts=object)
    module("vllm_gaudi.envs")
    module("vllm_gaudi.extension.ops", VllmMixtureOfExpertsOpMXFP4=object)
    module("vllm_gaudi.extension.runtime", get_config=lambda: types.SimpleNamespace(use_dispatch_fn=False))
    module("vllm_gaudi.ops.hpu_situ_moe", MXFP4_BLOCK=32, maybe_situ_moe=None, situ_impl_or_exact=None)
    module("vllm_gaudi.ops.hpu_fused_moe", _normalize_moe_activation=None, select_experts_from_routed=None)
    module("vllm_gaudi.v1.worker.hpu_dp_utils",
           dispatch_hidden_states=None,
           dispatch_tensor=None,
           get_hpu_dp_metadata=None)
    module("vllm_gaudi.patches",
           _current_config_is_kimi_k3=lambda: state["kimi"],
           _hpu_mxfp4_routed_apply=lambda *args: state["apply_calls"].append(args) or "out")

    spec = importlib.util.spec_from_file_location("hpu_mxfp4_under_test", _SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.state = state
    mod.upstream = mxfp4_mod
    return mod


def _moe():
    return types.SimpleNamespace(max_capture_size=8)


def test_kimi_k3_gets_the_hpu_method_without_backend_selection(mxfp4):
    mxfp4.state["kimi"] = True

    method = mxfp4.upstream.Mxfp4Config()._make_moe_method(_moe())

    assert isinstance(method, mxfp4.HPUKimiK3Mxfp4MoEMethod)
    assert method.mxfp4_backend is None and method.experts_cls is None and method.moe_kernel is None
    assert method.is_monolithic is False
    assert method.maybe_roundup_sizes(4096, 2048, None, None) == (4096, 2048)


def test_other_models_keep_the_upstream_method(mxfp4):
    mxfp4.state["kimi"] = False

    with pytest.raises(NotImplementedError):
        mxfp4.upstream.Mxfp4Config()._make_moe_method(_moe())


def test_gpt_oss_config_is_untouched_even_inside_a_kimi_process(mxfp4):
    mxfp4.state["kimi"] = True

    method = mxfp4.upstream.GptOssMxfp4Config()._make_moe_method(_moe())

    # the stubbed config builds its own (upstream) gpt-oss method: the Kimi wrapper did not intercept it
    assert type(method).__name__ == "GptOssMxfp4MoEMethod"
    assert not isinstance(method, mxfp4.HPUKimiK3Mxfp4MoEMethod)


def test_routing_is_installed_once(mxfp4):
    config_cls = mxfp4.upstream.Mxfp4Config
    wrapped = config_cls._make_moe_method

    mxfp4._route_kimi_k3_moe_method()

    assert config_cls._make_moe_method is wrapped


def test_kimi_method_applies_through_the_shared_routed_helper(mxfp4):
    mxfp4.state["kimi"] = True
    method = mxfp4.upstream.Mxfp4Config()._make_moe_method(_moe())

    out = method.apply("layer", "x", "weights", "ids")

    assert out == "out"
    assert mxfp4.state["apply_calls"] == [("layer", "x", "weights", "ids", 32)]


def test_hpu_gpt_oss_alias_is_still_the_only_upstream_replacement(mxfp4):
    assert mxfp4.upstream.GptOssMxfp4MoEMethod is mxfp4.HPUGptOssMxfp4MoEMethod
    assert mxfp4.upstream.Mxfp4MoEMethod is not mxfp4.HPUKimiK3Mxfp4MoEMethod
