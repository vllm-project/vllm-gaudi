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

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config):
        return hidden_size, intermediate_size_per_partition


@pytest.fixture
def mxfp4(monkeypatch):
    state = {"kimi": False, "apply_calls": []}

    class Mxfp4MoEMethod(_Base):
        supports_pre_processed_weights = True
        supports_eplb = True

        def maybe_roundup_sizes(self, *args, **kwargs):
            raise AssertionError("the backend-keyed CUDA rounding must not run on HPU")

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
    mod.orig_mxfp4_method = Mxfp4MoEMethod
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
    assert mxfp4.upstream.Mxfp4MoEMethod is mxfp4.orig_mxfp4_method
    assert mxfp4.upstream.Mxfp4MoEMethod is not mxfp4.HPUKimiK3Mxfp4MoEMethod


def test_kimi_method_init_sets_the_hpu_attributes(mxfp4):
    moe = types.SimpleNamespace(max_capture_size=8)

    method = mxfp4.HPUKimiK3Mxfp4MoEMethod(moe, activation_key="key")

    assert method.moe is moe
    assert method.weight_dtype == "mxfp4"
    assert method.activation_key == "key"
    assert method.group_size == 32 == mxfp4.HPUKimiK3Mxfp4MoEMethod.MXFP4_BLOCK_SIZE
    assert method.max_capture_size == 8
    assert method.use_dispatch_fn is False
    assert (method.is_k3_situ_aiter, method.is_k3_situ_int4_gfx942) == (False, False)
    assert method.mxfp4_backend is None and method.experts_cls is None and method.moe_kernel is None
    assert method.w13_precision_config is None and method.w2_precision_config is None
    assert method._cache_permute_indices == {}
    assert method.moe_quant_config is not None
    assert method.get_fused_moe_quant_config(object()) is None


def test_kimi_method_opts_out_of_eplb_and_pre_processed_weights(mxfp4):
    method = mxfp4.HPUKimiK3Mxfp4MoEMethod(types.SimpleNamespace(max_capture_size=8))

    # the upstream Mxfp4MoEMethod says True for both
    assert method.supports_eplb is False
    assert method.supports_pre_processed_weights is False
    assert mxfp4.orig_mxfp4_method.supports_pre_processed_weights is True
    assert (method.skip_forward_padding, method.has_unpadded_output, method.is_monolithic) == (False, False, False)


def test_kimi_method_rounding_does_not_use_the_backend_keyed_upstream_step(mxfp4):
    method = mxfp4.HPUKimiK3Mxfp4MoEMethod(types.SimpleNamespace(max_capture_size=8))

    # the stubbed upstream rounding raises; the base-class contract (identity here) is what runs
    rounded = method.maybe_roundup_sizes(hidden_size=4096,
                                         intermediate_size_per_partition=2048,
                                         act_dtype=None,
                                         moe_parallel_config=None)
    assert rounded == (4096, 2048)


def _fake_op_class(created):

    class _Slot:

        def __init__(self):
            self.calls = {}

        def set_weight(self, value):
            self.calls["weight"] = value

        def set_scale(self, value):
            self.calls["scale"] = value

        def set_bias(self, value):
            self.calls["bias"] = value

    class FakeOp:

        def __init__(self, global_experts, local_experts, experts_min, experts_max, **kwargs):
            self.args = (global_experts, local_experts, experts_min, experts_max)
            self.kwargs = kwargs
            self.w13_list = [_Slot() for _ in range(local_experts)]
            self.w2_list = [_Slot() for _ in range(local_experts)]
            self.cached = False
            created.append(self)

        def _cache_weight_lists(self):
            self.cached = True

    return FakeOp


def _layer(local=3, global_=6, ep_rank=1, dp_size=1):
    import torch
    return types.SimpleNamespace(
        local_num_experts=local,
        global_num_experts=global_,
        moe_config=types.SimpleNamespace(ep_rank=ep_rank, dp_size=dp_size, is_sequence_parallel=False),
        w13_weight=torch.nn.Parameter(torch.zeros(local, 4, 2, dtype=torch.uint8), requires_grad=False),
        w13_weight_scale=torch.nn.Parameter(torch.ones(local, 4, 1, dtype=torch.uint8), requires_grad=False),
        w2_weight=torch.nn.Parameter(torch.zeros(local, 2, 4, dtype=torch.uint8), requires_grad=False),
        w2_weight_scale=torch.nn.Parameter(torch.ones(local, 2, 1, dtype=torch.uint8), requires_grad=False),
    )


def test_kimi_method_process_weights_builds_the_hpu_op_from_the_loaded_params(mxfp4):
    created = []
    mxfp4.VllmMixtureOfExpertsOpMXFP4 = _fake_op_class(created)
    mxfp4.state["kimi"] = True
    method = mxfp4.HPUKimiK3Mxfp4MoEMethod(types.SimpleNamespace(max_capture_size=8))
    layer = _layer(local=3, global_=6, ep_rank=1)
    placeholder = method.moe_quant_config

    method.process_weights_after_loading(layer)

    (op, ) = created
    assert layer.moe_op is op and op.cached
    assert op.args == (6, 3, 3, 5)  # experts_min/max = ep_rank * local .. + local - 1
    assert op.kwargs["block_size"] == 32 and op.kwargs["has_bias"] is False
    assert op.kwargs["dispatch_fn"] is None
    assert op.kwargs["in_graph_call"] is True
    for i in range(3):
        assert op.w13_list[i].calls["weight"].shape == (4, 2)
        assert op.w13_list[i].calls["scale"].shape == (4, 1)
        assert op.w2_list[i].calls["weight"].shape == (2, 4)
        assert "bias" not in op.w13_list[i].calls
    assert method.moe_quant_config is not placeholder


def test_kimi_method_process_weights_in_graph_call_only_for_kimi_configs(mxfp4):
    created = []
    mxfp4.VllmMixtureOfExpertsOpMXFP4 = _fake_op_class(created)
    mxfp4.state["kimi"] = False
    method = mxfp4.HPUKimiK3Mxfp4MoEMethod(types.SimpleNamespace(max_capture_size=8))

    method.process_weights_after_loading(_layer(local=1, global_=1, ep_rank=0))

    assert created[0].kwargs["in_graph_call"] is False
