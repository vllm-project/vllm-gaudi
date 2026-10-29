# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests: the native MXFP4 MoE op traces into the caller's compiled region without graph breaks.

The op module is loaded with stubbed habana_frameworks / vllm imports and the HPU op call replaced by a CPU
stand-in, so only the dispatch logic of VllmMixtureOfExpertsOpMXFP4 is exercised.
"""

import importlib.util
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
import torch._dynamo as dynamo
from torch._dynamo.testing import CompileCounter

_OPS_PATH = pathlib.Path(__file__).resolve().parents[3] / "vllm_gaudi" / "extension" / "ops.py"
_STUBBED_MODULES = (
    "habana_frameworks",
    "habana_frameworks.torch",
    "habana_frameworks.torch.core",
    "habana_frameworks.torch.utils",
    "habana_frameworks.torch.utils.experimental",
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.quantization",
    "vllm.model_executor.layers.quantization.utils",
    "vllm.model_executor.layers.quantization.base_config",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.fused_moe.activation",
    "vllm_gaudi.extension.runtime",
    "vllm_gaudi.extension.utils",
    "vllm_gaudi.extension.scales",
)


@pytest.fixture
def ops(monkeypatch):
    for name in _STUBBED_MODULES:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    sys.modules["vllm_gaudi.extension.runtime"].get_config = lambda: types.SimpleNamespace(moe_chunk=[],
                                                                                           moe_token_boundary=[])
    sys.modules["vllm.model_executor.layers.fused_moe.activation"].MoEActivation = type("MoEActivation", (), {})
    spec = importlib.util.spec_from_file_location("mxfp4_ops_under_test", _OPS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def cpu_fused_fwd(hidden_states, expert_routing_table, router_weights, w12_list, w3_list, d_scale_w12, d_scale_w3,
                      block_size, activation, experts_min, experts_max, chunk_size, total_experts):
        # touches every weight / scale tensor so that they become graph inputs as with the real op
        acc = sum(w.float().sum() for w in w12_list + w3_list + d_scale_w12 + d_scale_w3)
        return hidden_states * router_weights.sum(-1, keepdim=True) + acc

    monkeypatch.setattr(module, "_mxfp4_fused_fwd", cpu_fused_fwd)
    dynamo.reset()
    dynamo.utils.counters.clear()
    yield module
    dynamo.reset()


def _make_moe_op(ops, num_experts=4):
    op = ops.VllmMixtureOfExpertsOpMXFP4(global_num_experts=num_experts,
                                         num_total_experts=num_experts,
                                         experts_min=0,
                                         experts_max=num_experts - 1)
    for i in range(num_experts):
        op.w13_list[i].set_weight(torch.full((8, 4), i, dtype=torch.uint8))
        op.w13_list[i].set_scale(torch.ones(8, 1, dtype=torch.uint8))
        op.w2_list[i].set_weight(torch.full((8, 2), i, dtype=torch.uint8))
        op.w2_list[i].set_scale(torch.ones(8, 1, dtype=torch.uint8))
    # as process_weights_after_loading does (ops/hpu_mxfp4.py); the lazy fill in forward would recompile once
    op._cache_weight_lists()
    return op


class _Layer(torch.nn.Module):
    """Stand-in for a decoder layer that calls the MoE op inside its regionally compiled forward."""

    def __init__(self, moe_op):
        super().__init__()
        self.moe_op = moe_op

    def forward(self, hidden_states, routing, weights):
        hidden_states = hidden_states + 1
        out = self.moe_op(hidden_states, routing, weights)
        return out * 2


def _inputs():
    return torch.randn(3, 8), torch.zeros(3, 2, dtype=torch.int64), torch.rand(3, 2)


def test_first_trace_of_every_layer_has_no_graph_break(ops):
    # Regional compilation: every layer is compiled separately but they share one code object. Before the fix the
    # first call of each layer built the nested torch.compile wrapper inside the trace, which broke the graph and
    # left the layer split into nested frames.
    layers = [_Layer(_make_moe_op(ops)) for _ in range(3)]
    counter = CompileCounter()
    compiled = [torch.compile(layer, backend=counter, fullgraph=True) for layer in layers]

    for _ in range(2):
        for layer in compiled:
            layer(*_inputs())

    assert not dynamo.utils.counters["graph_break"]
    assert counter.frame_count == 1  # one graph for all layers, no per-layer recompile
    assert all(layer.moe_op._compiled_forward is None for layer in layers)


def test_eager_call_still_compiles_the_op_once(ops, monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    op = _make_moe_op(ops)
    hidden_states, routing, weights = _inputs()

    first = op(hidden_states, routing, weights)
    op(hidden_states, routing, weights)

    assert len(calls) == 1
    assert calls[0][0] is ops._mxfp4_fused_fwd
    assert calls[0][1] == {"backend": "hpu_backend"}
    expected = hidden_states * weights.sum(-1, keepdim=True)
    assert first.shape == expected.shape
