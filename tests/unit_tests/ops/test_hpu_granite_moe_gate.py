# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HPU plugin builds the Granite MoE router as bf16 ReplicatedLinear.

Run with:
    pytest tests/unit_tests/ops/test_hpu_granite_moe_gate.py -v
"""

import inspect

import torch
from vllm.model_executor.layers.linear import ReplicatedLinear

from vllm_gaudi import patches


def test_granite_moe_gate_still_built_with_gate_linear():
    import vllm.model_executor.models.granitemoe as granite_moe

    source = inspect.getsource(granite_moe.GraniteMoeMoE.__init__)
    assert "GateLinear(" in source, ("GraniteMoeMoE no longer builds its gate with GateLinear; "
                                     "update vllm_gaudi.patches._patch_granite_moe_gate")


def test_granite_moe_gate_is_bf16_replicated_linear(default_vllm_config, dist_init, monkeypatch):
    import vllm.model_executor.models.granitemoe as granite_moe

    monkeypatch.setattr(granite_moe, "GateLinear", granite_moe.GateLinear)
    patches._patch_granite_moe_gate()

    gate = granite_moe.GateLinear(64, 8, params_dtype=torch.bfloat16, prefix="model.layers.0.block_sparse_moe.gate")

    assert type(gate) is ReplicatedLinear
    assert gate.bias is None
    assert gate.weight.dtype == torch.bfloat16


def test_granite_moe_gate_ignores_extra_gate_linear_args(default_vllm_config, dist_init, monkeypatch):
    import vllm.model_executor.models.granitemoe as granite_moe

    monkeypatch.setattr(granite_moe, "GateLinear", granite_moe.GateLinear)
    patches._patch_granite_moe_gate()

    gate = granite_moe.GateLinear(64,
                                  8,
                                  params_dtype=torch.bfloat16,
                                  out_dtype=torch.float32,
                                  force_fp32_compute=True,
                                  prefix="model.layers.0.block_sparse_moe.gate")

    assert type(gate) is ReplicatedLinear
    assert gate.weight.dtype == torch.bfloat16
