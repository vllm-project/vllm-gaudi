# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INC quantizes GateLinear MoE routers like ReplicatedLinear ones.

Run with:
    pytest tests/unit_tests/worker/test_inc_gate_linear.py -v
"""

import pytest
from vllm.model_executor.layers.fused_moe import GateLinear


def test_inc_patches_gate_linear_like_replicated_linear():
    pytest.importorskip("neural_compressor")
    from neural_compressor.torch.algorithms.fp8_quant.model_configs import get_patched_module_table

    from vllm_gaudi.v1.worker.hpu_model_runner import _apply_inc_patch

    _apply_inc_patch()
    _apply_inc_patch()

    table = get_patched_module_table()
    assert table[GateLinear.__name__] is table["ReplicatedLinear"]
