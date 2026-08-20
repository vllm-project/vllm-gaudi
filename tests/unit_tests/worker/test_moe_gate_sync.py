# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fused_moe.layer import MoERunner

from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner


class _BareMoERunner(MoERunner):
    """MoERunner with the heavyweight vLLM-config-dependent __init__ skipped."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.moe_config = SimpleNamespace()
        self.gate = None


class _StubModelRunner:
    _sync_shared_moe_gates = HPUModelRunner._sync_shared_moe_gates

    def __init__(self, layers):
        self._layers = layers
        self._detached_moe_gates: set[int] = set()

    def _get_model_layers(self):
        return self._layers


def _build_layer():
    gate = torch.nn.Linear(8, 4, bias=False)
    experts = _BareMoERunner()
    mlp = SimpleNamespace(gate=gate, experts=experts)
    return SimpleNamespace(mlp=mlp), gate, experts


def test_sync_shared_moe_gates_restores_runner_gate():
    """The runner owns gate application, so INC sync must not clear it.

    Models pass ``router_logits=hidden_states`` as a placeholder; a runner
    without a gate forwards that placeholder into expert selection.
    """
    layer, gate, experts = _build_layer()
    # _remove_duplicate_submodules() detaches the gate from the runner's
    # _modules before INC conversion so INC patches it only under mlp.
    experts._modules.pop("gate", None)
    object.__setattr__(experts, "gate", None)

    _StubModelRunner([layer])._sync_shared_moe_gates()

    assert experts.gate is gate
    assert "gate" not in experts._modules
