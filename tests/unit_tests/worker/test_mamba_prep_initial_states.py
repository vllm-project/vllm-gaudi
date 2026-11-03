# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the per-model mamba prefill graph split.

Granite 4.0-H compiles each mamba layer's prefill as two graphs, split by a
graph break at the initial-state branch; every other model, Nemotron-H
included, keeps one merged graph.
"""

import pytest
import torch

from vllm_gaudi.extension.config import Config
from vllm_gaudi.extension.features import get_features
from vllm_gaudi.v1.worker.hpu_model_runner import mamba_prep_initial_states


def _split_prefill_graph(**values):
    feature_values, feature_flags = get_features()
    return Config(feature_values, feature_flags, **values).get('mamba_split_prefill_graph')


@pytest.mark.parametrize("model_type, split", [("granitemoehybrid", True), ("nemotron_h", False)])
def test_split_prefill_graph_default(model_type, split):
    assert _split_prefill_graph(model_type=model_type) is split


def test_split_prefill_graph_off_under_fullgraph():
    assert _split_prefill_graph(model_type="granitemoehybrid", VLLM_T_COMPILE_FULLGRAPH=True) is False


@pytest.mark.parametrize("model_type, override", [("granitemoehybrid", False), ("nemotron_h", True)])
def test_split_prefill_graph_env_override(model_type, override):
    assert _split_prefill_graph(model_type=model_type, VLLM_MAMBA_SPLIT_PREFILL_GRAPH=override) is override


@pytest.mark.parametrize("has_initial_states", [[False, False], [False, True]])
def test_prep_initial_states_form(has_initial_states):
    states = torch.tensor(has_initial_states)
    merged = mamba_prep_initial_states(states, split_prefill_graph=False)
    split = mamba_prep_initial_states(states, split_prefill_graph=True)
    assert type(merged) is bool
    assert isinstance(split, torch.Tensor) and split.dim() == 0 and split.device.type == "cpu"
    assert merged == bool(split) == any(has_initial_states)


def _mixer_prefill(x, prep_initial_states):
    # The shape of HPUMambaMixer2's prefill: conv part, initial-state branch,
    # SSM part.
    x = x * 2
    if prep_initial_states:
        x = x + 1
    return x - 1


@pytest.mark.parametrize("split_prefill_graph, graphs", [(False, 1), (True, 2)])
def test_graph_break_only_when_split(split_prefill_graph, graphs):
    prep_initial_states = mamba_prep_initial_states(torch.tensor([False, True]), split_prefill_graph)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(_mixer_prefill)(torch.ones(4), prep_initial_states)
    assert explanation.graph_count == graphs
    assert explanation.graph_break_count == graphs - 1
