# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the validation warmup, which runs buckets again at the end of warmup.

Smaller prompt buckets make Dynamo compile more general graphs, which inference
then selects for the larger buckets warmed before them. The validation warmup
runs prompt buckets again at the end, so that those graphs have recipes for
their shapes: the largest seq_len for models with heterogeneous layers, every
bucket with prompt_validation_warmup (on by default for Nemotron-H in
torch.compile mode), in compile-only mode unless the layers are heterogeneous.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_gaudi.extension.config import Config
from vllm_gaudi.extension.features import get_features
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

BUCKETS = [(1, 4096, 0), (1, 4096, 32), (1, 3072, 0), (1, 2048, 0), (1, 2048, 32)]
DECODE_BUCKETS = [(1, 1, 128), (8, 1, 512)]
KV_CACHES = object()


def _prompt_validation_warmup(**values):
    values.setdefault("bridge_mode", "eager")
    feature_values, feature_flags = get_features()
    return Config(feature_values, feature_flags, **values).get('prompt_validation_warmup')


@pytest.mark.parametrize("model_type, bridge_mode, enabled", [
    ("nemotron_h", "eager", True),
    ("nemotron_h", "lazy", False),
    ("llama", "eager", False),
    ("granitemoehybrid", "eager", False),
])
def test_prompt_validation_warmup_default(model_type, bridge_mode, enabled):
    assert _prompt_validation_warmup(model_type=model_type, bridge_mode=bridge_mode) is enabled


@pytest.mark.parametrize("model_type, bridge_mode, override", [
    ("nemotron_h", "eager", False),
    ("nemotron_h", "lazy", True),
    ("llama", "eager", True),
])
def test_prompt_validation_warmup_env_override(model_type, bridge_mode, override):
    assert _prompt_validation_warmup(model_type=model_type,
                                     bridge_mode=bridge_mode,
                                     VLLM_PROMPT_VALIDATION_WARMUP=override) is override


def test_default_reruns_the_largest_seq_len():
    assert HPUModelRunner._validation_prompt_buckets(BUCKETS, False) == [(1, 4096, 0), (1, 4096, 32)]


def test_all_buckets_rerun_every_bucket_in_warmup_order():
    # (1, 3072, 0) too: the graphs that (1, 2048, 0) compiles shadow it at inference.
    assert HPUModelRunner._validation_prompt_buckets(BUCKETS, True) == BUCKETS


@pytest.mark.parametrize("all_buckets", [False, True])
def test_no_prompt_buckets(all_buckets):
    assert HPUModelRunner._validation_prompt_buckets([], all_buckets) == []


class _WarmupRecorder:
    """Stands in for warmup_graphs and PT_COMPILE_ONLY_MODE, recording each call and the mode it ran in."""

    def __init__(self, runner):
        self.runner = runner
        self.compile_only = False
        self.calls = []

    @contextmanager
    def compile_only_mode_context(self):
        self.compile_only = True
        try:
            yield
        finally:
            self.compile_only = False

    def warmup_graphs(self, buckets, is_prompt, kv_caches):
        assert kv_caches is KV_CACHES
        # The pass clears graphed_buckets, or warmup_graphs would skip every bucket as done.
        assert not self.runner.graphed_buckets
        self.calls.append((list(buckets), is_prompt, self.compile_only))


def _run_validation_warmup(prompt_validation_warmup, heterogeneous, enforce_eager=False, is_pooling_model=False):
    runner = MagicMock(spec=HPUModelRunner)
    runner.model_config = SimpleNamespace(enforce_eager=enforce_eager)
    runner.is_pooling_model = is_pooling_model
    runner._has_heterogeneous_layers = heterogeneous
    graphed = {(1, 4096, 0, True), (1, 1, 128, False)}
    runner.graphed_buckets = set(graphed)
    runner.bucketing_manager = SimpleNamespace(prompt_buckets=BUCKETS, decode_buckets=DECODE_BUCKETS)
    runner._validation_prompt_buckets = HPUModelRunner._validation_prompt_buckets
    recorder = _WarmupRecorder(runner)
    runner.warmup_graphs = recorder.warmup_graphs
    config = SimpleNamespace(prompt_validation_warmup=prompt_validation_warmup)
    with patch("vllm_gaudi.v1.worker.hpu_model_runner.get_config", return_value=config):
        HPUModelRunner._validation_warmup(runner, KV_CACHES, recorder.compile_only_mode_context)
    assert runner.graphed_buckets == graphed
    return recorder.calls


def test_flag_compiles_every_prompt_bucket_in_compile_only_mode():
    # Nemotron-H: the shadowed graphs only lack recipes, so the forwards are not executed.
    assert _run_validation_warmup(True, heterogeneous=False) == [(BUCKETS, True, True), ([(1, 1, 128)], False, False)]


def test_flag_runs_every_prompt_bucket_for_heterogeneous_layers():
    assert _run_validation_warmup(True, heterogeneous=True) == [(BUCKETS, True, False), ([(1, 1, 128)], False, False)]


def test_heterogeneous_layers_without_flag_run_the_largest_seq_len():
    assert _run_validation_warmup(False, heterogeneous=True) == [([(1, 4096, 0), (1, 4096, 32)], True, False),
                                                                 ([(1, 1, 128)], False, False)]


def test_no_validation_warmup_without_flag_or_heterogeneous_layers():
    assert _run_validation_warmup(False, heterogeneous=False) == []


@pytest.mark.parametrize("enforce_eager, is_pooling_model", [(True, False), (False, True)])
def test_no_validation_warmup_in_eager_or_pooling(enforce_eager, is_pooling_model):
    assert _run_validation_warmup(True,
                                  heterogeneous=True,
                                  enforce_eager=enforce_eager,
                                  is_pooling_model=is_pooling_model) == []
