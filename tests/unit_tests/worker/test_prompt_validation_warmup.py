# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the validation warmup, which runs buckets again at the end of warmup.

Smaller prompt buckets make Dynamo compile more general graphs, which inference
then selects for the larger buckets warmed before them. The validation warmup
runs prompt buckets again at the end, so that those graphs have recipes for
their shapes: the largest seq_len for models with heterogeneous layers, every
bucket with prompt_validation_warmup (on by default for Nemotron-H with prefix
caching in torch.compile mode), in compile-only mode unless the layers are
heterogeneous.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_gaudi.extension import environment
from vllm_gaudi.extension.config import Config
from vllm_gaudi.extension.features import get_features
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

BUCKETS = [(1, 4096, 0), (1, 4096, 32), (1, 3072, 0), (1, 2048, 0), (1, 2048, 32)]
DECODE_BUCKETS = [(1, 1, 128), (8, 1, 512)]
KV_CACHES = object()


def _prompt_validation_warmup(**values):
    values.setdefault("bridge_mode", "eager")
    values.setdefault("enable_prefix_caching", True)
    feature_values, feature_flags = get_features()
    return Config(feature_values, feature_flags, **values).get('prompt_validation_warmup')


@pytest.mark.parametrize("model_type, bridge_mode, prefix_caching, enabled", [
    ("nemotron_h", "eager", True, True),
    ("nemotron_h", "eager", False, False),
    ("nemotron_h", "lazy", True, False),
    ("llama", "eager", True, False),
    ("granitemoehybrid", "eager", True, False),
])
def test_prompt_validation_warmup_default(model_type, bridge_mode, prefix_caching, enabled):
    assert _prompt_validation_warmup(model_type=model_type,
                                     bridge_mode=bridge_mode,
                                     enable_prefix_caching=prefix_caching) is enabled


@pytest.mark.parametrize("model_type, bridge_mode, prefix_caching, override", [
    ("nemotron_h", "eager", True, False),
    ("nemotron_h", "eager", False, True),
    ("nemotron_h", "lazy", True, True),
    ("llama", "eager", True, True),
])
def test_prompt_validation_warmup_env_override(model_type, bridge_mode, prefix_caching, override):
    assert _prompt_validation_warmup(model_type=model_type,
                                     bridge_mode=bridge_mode,
                                     enable_prefix_caching=prefix_caching,
                                     VLLM_PROMPT_VALIDATION_WARMUP=override) is override


@pytest.mark.parametrize("enabled", [False, True])
def test_enable_prefix_caching_is_the_engine_setting(monkeypatch, enabled):
    # prefix_caching stays True unless VLLM_CONTIGUOUS_PA is set, so the default reads the engine setting.
    monkeypatch.setattr(environment, "_VLLM_VALUES", {})
    for env_var in ("VLLM_CONTIGUOUS_PA", "VLLM_PREFIX_CACHING", "VLLM_ENABLE_PREFIX_CACHING"):
        monkeypatch.delenv(env_var, raising=False)
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="nemotron_h")),
                                  cache_config=SimpleNamespace(enable_prefix_caching=enabled))
    environment.set_vllm_config(vllm_config)
    config = Config(*environment.get_environment())
    assert config.get('enable_prefix_caching') is enabled
    assert config.get('prefix_caching') is True


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
