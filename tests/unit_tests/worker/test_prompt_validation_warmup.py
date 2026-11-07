# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the prompt buckets the validation warmup runs again.

Smaller prompt buckets make Dynamo compile more general graphs, which inference
then selects for the larger buckets warmed before them. The validation warmup
runs prompt buckets again at the end, so that those graphs have recipes for
their shapes: the largest seq_len for models with heterogeneous layers, every
bucket with prompt_validation_warmup (on by default for Nemotron-H).
"""

import pytest

from vllm_gaudi.extension.config import Config
from vllm_gaudi.extension.features import get_features
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner

BUCKETS = [(1, 4096, 0), (1, 4096, 32), (1, 3072, 0), (1, 2048, 0), (1, 2048, 32)]


def _prompt_validation_warmup(**values):
    values.setdefault("bridge_mode", "eager")
    feature_values, feature_flags = get_features()
    return Config(feature_values, feature_flags, **values).get('prompt_validation_warmup')


@pytest.mark.parametrize("model_type, enabled", [("nemotron_h", True), ("llama", False), ("granitemoehybrid", False)])
def test_prompt_validation_warmup_default(model_type, enabled):
    assert _prompt_validation_warmup(model_type=model_type) is enabled


@pytest.mark.parametrize("model_type, override", [("nemotron_h", False), ("llama", True)])
def test_prompt_validation_warmup_env_override(model_type, override):
    assert _prompt_validation_warmup(model_type=model_type, VLLM_PROMPT_VALIDATION_WARMUP=override) is override


def test_default_reruns_the_largest_seq_len():
    assert HPUModelRunner._validation_prompt_buckets(BUCKETS, False) == [(1, 4096, 0), (1, 4096, 32)]


def test_all_buckets_rerun_every_bucket_in_warmup_order():
    # (1, 3072, 0) too: the graphs that (1, 2048, 0) compiles shadow it at inference.
    assert HPUModelRunner._validation_prompt_buckets(BUCKETS, True) == BUCKETS


@pytest.mark.parametrize("all_buckets", [False, True])
def test_no_prompt_buckets(all_buckets):
    assert HPUModelRunner._validation_prompt_buckets([], all_buckets) == []
