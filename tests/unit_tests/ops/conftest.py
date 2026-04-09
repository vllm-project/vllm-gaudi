# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config


@pytest.fixture
def default_vllm_config():
    """VllmConfig with a minimal model_config stub.

    Upstream Fp8LinearMethod.__init__ accesses model_config.dtype.
    We provide a SimpleNamespace with the attributes required by quantization
    methods so that ops-level unit tests can run without a full model setup.
    """
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16, is_moe=False, hf_config=None, quantization=None)

    with set_current_vllm_config(vllm_config):
        yield
