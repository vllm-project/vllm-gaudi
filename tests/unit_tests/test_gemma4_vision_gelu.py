# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Gemma4 vision GELU precision under HPU fusion."""

from types import SimpleNamespace

import pytest
import torch
from transformers.activations import GELUTanh

from vllm_gaudi.models.gemma4 import HpuGemma4VisionGELU, _apply_hpu_gemma4_vision_patches


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_cpu_fallback(dtype):
    x = torch.linspace(-8, 8, 1024, dtype=dtype)
    torch.testing.assert_close(HpuGemma4VisionGELU()(x), GELUTanh()(x), rtol=0, atol=0)


def test_vision_patch_scope():
    gelu_layer = torch.nn.Module()
    gelu_layer.mlp = torch.nn.Module()
    gelu_layer.mlp.act_fn = GELUTanh()
    other_layer = torch.nn.Module()
    other_layer.mlp = torch.nn.Module()
    other_layer.mlp.act_fn = torch.nn.ReLU()
    original_relu = other_layer.mlp.act_fn
    vision = SimpleNamespace(encoder=SimpleNamespace(layers=[gelu_layer, other_layer]))
    _apply_hpu_gemma4_vision_patches(None)
    _apply_hpu_gemma4_vision_patches(vision)
    _apply_hpu_gemma4_vision_patches(vision)
    assert isinstance(gelu_layer.mlp.act_fn, HpuGemma4VisionGELU)
    assert other_layer.mlp.act_fn is original_relu
    assert gelu_layer.state_dict() == {}


@pytest.mark.parametrize("batch", [1, 4])
def test_compiled_bf16_matches_eager(batch):
    pytest.importorskip("habana_frameworks.torch.core")
    if not torch.hpu.is_available():
        pytest.skip("Requires HPU")

    # Cover all finite BF16 inputs, including the small positive values where
    # fused GELU can change tanh rounding. Reshape and multiplication exercise
    # fusion with adjacent operations rather than standalone GELU compilation.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)].reshape(1, -1)
    x = values.repeat(batch, 1).to("hpu")
    up = torch.ones_like(x)
    expected = GELUTanh()(x) * up
    activation = HpuGemma4VisionGELU()

    def fused(gate, multiplier):
        return activation(gate.reshape(batch, -1)) * multiplier

    compiled = torch.compile(fused, backend="hpu_backend", fullgraph=True, dynamic=False)
    actual = compiled(x, up)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(compiled(x, up).cpu(), actual.cpu(), rtol=0, atol=0, equal_nan=True)


def test_hpu_fp32_fallback():
    pytest.importorskip("habana_frameworks.torch.core")
    if not torch.hpu.is_available():
        pytest.skip("Requires HPU")
    x = torch.linspace(-8, 8, 1024, device="hpu")
    torch.testing.assert_close(HpuGemma4VisionGELU()(x).cpu(), GELUTanh()(x).cpu(), rtol=0, atol=0)
