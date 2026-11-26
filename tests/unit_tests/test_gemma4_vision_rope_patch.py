# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
Gemma4VisionConfig = pytest.importorskip("transformers.models.gemma4").Gemma4VisionConfig

from vllm_gaudi.patches import (  # noqa: E402
    _hpu_gemma4_vision_rope_forward, _patch_gemma4_vision_rotary_embedding,
)


def _grid_positions(h: int = 48, w: int = 48) -> torch.Tensor:
    grid = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack(grid, dim=-1).reshape(1, -1, 2)


def _reference(rope, x, position_ids):
    freqs = position_ids[..., None].double() @ rope.inv_freq[None, ...].double()
    cos = rope.recomposition_frequencies(freqs.cos() * rope.attention_scaling)
    sin = rope.recomposition_frequencies(freqs.sin() * rope.attention_scaling)
    return cos.float().to(x.dtype), sin.float().to(x.dtype)


@pytest.fixture
def rope():
    _patch_gemma4_vision_rotary_embedding()
    return modeling_gemma4.Gemma4VisionRotaryEmbedding(Gemma4VisionConfig())


def test_patch_is_installed(rope):
    assert modeling_gemma4.Gemma4VisionRotaryEmbedding.forward is _hpu_gemma4_vision_rope_forward


def test_patch_is_idempotent(rope):
    _patch_gemma4_vision_rotary_embedding()
    assert modeling_gemma4.Gemma4VisionRotaryEmbedding.forward is _hpu_gemma4_vision_rope_forward


def test_matches_fp64_reference(rope):
    x = torch.zeros(1, 1, dtype=torch.float32)
    pos = _grid_positions()
    cos, sin = rope(x, pos)
    ref_cos, ref_sin = _reference(rope, x, pos)
    assert cos.shape == ref_cos.shape and cos.dtype == torch.float32
    torch.testing.assert_close(cos, ref_cos, atol=2e-5, rtol=0)
    torch.testing.assert_close(sin, ref_sin, atol=2e-5, rtol=0)


def test_angles_stay_fp32_under_bf16_autocast(rope):
    x = torch.zeros(1, 1, dtype=torch.float32)
    pos = _grid_positions()
    expected_cos, expected_sin = rope(x, pos)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        cos, sin = rope(x, pos)
    assert cos.dtype == torch.float32
    torch.testing.assert_close(cos, expected_cos, atol=0, rtol=0)
    torch.testing.assert_close(sin, expected_sin, atol=0, rtol=0)


def test_output_dtype_follows_input(rope):
    x = torch.zeros(1, 1, dtype=torch.bfloat16)
    cos, sin = rope(x, _grid_positions(8, 8))
    assert cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16
