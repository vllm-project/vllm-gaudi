# SPDX-License-Identifier: Apache-2.0
"""Kimi vision tower position embedding keeps the weight dtype for video grids (t > 1).

Runs without vllm / habana: the upstream module is replaced by a stub that carries the same
position-embedding arithmetic (bf16 weight + float32 ``time_weight`` buffer for t > 1).
"""
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

PATH = Path(__file__).resolve().parents[2] / "vllm_gaudi" / "models" / "kimi_k25_vit.py"
NAME = "vllm_gaudi.models.kimi_k25_vit"
UPSTREAM = "vllm.model_executor.models.kimi_k25_vit"
DIM, H, W, FRAMES = 8, 3, 4, 4


class _PosEmb(nn.Module):
    """Same dtype behaviour as upstream Learnable2DInterpPosEmbDivided_fixed.get_pos_embeds."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(H, W, DIM))
        self.register_buffer("time_weight", torch.ones(FRAMES, 1, DIM).float(), persistent=False)

    def get_pos_embeds(self, grid_thws):
        out = []
        for t, h, w in grid_thws:
            pos = self.weight.flatten(end_dim=1)
            if t > 1:
                pos = pos.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
            out.append(pos.reshape(-1, DIM))
        return torch.cat(out)


@pytest.fixture
def pos_emb_cls(monkeypatch):
    stub: Any = types.ModuleType(UPSTREAM)
    stub.Learnable2DInterpPosEmbDivided_fixed = type("PosEmb", (_PosEmb, ), {})
    bridge: Any = types.ModuleType("habana_frameworks.torch.internal.bridge_config")
    bridge.get_pt_compile_only_mode = lambda: False
    for name, mod in {
            "habana_frameworks": types.ModuleType("habana_frameworks"),
            "habana_frameworks.torch": types.ModuleType("habana_frameworks.torch"),
            "habana_frameworks.torch.internal": types.ModuleType("habana_frameworks.torch.internal"),
            "habana_frameworks.torch.internal.bridge_config": bridge,
            "vllm": types.ModuleType("vllm"),
            "vllm.model_executor": types.ModuleType("vllm.model_executor"),
            "vllm.model_executor.models": types.ModuleType("vllm.model_executor.models"),
            UPSTREAM: stub,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, NAME, raising=False)
    spec = importlib.util.spec_from_file_location(NAME, PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, NAME, mod)
    spec.loader.exec_module(mod)
    return stub.Learnable2DInterpPosEmbDivided_fixed


def test_video_grid_pos_embeds_keep_the_bf16_weight_dtype(pos_emb_cls):
    m = pos_emb_cls().to(torch.bfloat16)
    m.time_weight = m.time_weight.float()  # buffer left in float32, as when the tower is not cast
    out = m.get_pos_embeds([[4, H, W]])
    assert out.dtype == torch.bfloat16
    assert out.shape == (4 * H * W, DIM)


def test_image_grid_pos_embeds_are_unchanged(pos_emb_cls):
    m = pos_emb_cls().to(torch.bfloat16)
    assert m.get_pos_embeds([[1, H, W]]).dtype == torch.bfloat16
