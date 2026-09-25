# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn as nn
import habana_frameworks.torch as htorch
from types import SimpleNamespace
from typing import NamedTuple
from utils import temporary_op_registry_oot, register_op
from transformers.models.auto.configuration_auto import AutoConfig
from vllm_gaudi.utils import HPUCompileConfig
from vllm_gaudi.ops.hpu_rotary_embedding import (HPURotaryEmbedding, HPULinearScalingRotaryEmbedding,
                                                 HPUDynamicNTKScalingRotaryEmbedding, HPUYaRNScalingRotaryEmbedding,
                                                 HPUDeepseekScalingRotaryEmbedding, HPULlama3RotaryEmbedding,
                                                 HPUPhi3LongRoPEScaledRotaryEmbedding, HPULlama4VisionRotaryEmbedding,
                                                 HPUMRotaryEmbedding)
from vllm.model_executor.layers.rotary_embedding import (RotaryEmbedding, LinearScalingRotaryEmbedding,
                                                         DynamicNTKScalingRotaryEmbedding, YaRNScalingRotaryEmbedding,
                                                         DeepseekScalingRotaryEmbedding, Llama3RotaryEmbedding,
                                                         Llama4VisionRotaryEmbedding, MRotaryEmbedding)

# General settings
HIDDEN_SIZES = [4096]
SEQ_LENGTHS = [4096]
HEAD_SIZES = [32, 128, 512, 1024]
ROTARY_DIMS = [8, 32]
MAX_POSITION_EMBEDDINGS = [131072]
BASES = [500000.0]
IS_NEOX_STYLE = [False, True]
SCALING_FACTORS = [1.0, 2.0, 4.0, 8.0]
SCALING_FACTORS_WITH_LIST = [1.0, 2.0, 4.0, 8.0, [2.0, 4.0]]

# Vision model settings
IMAGE_SIZE = 336
PATCH_SIZE = 14
VISION_MAX_POSITION_EMBEDDINGS = [(IMAGE_SIZE // PATCH_SIZE)**2]
VISION_SEQ_LENGTHS = [x + 1 for x in VISION_MAX_POSITION_EMBEDDINGS]


class RotaryData(NamedTuple):
    """
    Data structure for rotary embedding test parameters.
    """
    cls: nn.Module
    dtype: torch.dtype
    device: str


def run_rotary_embedding_test(native_rotary_data: RotaryData, oot_rotary_data: RotaryData, seq_length: int,
                              hidden_size: int, **kwargs) -> None:
    """
    Common code for running rotary embedding tests. It compares output of
    native operator and out-of-tree custom operator. It allows to
    specify separate device for native operator and custom operator, 
    because for example native Llama4VisionRotaryEmbedding cannot be 
    used on hpu as it uses complex datatype. The same applies to dtype.
    """
    with temporary_op_registry_oot():
        # prepare native RotaryEmbedding module
        with torch.device(native_rotary_data.device):
            kwargs["dtype"] = native_rotary_data.dtype
            native_rotary_embedding = native_rotary_data.cls(**kwargs)
            assert isinstance(native_rotary_embedding,
                              native_rotary_data.cls) and not isinstance(native_rotary_embedding, oot_rotary_data.cls)

        # Prepare oot RotaryEmbedding module
        with torch.device(oot_rotary_data.device):
            register_op(native_rotary_data.cls, oot_rotary_data.cls)
            kwargs["dtype"] = oot_rotary_data.dtype
            oot_rotary_embedding = native_rotary_data.cls(**kwargs)  # Use native as it was registered above
            assert isinstance(oot_rotary_embedding, native_rotary_data.cls) and isinstance(
                oot_rotary_embedding, oot_rotary_data.cls)

            if not htorch.utils.internal.is_lazy():
                compile_config = HPUCompileConfig()
                oot_rotary_embedding = torch.compile(oot_rotary_embedding, **compile_config.get_compile_args())

        # Prepare input data
        positions = torch.randint(high=seq_length,
                                  size=(1, seq_length),
                                  dtype=torch.int32,
                                  device=native_rotary_data.device)
        query = torch.randn(1, seq_length, hidden_size, dtype=torch.bfloat16, device=native_rotary_data.device)
        key = torch.randn(1, seq_length, hidden_size, dtype=torch.bfloat16, device=native_rotary_data.device)
        if native_rotary_data.cls in (Llama4VisionRotaryEmbedding, DeepseekScalingRotaryEmbedding):
            query = query.view(query.shape[0], query.shape[1], -1, kwargs["head_size"])
            key = key.view(key.shape[0], key.shape[1], -1, kwargs["head_size"])

        # Execute layers
        with torch.device(native_rotary_data.device):
            if native_rotary_data.cls in (Llama4VisionRotaryEmbedding, ):
                ref_query_out, ref_key_out = native_rotary_embedding(query, key)
            elif native_rotary_data.cls in (MRotaryEmbedding, ):
                ref_query_out, ref_key_out = native_rotary_embedding(positions.flatten(), query, key)
            else:
                ref_query_out, ref_key_out = native_rotary_embedding(positions, query, key)

        if native_rotary_data.device != oot_rotary_data.device:
            positions = positions.to(oot_rotary_data.device)
            query = query.to(oot_rotary_data.device)
            key = key.to(oot_rotary_data.device)

        with torch.device(oot_rotary_data.device):
            if native_rotary_data.cls in (Llama4VisionRotaryEmbedding, ):
                query_out, key_out = oot_rotary_embedding(query, key)
            else:
                query_out, key_out = oot_rotary_embedding(positions, query, key)

        # Check correctness
        if native_rotary_data.device != oot_rotary_data.device:
            ref_query_out = ref_query_out.to("cpu")
            query_out = query_out.to("cpu")
            ref_key_out = ref_key_out.to("cpu")
            key_out = key_out.to("cpu")
        torch.testing.assert_close(query_out, ref_query_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(key_out, ref_key_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
def test_rotary_embedding(
    default_vllm_config: None,
    seq_length: int,
    hidden_size: int,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
) -> None:
    kwargs = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": max_position_embeddings,
        "base": base,
        "is_neox_style": is_neox_style,
    }
    native_rotary_data = RotaryData(cls=RotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    oot_rotary_data = RotaryData(cls=HPURotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    run_rotary_embedding_test(native_rotary_data, oot_rotary_data, seq_length, hidden_size, **kwargs)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("scaling_factors", SCALING_FACTORS_WITH_LIST)
def test_linear_scaling_rotary_embedding(
    default_vllm_config: None,
    seq_length: int,
    hidden_size: int,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    scaling_factors: float | list[float],
) -> None:
    kwargs = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": max_position_embeddings,
        "base": base,
        "is_neox_style": is_neox_style,
        "scaling_factors": scaling_factors,
    }
    native_rotary_data = RotaryData(cls=LinearScalingRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    oot_rotary_data = RotaryData(cls=HPULinearScalingRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    run_rotary_embedding_test(native_rotary_data, oot_rotary_data, seq_length, hidden_size, **kwargs)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("scaling_factor", SCALING_FACTORS)
def test_dynamic_ntk_scaling_rotary_embedding(
    default_vllm_config: None,
    seq_length: int,
    hidden_size: int,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    scaling_factor: float,
) -> None:
    kwargs = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": max_position_embeddings,
        "max_trained_positions": max_position_embeddings,
        "base": base,
        "is_neox_style": is_neox_style,
        "scaling_factor": scaling_factor,
    }
    native_rotary_data = RotaryData(cls=DynamicNTKScalingRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    oot_rotary_data = RotaryData(cls=HPUDynamicNTKScalingRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    run_rotary_embedding_test(native_rotary_data, oot_rotary_data, seq_length, hidden_size, **kwargs)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("scaling_factor", SCALING_FACTORS)
def test_yarn_scaling_rotary_embedding(
    default_vllm_config: None,
    seq_length: int,
    hidden_size: int,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    scaling_factor: float,
) -> None:
    kwargs = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": max_position_embeddings,
        "base": base,
        "is_neox_style": is_neox_style,
        "scaling_factor": scaling_factor,
    }
    native_rotary_data = RotaryData(cls=YaRNScalingRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    oot_rotary_data = RotaryData(cls=HPUYaRNScalingRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    run_rotary_embedding_test(native_rotary_data, oot_rotary_data, seq_length, hidden_size, **kwargs)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("scaling_factor", SCALING_FACTORS)
def test_deepseek_scaling_rotary_embedding(
    default_vllm_config: None,
    seq_length: int,
    hidden_size: int,
    head_size: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    scaling_factor: float,
) -> None:
    kwargs = {
        "head_size": head_size,
        "rotary_dim": head_size,
        "max_position_embeddings": max_position_embeddings,
        "base": base,
        "is_neox_style": is_neox_style,
        "scaling_factor": scaling_factor,
    }
    native_rotary_data = RotaryData(cls=DeepseekScalingRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    oot_rotary_data = RotaryData(cls=HPUDeepseekScalingRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    run_rotary_embedding_test(native_rotary_data, oot_rotary_data, seq_length, hidden_size, **kwargs)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
@pytest.mark.parametrize("scaling_factor", SCALING_FACTORS)
def test_llama3_rotary_embedding(
    default_vllm_config: None,
    seq_length: int,
    hidden_size: int,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    scaling_factor: float,
) -> None:
    kwargs = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": max_position_embeddings,
        "base": base,
        "is_neox_style": is_neox_style,
        "scaling_factor": scaling_factor,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "orig_max_position": 8192
    }
    native_rotary_data = RotaryData(cls=Llama3RotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    oot_rotary_data = RotaryData(cls=HPULlama3RotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    run_rotary_embedding_test(native_rotary_data, oot_rotary_data, seq_length, hidden_size, **kwargs)


# Positions as the HPU runner builds them: (batch_size, seq_length, valid tokens per row, pad value).
# Prompts are [bs, seq] padded with -1 (bucket tail and dummy rows); decode rows are [bs, 1] padded with 0.
PHI3_POSITION_LAYOUTS = {
    "prompt_bs1": (1, 1024, [1024], -1),
    "prompt_padded": (4, 256, [256, 200, 17, 0], -1),
    "decode": (8, 1, [1, 1, 1, 1, 1, 1, 0, 0], 0),
}


# (128, 96): Phi-4-mini partial rotary; (96, 96): Phi-3.5-mini full rotary. Both use 48 factors.
# max_model_len picks the short (4096) or long (16384) factors.
@pytest.mark.parametrize("head_size, rotary_dim", [(128, 96), (96, 96)])
@pytest.mark.parametrize("max_model_len", [4096, 16384])
@pytest.mark.parametrize("layout", PHI3_POSITION_LAYOUTS)
def test_phi3_long_rope_scaled_rotary_embedding(monkeypatch, head_size: int, rotary_dim: int, max_model_len: int,
                                                layout: str) -> None:
    # Upstream is not a CustomOp, so compare the two classes directly instead of via the OOT registry.
    import vllm.model_executor.layers.rotary_embedding as rope_pkg
    from vllm.model_executor.layers.rotary_embedding import phi3_long_rope_scaled_rope as phi3_rope
    assert rope_pkg.Phi3LongRoPEScaledRotaryEmbedding is HPUPhi3LongRoPEScaledRotaryEmbedding

    model_config = SimpleNamespace(max_model_len=max_model_len)
    monkeypatch.setattr(phi3_rope, "get_current_vllm_config", lambda: SimpleNamespace(model_config=model_config))
    config = AutoConfig.from_pretrained("microsoft/Phi-4-mini-instruct")
    kwargs = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "original_max_position_embeddings": config.original_max_position_embeddings,
        "base": 10000.0,
        "is_neox_style": True,
        "dtype": torch.bfloat16,
        "short_factor": config.rope_scaling["short_factor"],
        "long_factor": config.rope_scaling["long_factor"],
    }
    with torch.device("hpu"):
        native = phi3_rope.Phi3LongRoPEScaledRotaryEmbedding(**kwargs)
        oot = HPUPhi3LongRoPEScaledRotaryEmbedding(**kwargs)
    assert native.use_long_rope == oot.use_long_rope == (max_model_len > config.original_max_position_embeddings)

    batch_size, seq_length, valid_lens, pad_value = PHI3_POSITION_LAYOUTS[layout]
    hidden_size = 32 * head_size
    valid = torch.arange(seq_length, device="hpu") < torch.tensor(valid_lens, device="hpu").unsqueeze(-1)
    positions = torch.randint(high=max_model_len, size=(batch_size, seq_length), dtype=torch.int64, device="hpu")
    # The runner feeds the padding into prepare_cos_sin as is; padded rows are never compared.
    positions = torch.where(valid, positions, pad_value)
    query = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.bfloat16, device="hpu")
    key = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.bfloat16, device="hpu")
    # Upstream expects flat [T] positions / [T, hidden] q,k; the HPU runner passes [bs, seq] and [bs, seq, hidden].
    ref_query, ref_key = native(positions[valid], query[valid], key[valid])

    oot.prepare_cos_sin(positions)
    if not htorch.utils.internal.is_lazy():
        oot = torch.compile(oot, **HPUCompileConfig().get_compile_args())
    query_out, key_out = oot(positions, query, key)
    torch.testing.assert_close(query_out[valid], ref_query, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(key_out[valid], ref_key, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("seq_length", VISION_SEQ_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("max_position_embeddings", VISION_MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
def test_Llama4_vision_rotary_embedding(
    default_vllm_config: None,
    seq_length: int,
    hidden_size: int,
    head_size: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
) -> None:
    rotary_dim = int(hidden_size // (hidden_size / head_size) // 2)
    kwargs = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": max_position_embeddings,
        "base": base,
        "is_neox_style": is_neox_style,
    }
    native_rotary_data = RotaryData(cls=Llama4VisionRotaryEmbedding, dtype=torch.complex64, device="cpu")
    oot_rotary_data = RotaryData(cls=HPULlama4VisionRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    run_rotary_embedding_test(native_rotary_data, oot_rotary_data, seq_length, hidden_size, **kwargs)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("rotary_dim", ROTARY_DIMS)
@pytest.mark.parametrize("max_position_embeddings", MAX_POSITION_EMBEDDINGS)
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("is_neox_style", IS_NEOX_STYLE)
def test_m_rotary_embedding(
    default_vllm_config: None,
    seq_length: int,
    hidden_size: int,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
) -> None:
    kwargs = {
        "head_size": head_size,
        "rotary_dim": rotary_dim,
        "max_position_embeddings": max_position_embeddings,
        "base": base,
        "is_neox_style": is_neox_style,
        "mrope_section": [rotary_dim // 2]
    }
    native_rotary_data = RotaryData(cls=MRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    oot_rotary_data = RotaryData(cls=HPUMRotaryEmbedding, dtype=torch.bfloat16, device="hpu")
    run_rotary_embedding_test(native_rotary_data, oot_rotary_data, seq_length, hidden_size, **kwargs)
