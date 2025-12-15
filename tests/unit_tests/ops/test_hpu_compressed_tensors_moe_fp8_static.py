# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import habana_frameworks.torch as htorch
from utils import get_data_path, create_fused_moe
from unittest.mock import MagicMock
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm_gaudi.ops.hpu_compressed_tensors import (
    HPUCompressedTensorsW8A8Fp8MoEMethod
)
from vllm_gaudi.utils import HPUCompileConfig
from vllm.forward_context import override_forward_context
from safetensors import safe_open


def test_compressed_tensors_moe_method_w8a8fp8_static_per_channel(dist_init):
    """weight per-channel, activation per-tensor
    """
    config = {
        'config_groups': {
            'group_0': {
                'input_activations': {
                    'block_structure': None,
                    'dynamic': False,
                    'group_size': None,
                    'num_bits': 8,
                    'observer': 'memoryless',
                    'observer_kwargs': {},
                    'strategy': 'tensor',
                    'symmetric': True,
                    'type': 'float'
                },
                'output_activations': None,
                'targets': ['Linear'],
                'weights': {
                    'block_structure': None,
                    'dynamic': False,
                    'group_size': None,
                    'num_bits': 8,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'channel',
                    'symmetric': True,
                    'type': 'float'
                }
            }
        },
        'format': 'float-quantized',
        'global_compression_ratio': 1.239290831149584,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    # Prepare FusedMoE layer with oot HPUCompressedTensorsW8A8Fp8MoEMethod
    oot_op = create_fused_moe(oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsW8A8Fp8MoEMethod)

    # Weights were extracted from first FusedMoE layer of RedHatAI/Qwen3-30B-A3B-quantized.w4a16
    # (with adjusted shapes, to make tensors smaller)

    with safe_open(get_data_path("data/compressed_tensors/moe_w8a8fp8_static_per_channel.safetensors"), framework="pt") as f:
        w2_weight = f.get_tensor("w2_weight").to("hpu").repeat(128, 1, 1)
        oot_op.w2_weight.copy_(w2_weight)

        w13_weight = f.get_tensor("w13_weight").to("hpu").repeat(128, 1, 1)
        oot_op.w13_weight.copy_(w13_weight)

        w2_weight_scale = f.get_tensor("w2_weight_scale").to("hpu").repeat(128, 1, 1)
        oot_op.w2_weight_scale.copy_(w2_weight_scale)

        w13_weight_scale = f.get_tensor("w13_weight_scale").to("hpu").repeat(128, 1, 1)
        oot_op.w13_weight_scale.copy_(w13_weight_scale)

        w13_input_scale = f.get_tensor("w13_input_scale").to("hpu").repeat(128)
        oot_op.w13_input_scale.copy_(w13_input_scale)

        w2_input_scale = f.get_tensor("w2_input_scale").to("hpu").repeat(128)
        oot_op.w2_input_scale.copy_(w2_input_scale)


    oot_op.quant_method.process_weights_after_loading(oot_op)

    # Input and expected output
    with safe_open(get_data_path("data/compressed_tensors/moe_w8a8fp8_static_per_channel.safetensors"), framework="pt") as f:
        hidden_states = f.get_tensor("hidden_states").to("hpu")
        router_logits = f.get_tensor("router_logits").to("hpu").repeat(1, 128)
        ref_output = f.get_tensor("ref_output").to("hpu")

    # Execute layer
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op.forward_impl(hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)


def test_compressed_tensors_moe_method_w8a8fp8_static_per_tensor(dist_init):
    """weight per-tensor, activation per-tensor
    """
    config = {
        'config_groups': {
            'group_0': {
                'input_activations': {
                    'block_structure': None,
                    'dynamic': False,
                    'group_size': None,
                    'num_bits': 8,
                    'observer': 'memoryless',
                    'observer_kwargs': {},
                    'strategy': 'tensor',
                    'symmetric': True,
                    'type': 'float'
                },
                'output_activations': None,
                'targets': ['Linear'],
                'weights': {
                    'block_structure': None,
                    'dynamic': False,
                    'group_size': None,
                    'num_bits': 8,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'tensor',
                    'symmetric': True,
                    'type': 'float'
                }
            }
        },
        'format': 'float-quantized',
        'global_compression_ratio': 1.239290831149584,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    # Prepare FusedMoE layer with oot HPUCompressedTensorsW8A8Fp8MoEMethod
    oot_op = create_fused_moe(oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsW8A8Fp8MoEMethod)

    # Weights were extracted from first FusedMoE layer of RedHatAI/Qwen3-30B-A3B-quantized.w4a16
    # (with adjusted shapes, to make tensors smaller)

    with safe_open(get_data_path("data/compressed_tensors/moe_w8a8fp8_static_per_tensor.safetensors"), framework="pt") as f:
        w2_weight = f.get_tensor("w2_weight").to("hpu").repeat(128, 1, 1)
        oot_op.w2_weight.copy_(w2_weight)

        w13_weight = f.get_tensor("w13_weight").to("hpu").repeat(128, 1, 1)
        oot_op.w13_weight.copy_(w13_weight)

        w2_weight_scale = f.get_tensor("w2_weight_scale").to("hpu").repeat(128)
        oot_op.w2_weight_scale.copy_(w2_weight_scale)

        w13_weight_scale = f.get_tensor("w13_weight_scale").to("hpu").repeat(128, 2)
        oot_op.w13_weight_scale.copy_(w13_weight_scale)

        w13_input_scale = f.get_tensor("w13_input_scale").to("hpu").repeat(128)
        oot_op.w13_input_scale.copy_(w13_input_scale)

        w2_input_scale = f.get_tensor("w2_input_scale").to("hpu").repeat(128)
        oot_op.w2_input_scale.copy_(w2_input_scale)

    oot_op.quant_method.process_weights_after_loading(oot_op)

    # Input and expected output
    with safe_open(get_data_path("data/compressed_tensors/moe_w8a8fp8_static_per_tensor.safetensors"), framework="pt") as f:
        hidden_states = f.get_tensor("hidden_states").to("hpu")
        router_logits = f.get_tensor("router_logits").to("hpu").repeat(1, 128)
        ref_output = f.get_tensor("ref_output").to("hpu")

    # Execute layer
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op.forward_impl(hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)
