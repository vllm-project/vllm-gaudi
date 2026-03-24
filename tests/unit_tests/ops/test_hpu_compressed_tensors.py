# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import habana_frameworks.torch as htorch
from utils import get_data_path, create_row_parallel_linear, create_fused_moe
from unittest.mock import MagicMock
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm_gaudi.ops.hpu_compressed_tensors import (HPUCompressedTensorsLinearMethod, HPUCompressedTensorsW8A8Fp8,
                                                   HPUCompressedTensorsW8A8Int8,
                                                   HPUCompressedTensorsWNA16, HPUCompressedTensorsWNA16MoEMethod)
from vllm_gaudi.utils import HPUCompileConfig
from vllm.forward_context import override_forward_context
from safetensors import safe_open


def test_compressed_tensors_linear_method_w8a8fp8_static_per_tensor(default_vllm_config: None, dist_init):
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

    # Prepare linear layer with oot CompressedTensorsLinearMethod
    # with HPUCompressedTensorsW8A8Fp8 scheme
    oot_op = create_row_parallel_linear(input_size=2048, output_size=8, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Fp8)

    # Weight and weight_scale_inv were extracted from first o_proj layer of Intel/Qwen3-0.6B-FP8-Test-Only
    # which is RowParallelLinear
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/compressed_tensors/linear_w8a8fp8_static_per_tensor.safetensors"),
                   framework="pt",
                   device="hpu") as f:
        oot_op.weight.copy_(f.get_tensor("weight"))
        oot_op.weight_scale.copy_(f.get_tensor("weight_scale"))
        oot_op.input_scale.copy_(f.get_tensor("input_scale"))

    oot_op.quant_method.process_weights_after_loading(oot_op)
    """
    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())
    """

    # Input and expected output
    # Output tensor holds data that was returned by cuda impl of CompressedTensorsLinearMethod for given input
    # (CompressedTensorsLinearMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/compressed_tensors/linear_w8a8fp8_static_per_tensor.safetensors"),
                   framework="pt",
                   device="hpu") as f:
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)


def test_compressed_tensors_linear_method_w8a8fp8_static_per_channel(default_vllm_config: None, dist_init):
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

    # Prepare linear layer with oot CompressedTensorsLinearMethod
    # with HPUCompressedTensorsW8A8Fp8 scheme
    oot_op = create_row_parallel_linear(input_size=2048, output_size=8, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Fp8)

    # Weight and weight_scale_inv were extracted from first o_proj layer of Intel/Qwen3-0.6B-FP8-Static-Test-Only
    # which is RowParallelLinear
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/compressed_tensors/linear_w8a8fp8_static_per_channel.safetensors"),
                   framework="pt",
                   device="hpu") as f:
        oot_op.weight.copy_(f.get_tensor("weight"))
        oot_op.weight_scale.copy_(f.get_tensor("weight_scale"))
        oot_op.input_scale.copy_(f.get_tensor("input_scale"))

    oot_op.quant_method.process_weights_after_loading(oot_op)
    """
    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())
    """

    # Input and expected output
    # Output tensor holds data that was returned by cuda impl of CompressedTensorsLinearMethod for given input
    # (CompressedTensorsLinearMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/compressed_tensors/linear_w8a8fp8_static_per_channel.safetensors"),
                   framework="pt",
                   device="hpu") as f:
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)


def test_compressed_tensors_linear_method_w8a8fp8(default_vllm_config: None, dist_init):
    config = {
        'config_groups': {
            'group_0': {
                'input_activations': {
                    'block_structure': None,
                    'dynamic': True,
                    'group_size': None,
                    'num_bits': 8,
                    'observer': 'memoryless',
                    'observer_kwargs': {},
                    'strategy': 'token',
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
        'format': 'naive-quantized',
        'global_compression_ratio': 1.239290831149584,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'frozen'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    # Prepare linear layer with oot CompressedTensorsLinearMethod
    # with HPUCompressedTensorsW8A8Fp8 scheme
    oot_op = create_row_parallel_linear(input_size=256, output_size=256, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Fp8)

    # Weight and weight_scale_inv were extracted from first RowParallelLinear
    # layer of RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/compressed_tensors/linear_w8a8fp8.safetensors"), framework="pt",
                   device="hpu") as f:
        oot_op.weight.copy_(f.get_tensor("weight"))
        oot_op.weight_scale.copy_(f.get_tensor("weight_scale"))
    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds data that was returned by cuda impl of CompressedTensorsLinearMethod for given input
    # (CompressedTensorsLinearMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/compressed_tensors/linear_w8a8fp8.safetensors"), framework="pt",
                   device="hpu") as f:
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)


def test_compressed_tensors_linear_method_wna16(default_vllm_config: None, dist_init):
    config = {
        'config_groups': {
            'group_0': {
                'input_activations': None,
                'output_activations': None,
                'targets': ['Linear'],
                'weights': {
                    'actorder': 'weight',
                    'block_structure': None,
                    'dynamic': False,
                    'group_size': 128,
                    'num_bits': 4,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'group',
                    'symmetric': False,
                    'type': 'int'
                }
            }
        },
        'format': 'pack-quantized',
        'global_compression_ratio': None,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    # Prepare linear layer with oot CompressedTensorsLinearMethod
    # with HPUCompressedTensorsWNA16 scheme
    oot_op = create_row_parallel_linear(input_size=256, output_size=256, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsWNA16)

    # Weights were extracted from first RowParallelLinear layer of RedHatAI/Qwen3-8B-quantized.w4a16
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/compressed_tensors/linear_wna16.safetensors"), framework="pt",
                   device="hpu") as f:
        oot_op.weight_packed.copy_(f.get_tensor("weight_packed"))
        oot_op.weight_scale.copy_(f.get_tensor("weight_scale"))
        oot_op.weight_zero_point.copy_(f.get_tensor("weight_zero_point"))
        oot_op.weight_shape.data = torch.tensor([256, 256], device='hpu:0')
    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds data that was returned by cuda impl of CompressedTensorsLinearMethod for given input
    # (CompressedTensorsLinearMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/compressed_tensors/linear_wna16.safetensors"), framework="pt",
                   device="hpu") as f:
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)


def test_compressed_tensors_wna16_moe_method(default_vllm_config: None, dist_init):
    config = {
        'config_groups': {
            'group_0': {
                'input_activations': None,
                'output_activations': None,
                'targets': ['Linear'],
                'weights': {
                    'actorder': 'weight',
                    'block_structure': None,
                    'dynamic': False,
                    'group_size': 128,
                    'num_bits': 4,
                    'observer': 'minmax',
                    'observer_kwargs': {},
                    'strategy': 'group',
                    'symmetric': True,
                    'type': 'int'
                }
            }
        },
        'format': 'pack-quantized',
        'global_compression_ratio': None,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    # Prepare FusedMoE layer with oot HPUCompressedTensorsWNA16MoEMethod
    oot_op = create_fused_moe(oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsWNA16MoEMethod)

    # Weights were extracted from first FusedMoE layer of RedHatAI/Qwen3-30B-A3B-quantized.w4a16
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/compressed_tensors/moe_wna16.safetensors"), framework="pt", device="hpu") as f:
        w2_weight_packed = f.get_tensor("w2_weight_packed")
        w2_weight_packed = torch.swapaxes(w2_weight_packed, 0, 1).repeat(128, 1, 1)
        oot_op.w2_weight_packed.copy_(w2_weight_packed)

        w13_weight_packed = f.get_tensor("w13_weight_packed")
        w13_weight_packed = torch.swapaxes(w13_weight_packed, 0, 1).repeat(128, 1, 1)
        oot_op.w13_weight_packed.copy_(w13_weight_packed)

        w2_weight_scale = f.get_tensor("w2_weight_scale")
        w2_weight_scale = torch.swapaxes(w2_weight_scale, 0, 1).repeat(128, 1, 1)
        oot_op.w2_weight_scale.copy_(w2_weight_scale)

        w13_weight_scale = f.get_tensor("w13_weight_scale")
        w13_weight_scale = torch.swapaxes(w13_weight_scale, 0, 1).repeat(128, 1, 1)
        oot_op.w13_weight_scale.copy_(w13_weight_scale)

        w2_weight_shape = torch.tensor([512, 256], dtype=torch.bfloat16, device="hpu")
        oot_op.w2_weight_shape.copy_(w2_weight_shape.repeat(128, 1))

        w13_weight_shape = torch.tensor([256, 512], dtype=torch.bfloat16, device="hpu")
        oot_op.w13_weight_shape.copy_(w13_weight_shape.repeat(128, 1))

    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds data that was returned by cuda impl of CompressedTensorsWNA16MarlinMoEMethod for given input
    # (CompressedTensorsWNA16MarlinMoEMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/compressed_tensors/moe_wna16.safetensors"), framework="pt", device="hpu") as f:
        hidden_states = f.get_tensor("hidden_states")
        router_logits = f.get_tensor("router_logits")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op.runner.forward_impl(oot_op, hidden_states, router_logits, hidden_states)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)


def test_compressed_tensors_linear_method_w8a8_int8_per_channel(default_vllm_config: None, dist_init):
    """W8A8 INT8: weight per-channel, activation per-tensor, symmetric."""
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
                    'type': 'int'
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
                    'type': 'int'
                }
            }
        },
        'format': 'int-quantized',
        'global_compression_ratio': None,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    # Create the linear layer
    oot_op = create_row_parallel_linear(input_size=256, output_size=256, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Int8)

    # Set up synthetic INT8 weights and scales
    torch.manual_seed(42)
    int8_weight = torch.randint(-128, 127, (256, 256), dtype=torch.int8, device='hpu')
    weight_scale = torch.rand(256, 1, dtype=torch.float32, device='hpu') + 0.01
    input_scale = torch.tensor([0.05], dtype=torch.float32, device='hpu')

    oot_op.weight.copy_(int8_weight)
    oot_op.weight_scale.copy_(weight_scale)
    oot_op.input_scale.copy_(input_scale)

    oot_op.quant_method.process_weights_after_loading(oot_op)

    # Run inference
    input_tensor = torch.randn(1, 4, 256, dtype=torch.bfloat16, device='hpu')
    out = oot_op(input_tensor)

    # Basic shape check
    assert out.shape == (1, 4, 256), f"Expected shape (1, 4, 256), got {out.shape}"
    # Verify output is finite
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_compressed_tensors_linear_method_w8a8_int8_per_tensor(default_vllm_config: None, dist_init):
    """W8A8 INT8: weight per-tensor, activation per-tensor, symmetric."""
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
                    'type': 'int'
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
                    'type': 'int'
                }
            }
        },
        'format': 'int-quantized',
        'global_compression_ratio': None,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    # Create the linear layer
    oot_op = create_row_parallel_linear(input_size=256, output_size=256, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Int8)

    # Set up synthetic INT8 weights and scales
    torch.manual_seed(42)
    int8_weight = torch.randint(-128, 127, (256, 256), dtype=torch.int8, device='hpu')
    weight_scale = torch.tensor([0.05], dtype=torch.float32, device='hpu')
    input_scale = torch.tensor([0.05], dtype=torch.float32, device='hpu')

    oot_op.weight.copy_(int8_weight)
    oot_op.weight_scale.copy_(weight_scale)
    oot_op.input_scale.copy_(input_scale)

    oot_op.quant_method.process_weights_after_loading(oot_op)

    # Run inference
    input_tensor = torch.randn(1, 4, 256, dtype=torch.bfloat16, device='hpu')
    out = oot_op(input_tensor)

    # Basic shape check
    assert out.shape == (1, 4, 256), f"Expected shape (1, 4, 256), got {out.shape}"
    # Verify output is finite
    assert torch.isfinite(out).all(), "Output contains non-finite values"
