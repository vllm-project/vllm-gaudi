# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import habana_frameworks.torch as htorch
from utils import get_data_path, create_row_parallel_linear, create_fused_moe
from unittest.mock import MagicMock
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm_gaudi.ops.hpu_compressed_tensors import (HPUCompressedTensorsLinearMethod, HPUCompressedTensorsW8A8Fp8,
                                                   HPUCompressedTensorsWNA16, HPUCompressedTensorsWNA16MoEMethod,
                                                   HPUCompressedTensorsW8A8Int8_BF16Fallback,
                                                   HPUCompressedTensorsW8A8Fp8MoEMethod)
from vllm_gaudi.utils import HPUCompileConfig
from vllm.forward_context import ForwardContext, override_forward_context
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
    ctx = ForwardContext(
        no_compile_layers={oot_op.runner.layer_name: oot_op},
        attn_metadata={},
        slot_mapping={},
    )
    with override_forward_context(ctx):
        out = oot_op.runner.forward(hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)


def test_compressed_tensors_linear_method_w8a8int8_bf16fallback_static_per_channel(default_vllm_config: None,
                                                                                   dist_init):

    config = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": None,
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int"
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "mse",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int"
                }
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": 1.5302466391371097,
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }

    oot_quant_config = CompressedTensorsConfig.from_config(config)
    oot_op = create_row_parallel_linear(input_size=128, output_size=64, quant_config=oot_quant_config).to("hpu")

    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Int8_BF16Fallback)
    """
    The fixture was made by sampling BF16 random weight/input, per-row quantizing weight to INT8
    with a per-output-channel weight_scale, then computing ref_output = input @ dequant(weight, weight_scale)^T
    (saved alongside all tensors in a .safetensors).
    """
    with safe_open(get_data_path("data/compressed_tensors/linear_w8a8int8_bf16fallback_static_per_channel.safetensors"),
                   framework="pt",
                   device="hpu") as f:
        oot_op.weight.copy_(f.get_tensor("weight"))
        oot_op.weight_scale.copy_(f.get_tensor("weight_scale"))
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    oot_op.quant_method.process_weights_after_loading(oot_op)

    sut_output = oot_op(input)

    torch.testing.assert_close(ref_output, sut_output.float(), atol=1e-3, rtol=1e-3)


def test_compressed_tensors_linear_method_w8a8fp8_block(default_vllm_config: None, dist_init):
    """weight per-block, activation dynamic per-group
    Config based on mistralai/Mistral-Large-3-675B-Instruct-2512 params.json
    """
    block_structure = [128, 128]
    config = {
        'config_groups': {
            'FP8_BLOCK': {
                'format': 'float-quantized',
                'input_activations': {
                    'actorder': None,
                    'block_structure': None,
                    'dynamic': True,
                    'group_size': 128,
                    'num_bits': 8,
                    'observer': None,
                    'observer_kwargs': {},
                    'strategy': 'group',
                    'symmetric': True,
                    'type': 'float'
                },
                'output_activations': None,
                'targets': ['Linear'],
                'weights': {
                    'actorder': None,
                    'block_structure': block_structure,
                    'dynamic': False,
                    'group_size': None,
                    'num_bits': 8,
                    'observer': 'static_minmax',
                    'observer_kwargs': {},
                    'strategy': 'block',
                    'symmetric': True,
                    'type': 'float'
                }
            }
        },
        'format': 'float-quantized',
        'global_compression_ratio': None,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)
    input_size = 256
    output_size = 256
    block_n, block_k = block_structure

    oot_op = create_row_parallel_linear(input_size=input_size, output_size=output_size,
                                        quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Fp8)

    # Create synthetic FP8 block-quantized weights
    weight_fp32 = torch.randn(output_size, input_size, dtype=torch.bfloat16, device="hpu")
    weight_fp8 = weight_fp32.to(torch.float8_e4m3fn)
    scale_rows = (output_size + block_n - 1) // block_n
    scale_cols = (input_size + block_k - 1) // block_k
    weight_scale = torch.ones(scale_rows, scale_cols, dtype=torch.float32, device="hpu")
    oot_op.weight.data.copy_(weight_fp8)
    oot_op.weight_scale.data.copy_(weight_scale)

    oot_op.quant_method.process_weights_after_loading(oot_op)

    # Verify blockwise post-processing created the expected attributes
    assert hasattr(oot_op, "weight_scale_inv"), "weight_scale_inv should be created for block strategy"
    assert not hasattr(oot_op, "weight_scale"), "weight_scale should be removed after aliasing"

    # Execute layer with synthetic input
    x = torch.randn(1, 4, input_size, dtype=torch.bfloat16, device="hpu")
    out = oot_op.scheme.apply_weights(oot_op, x)
    assert out.shape == (1, 4, output_size)
    assert out.dtype == torch.bfloat16


def test_compressed_tensors_w8a8fp8_block_moe_method(default_vllm_config: None, dist_init):
    """FP8 block-quantized MoE: weight per-block, activation dynamic per-group
    Config based on mistralai/Mistral-Large-3-675B-Instruct-2512 params.json
    """
    block_structure = [128, 128]
    config = {
        'config_groups': {
            'FP8_BLOCK': {
                'format': 'float-quantized',
                'input_activations': {
                    'actorder': None,
                    'block_structure': None,
                    'dynamic': True,
                    'group_size': 128,
                    'num_bits': 8,
                    'observer': None,
                    'observer_kwargs': {},
                    'strategy': 'group',
                    'symmetric': True,
                    'type': 'float'
                },
                'output_activations': None,
                'targets': ['Linear'],
                'weights': {
                    'actorder': None,
                    'block_structure': block_structure,
                    'dynamic': False,
                    'group_size': None,
                    'num_bits': 8,
                    'observer': 'static_minmax',
                    'observer_kwargs': {},
                    'strategy': 'block',
                    'symmetric': True,
                    'type': 'float'
                }
            }
        },
        'format': 'float-quantized',
        'global_compression_ratio': None,
        'ignore': [],
        'kv_cache_scheme': None,
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed'
    }
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    oot_op = create_fused_moe(oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsW8A8Fp8MoEMethod)

    num_experts = 128
    hidden_size = 512
    intermediate_size = 256
    block_n, block_k = block_structure

    # Create synthetic FP8 block-quantized MoE weights
    w13_weight = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=torch.bfloat16,
                             device="hpu").to(torch.float8_e4m3fn)
    w2_weight = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16,
                            device="hpu").to(torch.float8_e4m3fn)

    w13_scale_rows = (2 * intermediate_size + block_n - 1) // block_n
    w13_scale_cols = (hidden_size + block_k - 1) // block_k
    w2_scale_rows = (hidden_size + block_n - 1) // block_n
    w2_scale_cols = (intermediate_size + block_k - 1) // block_k

    w13_weight_scale = torch.ones(num_experts, w13_scale_rows, w13_scale_cols, dtype=torch.float32, device="hpu")
    w2_weight_scale = torch.ones(num_experts, w2_scale_rows, w2_scale_cols, dtype=torch.float32, device="hpu")

    oot_op.w13_weight.data.copy_(w13_weight)
    oot_op.w2_weight.data.copy_(w2_weight)
    oot_op.w13_weight_scale.data.copy_(w13_weight_scale)
    oot_op.w2_weight_scale.data.copy_(w2_weight_scale)

    oot_op.quant_method.process_weights_after_loading(oot_op)

    # Verify blockwise post-processing created the expected attributes
    assert hasattr(oot_op, "w13_weight_scale_inv"), "w13_weight_scale_inv should be created for block MoE"
    assert hasattr(oot_op, "w2_weight_scale_inv"), "w2_weight_scale_inv should be created for block MoE"
    assert not hasattr(oot_op, "w13_weight_scale"), "w13_weight_scale should be removed after aliasing"
    assert not hasattr(oot_op, "w2_weight_scale"), "w2_weight_scale should be removed after aliasing"

    # Execute layer with synthetic input
    hidden_states = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="hpu")
    router_logits = torch.randn(4, num_experts, dtype=torch.bfloat16, device="hpu")

    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op.runner._forward_impl(oot_op, hidden_states, router_logits, hidden_states)

    assert out.shape == hidden_states.shape
    assert out.dtype == torch.bfloat16
