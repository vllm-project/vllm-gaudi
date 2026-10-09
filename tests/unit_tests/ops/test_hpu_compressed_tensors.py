# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import habana_frameworks.torch as htorch
from types import SimpleNamespace
from utils import get_data_path, create_row_parallel_linear, create_fused_moe
from unittest.mock import MagicMock
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from vllm_gaudi.ops.hpu_compressed_tensors import (HPUCompressedTensorsLinearMethod, HPUCompressedTensorsW8A8Fp8,
                                                   HPUCompressedTensorsWNA16, HPUCompressedTensorsWNA16MoEMethod,
                                                   HPUCompressedTensorsW8A8Int8_BF16Fallback,
                                                   HPUCompressedTensorsW8A8Fp8MoEMethod)
from vllm_gaudi.utils import HPUCompileConfig
import vllm_gaudi.extension.ops as hpu_ext_ops
import vllm_gaudi.ops.hpu_compressed_tensors as hpu_ct
from vllm_gaudi.extension.ops import int4_moe_native_available, rebias_uint4b8_to_int4_
from vllm.model_executor.layers.quantization.utils.quant_utils import pack_quantized_values_into_int32
from vllm.scalar_type import scalar_types
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


def _wna16_moe_quant_config(actorder=None):
    """compressed-tensors config of a symmetric, group-128 w4a16 MoE checkpoint."""
    return CompressedTensorsConfig.from_config({
        'config_groups': {
            'group_0': {
                'input_activations': None,
                'output_activations': None,
                'targets': ['Linear'],
                'weights': {
                    'actorder': actorder,
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
    })


def _load_moe_wna16_reference_weights(experts):
    # Weights were extracted from first FusedMoE layer of RedHatAI/Qwen3-30B-A3B-quantized.w4a16
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/compressed_tensors/moe_wna16.safetensors"), framework="pt", device="hpu") as f:
        w2_weight_packed = f.get_tensor("w2_weight_packed")
        w2_weight_packed = torch.swapaxes(w2_weight_packed, 0, 1).repeat(128, 1, 1)
        experts.w2_weight_packed.copy_(w2_weight_packed)

        w13_weight_packed = f.get_tensor("w13_weight_packed")
        w13_weight_packed = torch.swapaxes(w13_weight_packed, 0, 1).repeat(128, 1, 1)
        experts.w13_weight_packed.copy_(w13_weight_packed)

        w2_weight_scale = f.get_tensor("w2_weight_scale")
        w2_weight_scale = torch.swapaxes(w2_weight_scale, 0, 1).repeat(128, 1, 1)
        experts.w2_weight_scale.copy_(w2_weight_scale)

        w13_weight_scale = f.get_tensor("w13_weight_scale")
        w13_weight_scale = torch.swapaxes(w13_weight_scale, 0, 1).repeat(128, 1, 1)
        experts.w13_weight_scale.copy_(w13_weight_scale)

        w2_weight_shape = torch.tensor([512, 256], dtype=torch.bfloat16, device="hpu")
        experts.w2_weight_shape.copy_(w2_weight_shape.repeat(128, 1))

        w13_weight_shape = torch.tensor([256, 512], dtype=torch.bfloat16, device="hpu")
        experts.w13_weight_shape.copy_(w13_weight_shape.repeat(128, 1))


def _load_moe_wna16_reference_io():
    # Output tensor holds data that was returned by cuda impl of CompressedTensorsWNA16MarlinMoEMethod for given input
    # (CompressedTensorsWNA16MarlinMoEMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/compressed_tensors/moe_wna16.safetensors"), framework="pt", device="hpu") as f:
        return f.get_tensor("hidden_states"), f.get_tensor("router_logits"), f.get_tensor("ref_output")


def _run_fused_moe(oot_op, hidden_states, router_logits):
    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())
    ctx = ForwardContext(
        no_compile_layers={oot_op.layer_name: oot_op},
        attn_metadata={},
        slot_mapping={},
    )
    with override_forward_context(ctx):
        return oot_op.forward(hidden_states, router_logits)


def _dequant_uint4b8(codes, scales, group_size):
    """[E, out, in] uint4b8 codes (value + 8) and [E, out, in / group] scales -> float32 weights."""
    return (codes.float() - 8) * scales.float().repeat_interleave(group_size, dim=-1)


def test_compressed_tensors_wna16_moe_method(default_vllm_config: None, dist_init):
    # Prepare FusedMoE layer with oot HPUCompressedTensorsWNA16MoEMethod
    oot_op = create_fused_moe(_wna16_moe_quant_config(actorder='weight')).to("hpu")
    assert isinstance(oot_op.routed_experts.quant_method, HPUCompressedTensorsWNA16MoEMethod)

    _load_moe_wna16_reference_weights(oot_op.routed_experts)
    oot_op.routed_experts.quant_method.process_weights_after_loading(oot_op.routed_experts)

    hidden_states, router_logits, ref_output = _load_moe_wna16_reference_io()
    out = _run_fused_moe(oot_op, hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)


def test_compressed_tensors_wna16_moe_method_native_int4(default_vllm_config: None, dist_init, monkeypatch):
    """The native int4 kernel reproduces the dequant-path reference, and the dequant path then refuses to run."""
    if not int4_moe_native_available():
        pytest.skip("this Habana PyTorch bridge has no mixture_of_experts.int4_fused_weights")
    monkeypatch.setattr(hpu_ct, "get_config", lambda: SimpleNamespace(wna16_native_int4_moe=True))

    oot_op = create_fused_moe(_wna16_moe_quant_config(actorder='weight')).to("hpu")
    experts = oot_op.routed_experts
    _load_moe_wna16_reference_weights(experts)
    experts.quant_method.process_weights_after_loading(experts)
    assert experts.moe_op.native_int4
    assert experts.moe_op.codes_signed

    hidden_states, router_logits, ref_output = _load_moe_wna16_reference_io()
    out = _run_fused_moe(oot_op, hidden_states, router_logits)
    # Same tolerance as the dequant path (measured max abs diff 3e-5).
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)

    # The codes are signed nibbles now: convert_from_uint4 would decode them wrongly.
    experts.moe_op.native_int4 = False
    topk_ids = torch.zeros(hidden_states.shape[0], 8, dtype=torch.int64, device="hpu")
    topk_weights = torch.full((hidden_states.shape[0], 8), 1 / 8, dtype=torch.bfloat16, device="hpu")
    with pytest.raises(RuntimeError, match="rebiased"):
        experts.moe_op(hidden_states, topk_ids, topk_weights, permuted_weights=False, activation="silu")


@pytest.mark.parametrize("native_int4", [False, True], ids=["dequant", "native_int4"])
def test_compressed_tensors_wna16_moe_method_cpu_first_migration(default_vllm_config: None, dist_init, monkeypatch,
                                                                 native_int4):
    """A CPU-first (or INC) load keeps a single device copy of the WNA16 expert weights.

    Replays the model runner's sequence: process the weights on the host,
    model.to("hpu"), _rebind_moe_expert_weights, then the
    _move_remaining_tensors_to_device sweep. Every per-expert view, zero point
    and _cached_int4 entry must alias the moved Parameters (one storage each)
    instead of being copied to the device separately, and the layer must still
    reproduce the reference output.
    """
    from vllm_gaudi.v1.worker.hpu_model_runner import (_model_has_moe_experts, _move_remaining_tensors_to_device,
                                                       _rebind_moe_expert_weights)
    if native_int4 and not int4_moe_native_available():
        pytest.skip("this Habana PyTorch bridge has no mixture_of_experts.int4_fused_weights")
    monkeypatch.setattr(hpu_ct, "get_config", lambda: SimpleNamespace(wna16_native_int4_moe=native_int4))

    def storage(t):
        try:
            return t.untyped_storage().data_ptr()
        except Exception:
            return t.data_ptr()

    oot_op = create_fused_moe(_wna16_moe_quant_config(actorder='weight'))  # stays on the host
    experts = oot_op.routed_experts
    _load_moe_wna16_reference_weights(experts)
    experts.quant_method.process_weights_after_loading(experts)
    assert experts.w13_weight_packed.device.type == "cpu"
    assert experts.moe_op.native_int4 is native_int4
    assert _model_has_moe_experts(oot_op)

    oot_op = oot_op.to("hpu")
    _rebind_moe_expert_weights(oot_op)
    _move_remaining_tensors_to_device(oot_op, "hpu")

    op = experts.moe_op
    for weights, packed, scale, zero_point in ((op.w13_list, experts.w13_weight_packed, experts.w13_weight_scale,
                                                experts.w13_zero_point),
                                               (op.w2_list, experts.w2_weight_packed, experts.w2_weight_scale,
                                                experts.w2_zero_point)):
        assert packed.device.type == "hpu"
        for m in weights:
            assert storage(m.weight_packed) == storage(packed)
            assert storage(m.weight_scale) == storage(scale)
            assert storage(m.zero_point) == storage(zero_point)
    if native_int4:
        w13_p, w2_p, w13_s, w2_s = op._cached_int4
        assert {storage(t) for t in w13_p} == {storage(experts.w13_weight_packed)}
        assert {storage(t) for t in w2_p} == {storage(experts.w2_weight_packed)}
        assert {storage(t) for t in w13_s} == {storage(experts.w13_weight_scale)}
        assert {storage(t) for t in w2_s} == {storage(experts.w2_weight_scale)}

    hidden_states, router_logits, ref_output = _load_moe_wna16_reference_io()
    out = _run_fused_moe(oot_op, hidden_states, router_logits)
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)


def test_wna16_rebias_uint4b8_to_int4():
    """uint4b8 codes (value + 8) become the two's-complement nibbles int4_fused_weights reads."""
    values = torch.arange(-8, 8, dtype=torch.int32).repeat(2, 1)  # every int4 value
    packed = pack_quantized_values_into_int32(values + 8, scalar_types.uint4b8, packed_dim=1)
    original = packed.clone()

    rebias_uint4b8_to_int4_(packed)
    nibbles = torch.stack([(packed >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(values.shape)
    torch.testing.assert_close(torch.where(nibbles >= 8, nibbles - 16, nibbles), values)

    rebias_uint4b8_to_int4_(packed)  # self-inverse
    torch.testing.assert_close(packed, original)


@pytest.mark.parametrize("reason", ["g_idx", "w8a16", "no_overload"])
def test_compressed_tensors_wna16_moe_method_native_int4_refusals(default_vllm_config: None, dist_init, monkeypatch,
                                                                  reason):
    """Checkpoints the int4 kernel cannot run fall back to the dequant path with their codes untouched."""
    monkeypatch.setattr(hpu_ct, "get_config", lambda: SimpleNamespace(wna16_native_int4_moe=True))
    if reason == "no_overload":
        monkeypatch.setattr(hpu_ext_ops, "int4_moe_native_available", lambda: False)

    oot_op = create_fused_moe(_wna16_moe_quant_config()).to("hpu")
    experts = oot_op.routed_experts
    if reason == "g_idx":
        # vLLM's config parser now rejects actorder="group" checkpoints, so set it on
        # the method to exercise the op's own guard (process attaches the g_idx).
        experts.quant_method.actorder = "group"
    elif reason == "w8a16":
        # Only the bit width matters for this decision; the 4-bit layer keeps the shapes simple.
        experts.quant_method.num_bits = 8
    for param in (experts.w13_weight_packed, experts.w2_weight_packed):
        param.data.copy_(torch.randint(-2**31, 2**31 - 1, param.shape, dtype=torch.int32))
    expected_w13 = experts.quant_method.gptq_hpu_moe_repack(experts.w13_weight_packed)

    experts.quant_method.process_weights_after_loading(experts)

    assert not experts.moe_op.native_int4
    assert not experts.moe_op.codes_signed
    torch.testing.assert_close(experts.w13_weight_packed.data, expected_w13)  # not rebiased


@pytest.mark.parametrize("native_int4", [False, True], ids=["dequant", "native_int4"])
def test_compressed_tensors_wna16_moe_method_non_gated_matches_dense_reference(default_vllm_config: None, dist_init,
                                                                               monkeypatch, native_int4):
    """Squared-ReLU (non-gated) experts on the real kernels against y = sum_k p_k * W2 relu(W1 x)^2.

    Both the dequant path and the native int4 kernel are gated-only, so they run
    the w1 -> w3 mirror; each must match the dense reference.
    """
    if native_int4 and not int4_moe_native_available():
        pytest.skip("this Habana PyTorch bridge has no mixture_of_experts.int4_fused_weights")
    monkeypatch.setattr(hpu_ct, "get_config", lambda: SimpleNamespace(wna16_native_int4_moe=native_int4))
    torch.manual_seed(0)
    num_experts, hidden, intermediate, group, top_k = 128, 512, 256, 128, 8  # create_fused_moe sizes

    oot_op = create_fused_moe(_wna16_moe_quant_config(), activation="relu2_no_mul").to("hpu")
    experts = oot_op.routed_experts
    up_codes = torch.randint(0, 16, (num_experts, intermediate, hidden), dtype=torch.int32)
    down_codes = torch.randint(0, 16, (num_experts, hidden, intermediate), dtype=torch.int32)
    up_scales = (torch.rand(num_experts, intermediate, hidden // group) * 0.01 + 0.005).to(torch.bfloat16)
    down_scales = (torch.rand(num_experts, hidden, intermediate // group) * 0.01 + 0.005).to(torch.bfloat16)
    # Checkpoint layout, packed along the input dim. For non-gated experts the
    # loader fills only the first I rows of w13.
    experts.w13_weight_packed.data[:, :intermediate].copy_(
        pack_quantized_values_into_int32(up_codes, scalar_types.uint4b8, packed_dim=2))
    experts.w13_weight_scale.data[:, :intermediate].copy_(up_scales)
    experts.w2_weight_packed.data.copy_(pack_quantized_values_into_int32(down_codes, scalar_types.uint4b8,
                                                                         packed_dim=2))
    experts.w2_weight_scale.data.copy_(down_scales)
    experts.quant_method.process_weights_after_loading(experts)
    assert experts.moe_op.native_int4 is native_int4

    hidden_states = torch.randn(4, hidden, dtype=torch.bfloat16)
    router_logits = torch.randn(4, num_experts, dtype=torch.bfloat16)
    out = _run_fused_moe(oot_op, hidden_states.to("hpu"), router_logits.to("hpu")).float().cpu()

    w1 = _dequant_uint4b8(up_codes, up_scales, group)
    w2 = _dequant_uint4b8(down_codes, down_scales, group)
    topk_p, topk_ids = torch.softmax(router_logits.float(), dim=-1).topk(top_k, dim=-1)
    topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
    x = hidden_states.float()
    ref = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for p, e in zip(topk_p[t], topk_ids[t]):
            ref[t] += p * (w2[e] @ torch.relu(w1[e] @ x[t]).square())
    rel_err = ((out - ref).norm() / ref.norm()).item()
    assert rel_err < 2e-2, f"relative error vs the dense reference: {rel_err:.4f}"


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
        # Load through the real weight_loader (not param.copy_()) so the full
        # served-model path is exercised. See issue #1612: the INT8 scheme must
        # not wrap this loader with the FP8-only gaudi_weight_wrapper.
        oot_op.weight.weight_loader(oot_op.weight, f.get_tensor("weight"))
        oot_op.weight_scale.weight_loader(oot_op.weight_scale, f.get_tensor("weight_scale"))
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    oot_op.quant_method.process_weights_after_loading(oot_op)

    sut_output = oot_op(input)

    torch.testing.assert_close(ref_output, sut_output.float(), atol=1e-3, rtol=1e-3)


def test_compressed_tensors_w8a8int8_weight_loader_does_not_corrupt_int8(default_vllm_config: None, dist_init):
    """Regression test for #1612.

    Load INT8 weights and their fp32 scales through the *real* `weight_loader`
    stored on each parameter (the path a served model actually takes), not via
    `param.copy_()`. On Gaudi2 the INT8 scheme used to wrap that loader with the
    FP8-only `gaudi_weight_wrapper`, which doubled every non-fp8 tensor (INT8
    weight overflowed +/-127, fp32 scale scaled by 2x) -> garbage output.

    Prior unit tests missed this because they copied straight into the params,
    bypassing the wrapper entirely.
    """
    config = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "dynamic": True,
                    "num_bits": 8,
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int"
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "dynamic": False,
                    "num_bits": 8,
                    "observer": "mse",
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int"
                }
            }
        },
        "format": "int-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }

    oot_quant_config = CompressedTensorsConfig.from_config(config)
    oot_op = create_row_parallel_linear(input_size=128, output_size=64, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Int8_BF16Fallback)

    # Deliberately include a near-int8-max value: if the loader doubles the
    # weight it overflows int8 and wraps around, which torch.int8.copy_ would
    # also clamp -> the corruption is unmistakable.
    src_weight = torch.zeros(64, 128, dtype=torch.int8)
    src_weight.fill_(3)
    src_weight[0, 0] = 100  # 100 * 2 = 200 overflows int8 (max 127)
    src_weight[1, 1] = -100
    src_weight = src_weight.to("hpu")
    src_scale = torch.full((64, 1), 0.02093, dtype=torch.float32, device="hpu")

    # Drive the real loader bound to each param.
    oot_op.weight.weight_loader(oot_op.weight, src_weight)
    oot_op.weight_scale.weight_loader(oot_op.weight_scale, src_scale)

    # The loader must not mutate INT8 weights or fp32 scales.
    torch.testing.assert_close(oot_op.weight.data.cpu(), src_weight.cpu())
    torch.testing.assert_close(oot_op.weight_scale.data.cpu(), src_scale.cpu())


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
    assert isinstance(oot_op.routed_experts.quant_method, HPUCompressedTensorsW8A8Fp8MoEMethod)

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

    oot_op.routed_experts.w13_weight.data.copy_(w13_weight)
    oot_op.routed_experts.w2_weight.data.copy_(w2_weight)
    oot_op.routed_experts.w13_weight_scale.data.copy_(w13_weight_scale)
    oot_op.routed_experts.w2_weight_scale.data.copy_(w2_weight_scale)

    oot_op.routed_experts.quant_method.process_weights_after_loading(oot_op.routed_experts)

    # Verify blockwise post-processing created the expected attributes
    assert hasattr(oot_op.routed_experts,
                   "w13_weight_scale_inv"), ("w13_weight_scale_inv should be created for block MoE")
    assert hasattr(oot_op.routed_experts, "w2_weight_scale_inv"), "w2_weight_scale_inv should be created for block MoE"
    assert not hasattr(oot_op.routed_experts, "w13_weight_scale"), "w13_weight_scale should be removed after aliasing"
    assert not hasattr(oot_op.routed_experts, "w2_weight_scale"), "w2_weight_scale should be removed after aliasing"

    # Execute layer with synthetic input
    hidden_states = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="hpu")
    router_logits = torch.randn(4, num_experts, dtype=torch.bfloat16, device="hpu")

    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op._forward_impl(hidden_states, router_logits, hidden_states)

    assert out.shape == hidden_states.shape
    assert out.dtype == torch.bfloat16
