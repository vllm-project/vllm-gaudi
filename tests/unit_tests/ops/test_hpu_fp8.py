# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import habana_frameworks.torch as htorch
from utils import get_data_path, create_row_parallel_linear, create_fused_moe
from unittest.mock import MagicMock
from vllm_gaudi.ops.hpu_fp8 import Fp8LinearMethod, HPUFp8MoEMethod
from vllm_gaudi.utils import HPUCompileConfig
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from safetensors import safe_open


def test_fp8_linear_method(dist_init, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", "0")
    config = {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]}
    oot_quant_config = Fp8Config.from_config(config)

    # Prepare linear layer with oot Fp8LinearMethod
    oot_op = create_row_parallel_linear(input_size=256, output_size=256, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, Fp8LinearMethod)

    # Weight and weight_scale_inv were extracted from first RowParallelLinear layer of Qwen/Qwen3-8B-FP8
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/fp8/linear.safetensors"), framework="pt", device="hpu") as f:
        oot_op.weight.copy_(f.get_tensor("weight"))
        oot_op.weight_scale_inv.copy_(f.get_tensor("weight_scale_inv"))
    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        # Setting fullgraph to False, because currently there is a graph break
        compile_config = HPUCompileConfig(fullgraph=False)
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of Fp8LinearMethod for given input
    # (Fp8LinearMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/fp8/linear.safetensors"), framework="pt", device="hpu") as f:
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)


def test_fp8_moe_method(dist_init, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", "0")
    config = {
        'activation_scheme': 'dynamic',
        'modules_to_not_convert': [],
        'fmt': 'e4m3',
        'quant_method': 'fp8',
        'weight_block_size': [128, 128]
    }
    oot_quant_config = Fp8Config.from_config(config)

    # Prepare FusedMoE layer with oot HPUFp8MoEMethod
    oot_op = create_fused_moe(oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, HPUFp8MoEMethod)

    # Weights were extracted from first FusedMoE layer of Qwen/Qwen3-30B-A3B-FP8
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/fp8/moe.safetensors"), framework="pt", device="hpu") as f:
        w13_weight = f.get_tensor("w13_weight")
        oot_op.w13_weight.copy_(w13_weight.repeat(128, 1, 1))

        w13_weight_scale_inv = f.get_tensor("w13_weight_scale_inv")
        oot_op.w13_weight_scale_inv.copy_(w13_weight_scale_inv.repeat(128, 1, 1))

        w2_weight = f.get_tensor("w2_weight")
        oot_op.w2_weight.copy_(w2_weight.repeat(128, 1, 1))

        w2_weight_scale_inv = f.get_tensor("w2_weight_scale_inv")
        oot_op.w2_weight_scale_inv.copy_(w2_weight_scale_inv.repeat(128, 1, 1))

    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of Fp8MoEMethod for given input
    # (Fp8MoEMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/fp8/moe.safetensors"), framework="pt", device="hpu") as f:
        hidden_states = f.get_tensor("hidden_states")
        router_logits = f.get_tensor("router_logits")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op.forward_impl(hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)
