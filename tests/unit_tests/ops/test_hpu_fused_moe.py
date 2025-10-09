# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import habana_frameworks.torch as htorch
from utils import get_data_path
from unittest.mock import MagicMock
from vllm_gaudi.ops.hpu_fused_moe import HPUUnquantizedFusedMoEMethod
from vllm_gaudi.utils import HPUCompileConfig
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


def test_unquantized_fused_moe_method(dist_init):
    # Prepare FusedMoE layer with oot HPUUnquantizedFusedMoEMethod
    oot_op = FusedMoE(num_experts=128,
                      top_k=8,
                      hidden_size=512,
                      intermediate_size=256,
                      params_dtype=torch.bfloat16,
                      reduce_results=True,
                      renormalize=True,
                      use_grouped_topk=False,
                      num_expert_group=None,
                      topk_group=None,
                      quant_config=None,
                      tp_size=None,
                      ep_size=None,
                      dp_size=None,
                      custom_routing_function=None,
                      scoring_func="softmax",
                      routed_scaling_factor=1.0,
                      e_score_correction_bias=None,
                      apply_router_weight_on_input=False,
                      activation="silu",
                      enable_eplb=False,
                      num_redundant_experts=0,
                      has_bias=False,
                      is_sequence_parallel=False,
                      zero_expert_num=0,
                      zero_expert_type=None).to("hpu")
    assert isinstance(oot_op.quant_method, HPUUnquantizedFusedMoEMethod)

    # Weights were extracted from first FusedMoE layer of Qwen/Qwen3-30B-A3
    # (with adjusted shapes, to make tensors smaller)
    w2_weight = torch.load(get_data_path("data/fused_moe/w2_weight.pt"), weights_only=False, map_location="hpu")
    oot_op.w2_weight.copy_(w2_weight.repeat(128, 1, 1))
    w13_weight = torch.load(get_data_path("data/fused_moe/w13_weight.pt"), weights_only=False, map_location="hpu")
    oot_op.w13_weight.copy_(w13_weight.repeat(128, 1, 1))

    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds data that was returned by cuda impl of UnquantizedFusedMoEMethod for given input
    # (UnquantizedFusedMoEMethod was triggered offline with the same input as below to get the ref_output)
    hidden_states = torch.load(get_data_path("data/fused_moe/input_hidden_states.pt"),
                               weights_only=False,
                               map_location="hpu")
    router_logits = torch.load(get_data_path("data/fused_moe/input_router_logits.pt"),
                               weights_only=False,
                               map_location="hpu")
    ref_output = torch.load(get_data_path("data/fused_moe/output.pt"), weights_only=False, map_location="hpu")

    # Execute layer
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op.forward_impl(hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)
