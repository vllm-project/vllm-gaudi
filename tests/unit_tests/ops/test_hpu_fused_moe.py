# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import habana_frameworks.torch as htorch
from utils import get_data_path, create_fused_moe
import vllm_gaudi.ops.hpu_fused_moe as hpu_fused_moe
from vllm_gaudi.ops.hpu_fused_moe import (
    HPUUnquantizedFusedMoEMethod,
    _gather_swigluoai_moe,
    _unfused_swigluoai_moe,
)
from vllm_gaudi.utils import HPUCompileConfig
from vllm.forward_context import ForwardContext, override_forward_context
from safetensors import safe_open


def test_unquantized_fused_moe_method(default_vllm_config: None, dist_init):
    # Prepare FusedMoE layer with oot HPUUnquantizedFusedMoEMethod
    oot_op = create_fused_moe().to("hpu")
    assert isinstance(oot_op.routed_experts.quant_method, HPUUnquantizedFusedMoEMethod)

    # Weights were extracted from first FusedMoE layer of Qwen/Qwen3-30B-A3
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/fused_moe/unquantized.safetensors"), framework="pt", device="hpu") as f:
        w2_weight = f.get_tensor("w2_weight")
        oot_op.routed_experts.w2_weight.copy_(w2_weight.repeat(128, 1, 1))
        w13_weight = f.get_tensor("w13_weight")
        oot_op.routed_experts.w13_weight.copy_(w13_weight.repeat(128, 1, 1))
    oot_op.routed_experts.quant_method.process_weights_after_loading(oot_op.routed_experts)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds data that was returned by cuda impl of UnquantizedFusedMoEMethod for given input
    # (UnquantizedFusedMoEMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/fused_moe/unquantized.safetensors"), framework="pt", device="hpu") as f:
        hidden_states = f.get_tensor("hidden_states")
        router_logits = f.get_tensor("router_logits")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    ctx = ForwardContext(
        no_compile_layers={oot_op.layer_name: oot_op},
        attn_metadata={},
        slot_mapping={},
    )
    with override_forward_context(ctx):
        out = oot_op.forward(hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("tokens", "ep_rank"), [(1, 0), (2, 1), (8, 0)])
def test_gather_swigluoai_moe_matches_dense(tokens: int, ep_rank: int):
    num_experts = 64
    top_k = 8
    hidden_size = 16
    intermediate_size = 12
    experts_min = ep_rank * num_experts

    layer = SimpleNamespace(
        w13_weight=torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device="hpu",
        ),
        w2_weight=torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device="hpu",
        ),
        moe_config=SimpleNamespace(ep_rank=ep_rank),
        local_num_experts=num_experts,
    )
    x = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="hpu")
    local_ids = (torch.arange(tokens * top_k, device="hpu").view(tokens, top_k) + 3) % num_experts
    topk_ids = local_ids + experts_min
    if ep_rank > 0:
        topk_ids[:, 0] = 0  # Expert owned by another EP rank.
    topk_weights = torch.softmax(
        torch.randn(tokens, top_k, dtype=torch.float32, device="hpu"),
        dim=-1,
    ).to(torch.bfloat16)

    dense = _unfused_swigluoai_moe(
        layer,
        x,
        topk_ids,
        topk_weights,
        alpha=1.702,
        beta=1.0,
        limit=7.0,
    )
    gathered = _gather_swigluoai_moe(
        layer,
        x,
        topk_ids,
        topk_weights,
        alpha=1.702,
        beta=1.0,
        limit=7.0,
    )

    torch.testing.assert_close(gathered, dense, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("enabled", "tokens", "top_k", "expected_path"),
    [
        (True, 1, 8, "gather"),
        (True, 16, 2, "gather"),
        (True, 17, 2, "dense"),
        (True, 16, 4, "dense"),
        (False, 1, 8, "dense"),
    ],
)
def test_swigluoai_moe_dispatch(monkeypatch, enabled: bool, tokens: int, top_k: int, expected_path: str):
    calls: list[str] = []

    def gather(*args, **kwargs):
        calls.append("gather")
        return args[1]

    def dense(*args, **kwargs):
        calls.append("dense")
        return args[1]

    monkeypatch.setattr(hpu_fused_moe, "_MOE_DECODE_GATHER", enabled)
    monkeypatch.setattr(hpu_fused_moe, "_MOE_GATHER_MAX_TOKENS", 16)
    monkeypatch.setattr(hpu_fused_moe, "_gather_swigluoai_moe", gather)
    monkeypatch.setattr(hpu_fused_moe, "_unfused_swigluoai_moe", dense)

    layer = SimpleNamespace(w13_weight=torch.empty(64, 1, 1))
    x = torch.empty(tokens, 1)
    topk_ids = torch.empty(tokens, top_k, dtype=torch.int64)
    topk_weights = torch.empty(tokens, top_k)

    result = hpu_fused_moe._swigluoai_moe(
        layer,
        x,
        topk_ids,
        topk_weights,
        alpha=1.702,
        beta=1.0,
        limit=7.0,
    )

    assert result is x
    assert calls == [expected_path]


def _routing_layer(num_experts: int = 16, top_k: int = 4, **overrides):
    attrs = dict(
        top_k=top_k,
        use_grouped_topk=False,
        custom_routing_function=None,
        scoring_func="softmax",
        e_score_correction_bias=None,
        renormalize=True,
        routed_scaling_factor=1.0,
        topk_group=None,
        num_expert_group=None,
        moe_config=SimpleNamespace(num_logical_experts=num_experts),
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (dict(), True),
        (dict(scoring_func="sigmoid"), False),
        (dict(e_score_correction_bias=torch.zeros(16)), False),
        (dict(renormalize=False), False),
        (dict(use_grouped_topk=True), False),
        (dict(custom_routing_function=lambda **kwargs: None), False),
    ],
)
def test_is_plain_softmax_topk(overrides, expected: bool):
    assert hpu_fused_moe.is_plain_softmax_topk(_routing_layer(**overrides)) is expected


def test_select_experts_sigmoid_bias_affects_selection_only():
    num_experts, top_k, tokens = 16, 4, 8
    router_logits = torch.randn(tokens, num_experts, dtype=torch.bfloat16, device="hpu")
    # A large bias on the last experts forces their selection regardless of the logits.
    bias = torch.zeros(num_experts, dtype=torch.float32, device="hpu")
    bias[-top_k:] = 10.0
    layer = _routing_layer(num_experts,
                           top_k,
                           scoring_func="sigmoid",
                           e_score_correction_bias=bias,
                           routed_scaling_factor=2.5)
    x = torch.empty(tokens, 8, dtype=torch.bfloat16, device="hpu")

    topk_weights, topk_ids = hpu_fused_moe.select_experts_from_routed(layer, x, router_logits)

    assert topk_ids.dtype == torch.int32
    expected_ids = torch.arange(num_experts - top_k, num_experts).expand(tokens, -1)
    torch.testing.assert_close(topk_ids.to(torch.int64).sort(dim=-1).values.cpu(), expected_ids)
    # Weights are the unbiased sigmoid scores, renormalized, then scaled once.
    scores = router_logits.float().sigmoid().gather(1, topk_ids.to(torch.int64))
    expected_weights = scores / scores.sum(dim=-1, keepdim=True) * 2.5
    torch.testing.assert_close(topk_weights, expected_weights)


def test_select_experts_softmax_without_renormalize():
    num_experts, top_k, tokens = 16, 4, 8
    router_logits = torch.randn(tokens, num_experts, dtype=torch.bfloat16, device="hpu")
    layer = _routing_layer(num_experts, top_k, renormalize=False)
    x = torch.empty(tokens, 8, dtype=torch.bfloat16, device="hpu")

    topk_weights, topk_ids = hpu_fused_moe.select_experts_from_routed(layer, x, router_logits)

    ref_weights, ref_ids = torch.topk(torch.softmax(router_logits.float(), dim=-1), top_k, dim=-1)
    torch.testing.assert_close(topk_ids.to(torch.int64), ref_ids)
    torch.testing.assert_close(topk_weights, ref_weights)


def test_select_experts_rejects_unsupported_scoring_func():
    layer = _routing_layer(scoring_func="sqrtsoftplus")
    router_logits = torch.randn(2, 16, device="hpu")
    with pytest.raises(ValueError, match="sqrtsoftplus"):
        hpu_fused_moe.select_experts_from_routed(layer, torch.empty(2, 8, device="hpu"), router_logits)
