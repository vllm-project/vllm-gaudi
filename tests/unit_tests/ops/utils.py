# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
import contextlib
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


@contextlib.contextmanager
def temporary_op_registry_oot():
    """
    Contextmanager which allows to temporarily modify the op registry content.
    It clears current op_registry_oot and restores its content on exit.
    It is useful for testing purposes, e.g. to deregister hpu version
    of the op. (Because when running tests, if registration happened in one
    of them, then it is still valid in every other test).
    """
    old_registry = CustomOp.op_registry_oot
    CustomOp.op_registry_oot = {}
    try:
        yield
    finally:
        CustomOp.op_registry_oot = old_registry


def register_op(base_cls, oot_cls):
    """
    Manual registration of the oot op. It should be used
    within temporary_op_registry_oot context manager.
    """
    CustomOp.op_registry_oot[base_cls.__name__] = oot_cls


def create_row_parallel_linear(input_size, output_size, quant_config=None):
    return RowParallelLinear(input_size=input_size,
                             output_size=output_size,
                             bias=False,
                             input_is_parallel=True,
                             skip_bias_add=False,
                             params_dtype=torch.bfloat16,
                             reduce_results=True,
                             quant_config=quant_config,
                             return_bias=False,
                             disable_tp=False)


def create_fused_moe(quant_config=None):
    return FusedMoE(num_experts=128,
                    top_k=8,
                    hidden_size=512,
                    intermediate_size=256,
                    params_dtype=torch.bfloat16,
                    reduce_results=True,
                    renormalize=True,
                    use_grouped_topk=False,
                    num_expert_group=None,
                    topk_group=None,
                    quant_config=quant_config,
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
                    zero_expert_type=None)


def get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)
