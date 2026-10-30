# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU override for GptOssMxfp4MoEMethod.

Monkey-patches the CUDA GptOssMxfp4MoEMethod so that when the quant config
flows through (HPU_MXFP4_NATIVE=1), the HPU-specific kernel setup is used
instead of CUDA kernel infrastructure.
"""
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
import vllm_gaudi.envs as gaudi_envs
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization import mxfp4 as mxfp4_module
from vllm.model_executor.layers.quantization.mxfp4 import GptOssMxfp4MoEMethod, Mxfp4MoEMethod

from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpMXFP4
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.ops.hpu_situ_moe import MXFP4_BLOCK, maybe_situ_moe, situ_impl_or_exact
from vllm_gaudi.ops.hpu_fused_moe import (
    _normalize_moe_activation,
    select_experts_from_routed,
)
from vllm_gaudi.v1.worker.hpu_dp_utils import (
    dispatch_hidden_states,
    dispatch_tensor,
    get_hpu_dp_metadata,
)


class HPUGptOssMxfp4MoEMethod(GptOssMxfp4MoEMethod):
    """HPU override of GptOssMxfp4MoEMethod.

    Reuses parent's create_weights() (allocates uint8 params with correct shapes).
    Overrides process_weights_after_loading to build VllmMixtureOfExpertsOpMXFP4
    instead of CUDA kernels.
    """

    MXFP4_BLOCK_SIZE = 32

    def __init__(self, moe):
        # Call grandparent (FusedMoEMethodBase) init, skip CUDA backend selection
        super(GptOssMxfp4MoEMethod, self).__init__(moe)
        self.weight_dtype = "gpt_oss_mxfp4"
        # No CUDA backend needed
        self.mxfp4_backend = None
        self.experts_cls = None
        self.moe_kernel = None
        self._cache_permute_indices = {}

        self.use_dispatch_fn = get_config().use_dispatch_fn
        torch.hpu.synchronize()
        vllm_config = get_current_vllm_config()
        self.model_type = None
        if (vllm_config is not None and vllm_config.model_config is not None
                and vllm_config.model_config.hf_config is not None):
            self.model_type = vllm_config.model_config.hf_config.model_type

    @property
    def is_monolithic(self) -> bool:
        return True

    def _select_monolithic(self) -> Callable:
        return self.apply_monolithic

    def process_weights_after_loading(self, layer) -> None:
        """Build VllmMixtureOfExpertsOpMXFP4 from the uint8 params loaded by upstream."""
        num_experts = layer.local_num_experts
        ep_shift = layer.moe_config.ep_rank * num_experts
        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1

        if layer.moe_config.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(
                dispatch_hidden_states,
                is_sequence_parallel=layer.moe_config.is_sequence_parallel,
            )
        else:
            dispatch_fn = None

        # GPT-OSS uses SwiGLU-OAI: per-expert bias on both projections, applied
        # by the native op via the GPT_SWIGLU flag (alpha/limit). When the layer
        # carries w13_bias/w2_bias, route through bias_mxfp4_fused_weights.
        has_bias = hasattr(layer, "w13_bias") and hasattr(layer, "w2_bias")

        from vllm_gaudi.patches import _current_config_is_kimi_k3
        layer.moe_op = VllmMixtureOfExpertsOpMXFP4(
            layer.global_num_experts,
            num_experts,
            experts_min,
            experts_max,
            block_size=self.MXFP4_BLOCK_SIZE,
            has_bias=has_bias,
            dispatch_fn=dispatch_fn,
            in_graph_call=_current_config_is_kimi_k3(),
        )

        for expert_id in range(num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
            layer.moe_op.w13_list[expert_id].set_scale(layer.w13_weight_scale.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_scale(layer.w2_weight_scale.data[expert_id])
            if has_bias:
                layer.moe_op.w13_list[expert_id].set_bias(layer.w13_bias.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_bias(layer.w2_bias.data[expert_id])

        layer.moe_op._cache_weight_lists()

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])

        if self.model_type == "gpt_oss":
            topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
        else:
            if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
                # `RoutedExperts` no longer owns a `.router` (upstream PR #41184
                # moved it onto `MoERunner`); route via the shared helper that
                # reproduces upstream's select_experts from the layer's params.
                topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
            else:
                topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
                topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)

        if not (layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None):
            # MXFP4 kernel requires int32 routing table
            topk_ids = topk_ids.to(torch.int32)
            topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
            x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        # The fused kernel has no SiTU: exact unfused routed experts (see ops/hpu_situ_moe.py).
        # Env reads and the SiTU helper run only for SiTU layers; every other model calls the fused op directly.
        output = None
        if getattr(layer.activation, "value", layer.activation) == "situ":
            output = maybe_situ_moe(layer, x, topk_ids, topk_weights,
                                    situ_impl_or_exact(gaudi_envs.HPU_KIMI_SITU_IMPL), MXFP4_BLOCK,
                                    gaudi_envs.VLLM_MINIMAX_M3_MOE_TOKEN_TILE)
        if output is None:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=_normalize_moe_activation(layer.activation),
            )
        if layer.moe_config.dp_size > 1:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)


# Monkey-patch: replace CUDA GptOssMxfp4MoEMethod with HPU version
mxfp4_module.GptOssMxfp4MoEMethod = HPUGptOssMxfp4MoEMethod


class HPUKimiK3Mxfp4MoEMethod(Mxfp4MoEMethod):
    """HPU method for the routed experts of a Kimi-K3 checkpoint that vLLM maps to ``Mxfp4Config``.

    The base ``Mxfp4MoEMethod`` selects a CUDA/ROCm backend in ``__init__`` (``select_deepseek_v4_mxfp4_moe_backend``
    raises on HPU) and builds a kernel after loading. This keeps the upstream ``create_weights`` (uint8 ``w13_weight``,
    ``w2_weight`` and their ``*_scale``) and runs the same HPU MXFP4 op and exact SiTU path as the compressed-tensors
    Kimi-K3 method, through the routed ``apply`` (not monolithic).
    """

    MXFP4_BLOCK_SIZE = HPUGptOssMxfp4MoEMethod.MXFP4_BLOCK_SIZE
    # The upstream class sets this to True (loader hands over pre-processed weights); the HPU path loads the raw
    # checkpoint tensors and repacks them in process_weights_after_loading.
    supports_pre_processed_weights = False

    def __init__(self, moe, activation_key=None):
        super(Mxfp4MoEMethod, self).__init__(moe)
        self.weight_dtype = "mxfp4"
        self.activation_key = activation_key
        self.is_k3_situ_aiter = False
        self.is_k3_situ_int4_gfx942 = False
        self.mxfp4_backend = None
        self.experts_cls = None
        self.moe_kernel = None
        self.group_size = self.MXFP4_BLOCK_SIZE
        self.max_capture_size = moe.max_capture_size
        self._cache_permute_indices = {}
        self.w13_precision_config = None
        self.w2_precision_config = None
        self.moe_quant_config = object()
        self.use_dispatch_fn = get_config().use_dispatch_fn

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def supports_eplb(self) -> bool:
        # vLLM EPLB is not wired into the HPU runner
        return False

    @property
    def skip_forward_padding(self) -> bool:
        return False

    @property
    def has_unpadded_output(self) -> bool:
        return False

    def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config):
        # Skip Mxfp4MoEMethod's own step: its rounding is keyed on ``mxfp4_backend``, which HPU does not have. The
        # FusedMoEMethodBase contract (the next class in the MRO) still applies.
        return super(Mxfp4MoEMethod, self).maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )

    def process_weights_after_loading(self, layer) -> None:
        HPUGptOssMxfp4MoEMethod.process_weights_after_loading(self, layer)
        self.moe_quant_config = object()

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        from vllm_gaudi.patches import _hpu_mxfp4_routed_apply
        return _hpu_mxfp4_routed_apply(layer, x, topk_weights, topk_ids, self.group_size)


def _route_kimi_k3_moe_method() -> None:
    """Make ``Mxfp4Config`` build the HPU method for Kimi-K3 (and only Kimi-K3).

    ``Mxfp4Config._make_moe_method`` returns ``Mxfp4MoEMethod``, whose ``__init__`` fails on HPU. The wrapper is
    active from the moment this module is imported, i.e. before any model is built, so it does not depend on the
    Kimi-K3 patches having been installed. Other models, and ``GptOssMxfp4Config`` (which overrides the method),
    keep the upstream behaviour.
    """
    from vllm_gaudi.patches import _current_config_is_kimi_k3
    config_cls = mxfp4_module.Mxfp4Config
    if getattr(config_cls, "_hpu_kimi_k3_routed", False):
        return
    orig = config_cls._make_moe_method

    def _make_moe_method(self, moe):
        if _current_config_is_kimi_k3():
            return HPUKimiK3Mxfp4MoEMethod(moe)
        return orig(self, moe)

    config_cls._make_moe_method = _make_moe_method
    config_cls._hpu_kimi_k3_routed = True


_route_kimi_k3_moe_method()
