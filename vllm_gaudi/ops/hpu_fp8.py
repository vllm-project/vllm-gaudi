from functools import partial
from typing import Optional

import torch
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm_gaudi import envs
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory as FusedMoE

from vllm.model_executor.layers.quantization import fp8
from vllm.model_executor.layers.quantization.fp8 import (Fp8LinearMethod as OrigFp8LinearMethod, Fp8MoEMethod,
                                                         Fp8Config)
import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOpFP8PerChannel, VllmMixtureOfExpertsOpFP8)
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.ops.hpu_fused_moe import (_normalize_moe_activation, model_has_quant_config, select_experts_from_routed)
from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states, dispatch_tensor, get_hpu_dp_metadata

from vllm.model_executor.kernels.linear import _POSSIBLE_FP8_BLOCK_KERNELS, _POSSIBLE_FP8_KERNELS
from vllm.platforms import PlatformEnum
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    PerTensorTorchFP8ScaledMMLinearKernel,
    ChannelWiseTorchFP8ScaledMMLinearKernel,
)

logger = init_logger(__name__)


# EXPERIMENTAL custom MoE combine: replace the Habana mixture_of_experts op (a
# fixed per-layer stage pipeline) with a pure-PyTorch gathered-expert path.
# Default stock. `VLLM_HPU_MOE_GATHER_VERIFY=1` (with VLLM_HPU_MOE_GATHER=1) runs
# BOTH the custom path and the Habana op on the same inputs and reduces their
# maximum FP8-ULP over the expert-parallel group in-memory (no files).
_HPU_MOE_GATHER = envs.VLLM_HPU_MOE_GATHER
# Only meaningful together with the gather path.
_HPU_MOE_GATHER_VERIFY = envs.VLLM_HPU_MOE_GATHER_VERIFY and _HPU_MOE_GATHER
# Correctness bar for the verify path: any element exceeding this many FP8-ULP
# is an error (matches the archived offline analysis' "no element > 2 ULP" bar).
_HPU_MOE_GATHER_VERIFY_MAX_ULP = 2
# Max tokens*topk (== gathered-expert count g) for which the custom gather path
# is used. The gathered pure-PyTorch path wins below ~g=64 and LOSES to the stock
# fused op once g approaches E (the dense gather + fp32 bmm path is slower than
# the Habana op).
_HPU_MOE_GATHER_MAX_TP = envs.VLLM_HPU_MOE_GATHER_MAX_TP
if _HPU_MOE_GATHER:
    from vllm_gaudi.ops.hpu_moe_combine import gather_silu_fp8_moe  # noqa: E402
else:
    gather_silu_fp8_moe = None

if _HPU_MOE_GATHER_VERIFY:
    logger.info("MoE gather combine VERIFY mode enabled: comparing custom vs stock "
                "per layer (FP8-ULP bar = %d)", _HPU_MOE_GATHER_VERIFY_MAX_ULP)


def _fp8_ulp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sign-aware FP8-ULP distance (ported from the archived `moe_ulp.fp8_ulp`).

    Same-sign pairs use real E4M3 ulps: |uint8(quantize(a)) - uint8(quantize(b))|.
    In E4M3FN adjacent same-sign representable values differ by exactly +/-1 in
    uint8, so this IS the representable-step count. Cross-sign pairs use
    universal subnormal units |a - b| / 2^-9. Returns elementwise Float32 ulps.
    """
    a_fp32 = a.float()
    b_fp32 = b.float()
    a_bits = a_fp32.to(torch.float8_e4m3fn).view(torch.uint8).int()
    b_bits = b_fp32.to(torch.float8_e4m3fn).view(torch.uint8).int()
    same_sign = (a_bits >= 128) == (b_bits >= 128)
    same_sign_ulp = (a_bits - b_bits).abs().float()
    cross_ulp = (a_fp32 - b_fp32).abs() / (2.0 ** -9)
    return torch.where(same_sign, same_sign_ulp, cross_ulp)


def _verify_moe_combine(stock: torch.Tensor, custom: torch.Tensor) -> torch.Tensor:
    if not _HPU_MOE_GATHER_VERIFY:
        return stock
    max_ulp = _fp8_ulp(stock, custom).amax()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        ep_group = get_ep_group()
        if ep_group.world_size > 1:
            torch.distributed.all_reduce(
                max_ulp,
                op=torch.distributed.ReduceOp.MAX,
                group=ep_group.device_group,
            )
    if max_ulp > _HPU_MOE_GATHER_VERIFY_MAX_ULP:
        logger.warning("MoE gather combine mismatch: max FP8-ULP = %s exceeds %d",
                       max_ulp.item(), _HPU_MOE_GATHER_VERIFY_MAX_ULP)
    return max_ulp


class HPUPerTensorTorchFP8ScaledMMLinearKernel(PerTensorTorchFP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None


class HPUChannelWiseTorchFP8ScaledMMLinearKernel(ChannelWiseTorchFP8ScaledMMLinearKernel):

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None


class HPUFp8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """HPU stub for block-scaled FP8 linear.

    The actual computation is handled by HPU-specific ops in
    Fp8LinearMethod.apply(), so this kernel only needs to satisfy
    the kernel selection interface.
    """

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        return True, None

    def apply_weights(self, layer, x, bias=None):
        raise NotImplementedError("HPU uses Fp8LinearMethod.apply() directly")

    def apply_block_scaled_mm(self, A, B, As, Bs):
        raise NotImplementedError("HPU uses Fp8LinearMethod.apply() directly")


if PlatformEnum.OOT not in _POSSIBLE_FP8_KERNELS:
    _POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = [
        HPUPerTensorTorchFP8ScaledMMLinearKernel,
        HPUChannelWiseTorchFP8ScaledMMLinearKernel,
    ]

if PlatformEnum.OOT not in _POSSIBLE_FP8_BLOCK_KERNELS:
    _POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT] = [
        HPUFp8BlockScaledMMLinearKernel,
    ]


class Fp8LinearMethod(OrigFp8LinearMethod):

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.quant_config = self.quant_config
        if self.block_quant:
            layer = hpu_ops.fp8_block_linear_postprocess_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
            return
        # If checkpoint not serialized fp8, quantize the weights.
        elif not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = hpu_ops.scaled_fp8_quant(layer.weight, scale=None)
            weight = qweight.t()

        # If checkpoint is fp8 per-tensor, handle that there are N scales for N
        # shards in a fused module
        else:
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.

            weight, weight_scale, input_scale = hpu_ops.process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

        # Update layer with new values.
        layer.weight = Parameter(weight.data, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
        layer.input_scale = (Parameter(input_scale, requires_grad=False) if input_scale is not None else None)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return hpu_ops.apply_block_fp8_linear_hpu(
                input=x,
                layer=layer,
                block_size=self.quant_config.weight_block_size,
                bias=bias,
                do_unpad=True,
                force_channel_fp8=envs.VLLM_HPU_FORCE_CHANNEL_FP8,
            )

        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, 'input_scale', None)
        input_2d = x.view(-1, x.shape[-1])
        output = hpu_ops.apply_fp8_linear_hpu(input=input_2d,
                                              weight=layer.weight,
                                              weight_scale=weight_scale,
                                              input_scale=input_scale,
                                              bias=bias,
                                              trans_B=False)
        return output.view(*x.shape[:-1], -1)

    def dequant_fp8_weight(self, layer) -> torch.Tensor:
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight
        dequant_weight = hpu_ops.dequant_block_fp8_weight_naive(
            layer.weight,
            layer.weight_scale_inv.data,
            self.quant_config.weight_block_size,
            original_M=layer.orig_M,
            original_N=layer.orig_N,
            do_unpad=True,
        )
        return dequant_weight


class HPUFp8MoEMethod(Fp8MoEMethod):

    def __init__(self, quant_config: Fp8Config, layer: torch.nn.Module):
        super().__init__(quant_config, layer)

        # Disable marlin
        self.use_marlin = False
        self.fp8_backend = False

        # disable DeepGemm support.
        self.allow_deep_gemm = False

        self.use_dispatch_fn = get_config().use_dispatch_fn
        # Snapshot the (static) quant-config flag while the vLLM config context
        # is set; the forward hot path reads this cached value instead.
        self.has_moe_quant_config = model_has_quant_config()

    @property
    def is_monolithic(self) -> bool:
        return True

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        kwargs['weight_loader'] = hpu_ops.synced_weight_loader(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.local_num_experts
        ep_shift = layer.moe_config.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        if layer.moe_config.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.moe_config.is_sequence_parallel)
        else:
            dispatch_fn = None

        if self.block_quant and not envs.VLLM_HPU_FORCE_CHANNEL_FP8:
            layer.moe_op = VllmMixtureOfExpertsOpFP8(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                dispatch_fn,
            )
        else:
            layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
                layer.global_num_experts,
                num_experts,
                experts_min,
                experts_max,
                dispatch_fn,
            )
        if self.block_quant:
            layer = hpu_ops.fp8_block_moe_prepare_weights(layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
        else:
            if self.quant_config.activation_scheme == "static":
                if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                    raise ValueError("QuantConfig has static quantization, but found "
                                     "activation scales are None.")
                layer.w13_input_scale = torch.nn.Parameter(layer.w13_input_scale.max(), requires_grad=False)
            layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        is_sequence_parallel = layer.moe_config.is_sequence_parallel
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # The HPU mixture_of_experts kernel (including the chunked
        # weighted_sum_reduction_bf16 reduction) compiles for int64 routing
        # tables and bf16 (x.dtype) router weights. The grouped-topk /
        # custom-routing helper returns int32 ids and float32 weights; the
        # regular-topk branch above already normalized them, but the grouped
        # path was previously left unconverted -> the bf16 reduction kernel
        # received float32 router_weights and failed to compile
        # (GLUE_INCOMPATIBLE_DATA_TYPE). Normalize for every routing path so the
        # kernel graph receives dtype-consistent inputs.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, is_sequence_parallel)
        elif is_sequence_parallel:
            # See HPUCompressedTensorsW8A8Fp8MoEMethod.apply_monolithic: at
            # dp_size == 1 with sequence-parallel MoE (TP>1 + EP),
            # MoERunner._maybe_combine reduce-scatters the expert output over the
            # EP group but no paired dispatch all-gather runs (dispatch_fn is
            # wired only for dp_size > 1). Restore symmetry by all-gathering the
            # inputs over the EP group so the combine leaves the token count
            # unchanged for the block's post-experts reshape.
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        activation = _normalize_moe_activation(layer.activation)
        # Use the custom gathered-expert combine only when it wins: g = tokens*K
        # must stay small (below the dense crossover). Beyond that (large batch /
        # long prefill) fall back to the stock fused op, which is faster and keeps
        # the graph shapes fixed. `tokens`/`K` are static (T, K from x/topk_ids).
        use_gather = (
            _HPU_MOE_GATHER
            and activation == "silu"
            and self.quant_config.activation_scheme != "static"
            and x.shape[0] * topk_ids.shape[-1] <= _HPU_MOE_GATHER_MAX_TP
        )
        if use_gather:
            # EXPERIMENTAL custom combine: gather only the routed experts
            # (bypasses the Habana op's fixed per-layer stage pipeline).
            if _HPU_MOE_GATHER_VERIFY:
                stock = layer.moe_op(x, topk_ids, topk_weights,
                                     permuted_weights=True, activation=activation)
                custom = gather_silu_fp8_moe(layer, x, topk_ids, topk_weights)
                _verify_moe_combine(stock, custom)
                output = custom
            else:
                output = gather_silu_fp8_moe(layer, x, topk_ids, topk_weights)
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        return output.view(*(output.size(0), *input_shape[1:]))


fp8.Fp8LinearMethod = Fp8LinearMethod
fp8.Fp8MoEMethod = HPUFp8MoEMethod
