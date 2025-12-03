from typing import Callable, Optional, Union
import habana_frameworks.torch as htorch
import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE, FusedMoEConfig)
from compressed_tensors.quantization import (QuantizationArgs, QuantizationStrategy)

from vllm.model_executor.layers.quantization.utils.w8a8_utils import convert_to_channelwise
from vllm.model_executor.parameter import (ChannelQuantScaleParameter, ModelWeightParameter, PerTensorScaleParameter,
                                           BasevLLMParameter, GroupQuantScaleParameter, PackedColumnParameter,
                                           PackedvLLMParameter, RowvLLMParameter)
from vllm.model_executor.layers.quantization.compressed_tensors import (compressed_tensors)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod as OrigCompressedTensorsLinearMethod, CompressedTensorsConfig)
from vllm.model_executor.layers.quantization.compressed_tensors import (compressed_tensors_moe)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
    CompressedTensorsScheme, CompressedTensorsWNA16)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_TYPES_MAP)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (find_matched_target)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsW8A8Fp8MoEMethod, CompressedTensorsWNA16MarlinMoEMethod)
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (MPLinearKernel, MPLinearLayerConfig)
from vllm.model_executor.layers.quantization.utils.quant_utils import (pack_quantized_values_into_int32,
                                                                       unpack_quantized_values_into_int32)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (marlin_repeat_scales_on_all_ranks)
from vllm.model_executor.utils import set_weight_attrs
import vllm_gaudi.extension.ops as hpu_ops
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOpFP8PerChannel, VllmMixtureOfExpertsOpWNA16)

SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR]

logger = init_logger(__name__)


@CustomOp.register_oot(name='CompressedTensorsLinearMethod')
class HPUCompressedTensorsLinearMethod(OrigCompressedTensorsLinearMethod):

    def __init__(self, quantization_config: CompressedTensorsConfig):
        super().__init__(quantization_config)
        torch.hpu.synchronize()

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")

        # Explicitly override scheme since register_oot and monkey-patching not working
        layer.scheme = self.get_hpu_scheme(layer)
        layer.scheme.create_weights(layer=layer,
                                    input_size=input_size,
                                    input_size_per_partition=input_size_per_partition,
                                    output_partition_sizes=output_partition_sizes,
                                    output_size=output_size,
                                    params_dtype=params_dtype,
                                    weight_loader=weight_loader)

    def get_hpu_scheme(self, layer: torch.nn.Module):
        scheme = layer.scheme
        if scheme is None:
            raise ValueError("A scheme must be defined for each layer")
        scheme_classname = scheme.__class__.__name__
        if (scheme_classname in ("CompressedTensorsW8A8Fp8", "CompressedTensorsW8A16Fp8")):
            hpu_scheme = HPUCompressedTensorsW8A8Fp8(scheme.strategy, scheme.is_static_input_scheme)
        elif (scheme_classname == "CompressedTensorsWNA16"):
            matched_target = find_matched_target(layer_name=layer.prefix,
                                                 module=layer,
                                                 targets=self.quantization_config.target_scheme_map.keys(),
                                                 fused_mapping=self.quantization_config.packed_modules_mapping)

            scheme_dict = self.quantization_config.target_scheme_map[matched_target]
            weight_quant = scheme_dict.get("weights")

            hpu_scheme = HPUCompressedTensorsWNA16(num_bits=weight_quant.num_bits,
                                                   strategy=scheme.strategy,
                                                   symmetric=scheme.symmetric,
                                                   group_size=scheme.group_size,
                                                   actorder=weight_quant.actorder)
        else:
            raise ValueError(f"{scheme_classname} compressed format is not supported on HPU")
        return hpu_scheme

    def dequant_fp8_weight(self, layer: torch.nn.Module) -> torch.Tensor:
        if layer.scheme.strategy == QuantizationStrategy.CHANNEL:  # weights were quantized per-channel
            dequant_weight = layer.weight.to(layer.weight_scale.dtype) * layer.weight_scale.squeeze()
            return dequant_weight.to(torch.bfloat16).t()
        else:
            raise NotImplementedError("Implemented per-channel dequantization only")


@CustomOp.register_oot(name='CompressedTensorsW8A8Fp8')
class HPUCompressedTensorsW8A8Fp8(CompressedTensorsScheme):

    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if layer.scheme.strategy == QuantizationStrategy.TENSOR:
            ws_channelwise = convert_to_channelwise(layer.weight_scale, layer.logical_widths)
            layer.weight_scale = torch.nn.Parameter(ws_channelwise, requires_grad=False)
        else:
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data, requires_grad=False)

        # Weights must be transposed for marlin
        layer.weight = torch.nn.Parameter(layer.weight.t(), requires_grad=False)

        # see the reference: https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py#L169-L173
        if layer.scheme.is_static_input_scheme and hasattr(layer, "input_scale"):
            # required by torch.compile to be torch.nn.Parameter, only per-tensor supported
            layer.input_scale = torch.nn.Parameter(layer.input_scale.max(), requires_grad=False)
        else:
            layer.input_scale = None

        # postprocess weights for perchannel strategy
        if layer.scheme.strategy == QuantizationStrategy.CHANNEL:
            hpu_ops.fp8_perchannel_linear_postprocess_weights(layer)

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        """
        Use the CompressedTensorsScheme associated with each layer to create
        the necessary parameters for the layer. See LinearMethodBase for param
        details
        """
        weight_loader = extra_weight_attrs.get("weight_loader")
        if hpu_ops.is_hpu_gaudi2:
            weight_loader = hpu_ops.gaudi_weight_wrapper(weight_loader)
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(output_size_per_partition,
                                                       input_size_per_partition,
                                                       dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if layer.scheme.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(data=torch.empty((sum(output_partition_sizes), 1),
                                                                       dtype=torch.float32),
                                                      output_dim=0,
                                                      weight_loader=weight_loader)
        elif layer.scheme.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)
        else:
            raise ValueError(f"Unsupported weight strategy={layer.scheme.strategy}, "
                             f"supported strategies are {SUPPORTED_STRATEGIES}")

        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (to deal with converted checkpoints)
        if layer.scheme.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None):
        weight_scale = layer.weight_scale.transpose(0, 1) if layer.weight_scale.dim() > 1 else layer.weight_scale
        input_scale = getattr(layer, 'input_scale', None)
        return hpu_ops.apply_fp8_linear_hpu(input=x,
                                            weight=layer.weight,
                                            weight_scale=weight_scale,
                                            input_scale=input_scale,
                                            bias=bias,
                                            trans_B=False)


@CustomOp.register_oot(name='CompressedTensorsW8A8Fp8MoEMethod')
class HPUCompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsW8A8Fp8MoEMethod):
    """MoE method without quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.hpu.synchronize()

    def create_weights(self, *args, **kwargs) -> None:
        if hpu_ops.is_hpu_gaudi2:
            kwargs['weight_loader'] = hpu_ops.gaudi_weight_wrapper(kwargs.get('weight_loader'))
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        #NOTE: This method is called after the weights are loaded.
        # super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
            num_experts,
            experts_min,
            experts_max,
        )

        layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)
        return

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        **kwargs,
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if use_grouped_topk or custom_routing_function is not None:
            topk_weights, topk_ids, zero_expert_result = layer.select_experts(hidden_states=x,
                                                                              router_logits=router_logits)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=True,
            activation=activation,
        )
        return output.view(*input_shape)


class HPUCompressedTensorsWNA16(CompressedTensorsWNA16):
    _kernel_backends_being_used: set[str] = set()

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    def create_weights(self, layer: torch.nn.Module, output_size: int, input_size: int,
                       output_partition_sizes: list[int], input_size_per_partition: int, params_dtype: torch.dtype,
                       weight_loader: Callable, **kwargs):
        output_size_per_partition = sum(output_partition_sizes)

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=\
                (input_size_per_partition, output_size_per_partition),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx
        )

        kernel_type = HPUMPLinearKernel

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsWNA16", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        partition_scales = not marlin_repeat_scales_on_all_ranks(self.has_g_idx, self.group_size, row_parallel)

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=self.pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         input_size_per_partition // self.pack_factor,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }

        zeros_args = {
            "weight_loader": weight_loader,
            "data": torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            )
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(output_dim=0, packed_dim=0, packed_factor=self.pack_factor, **zeros_args)
        else:
            weight_scale = GroupQuantScaleParameter(output_dim=0, input_dim=1, **weight_scale_args)
            if not self.symmetric:
                qzeros = PackedvLLMParameter(input_dim=1,
                                             output_dim=0,
                                             packed_dim=0,
                                             packed_factor=self.pack_factor,
                                             **zeros_args)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
                                            input_dim=0,
                                            weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", weight_g_idx)

        self.kernel = kernel_type(mp_linear_kernel_config,
                                  w_q_param_name="weight_packed",
                                  w_s_param_name="weight_scale",
                                  w_zp_param_name="weight_zero_point",
                                  w_gidx_param_name="weight_g_idx")


class HPUMPLinearKernel(MPLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, Optional[str]]:
        bits = c.weight_type.size_bits
        assert bits == 4, f"w{bits}a16 not yet supported on HPU"
        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            qweight = unpack_quantized_values_into_int32(x.data.transpose(0, 1), c.weight_type, packed_dim=0)
            x.data = pack_quantized_values_into_int32(qweight, c.weight_type, packed_dim=1)

            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            x.data = x.data.transpose(0, 1).contiguous()
            return x

        def transform_w_zp(x):
            x.data = x.data.transpose(0, 1).contiguous()
            return x

        if c.zero_points:
            self._transform_param(layer, self.w_zp_name, transform_w_zp)
        else:
            self.w_zp_name: str = "qzeros"
            device = getattr(layer, self.w_q_name).device
            # use groups=1 for channelwise quantization
            groups = (c.partition_weight_shape[0] // c.group_size) if c.group_size > 0 else 1
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # if the type has a bias we have to create a zeros tensor that
                # contains the bias values repeated for each group
                # Documentation of the bug can be found here:
                #  https://garden.danieldk.eu/GPTQ-Checkpoint-Format
                zeros = torch.full((groups, out_features), c.weight_type.bias, dtype=torch.int32, device=device)
            else:
                raise NotImplementedError("A 0 zero-point is not supported on HPU compressed wNa16 format")
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            setattr(layer, self.w_zp_name, torch.nn.Parameter(zeros, requires_grad=False))

        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        reshaped_x = x.reshape(-1, x.shape[-1])
        out_shape = x.shape[:-1] + (c.partition_weight_shape[1], )

        weight = torch.ops.hpu.convert_from_uint4(w_q, w_s, w_zp, x.dtype, w_gidx)
        output = torch.matmul(reshaped_x, weight)

        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)


@CustomOp.register_oot(name='CompressedTensorsWNA16MarlinMoEMethod')
class HPUCompressedTensorsWNA16MoEMethod(CompressedTensorsWNA16MarlinMoEMethod):

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(weight_quant, input_quant, moe)

        self.weight_quant = weight_quant
        self.input_quant = input_quant
        assert weight_quant.symmetric, ("Only symmetric quantization is supported for MoE")
        self.quant_type = WNA16_SUPPORTED_TYPES_MAP[self.num_bits]
        self.layer_name = layer_name

    def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs):
        extra_weight_attrs["intermediate_size_full"] = intermediate_size_per_partition * layer.tp_size

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({"is_transposed": False, "quant_method": self.strategy})
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size_per_partition,
                                                    hidden_size // self.packed_factor,
                                                    dtype=torch.int32),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size,
                                                   intermediate_size_per_partition // self.packed_factor,
                                                   dtype=torch.int32),
                                       requires_grad=False)
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_scales_size = intermediate_size_per_partition

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                  2 * intermediate_size_per_partition,
                                                  num_groups_w13,
                                                  dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts, hidden_size, num_groups_w2, dtype=params_dtype),
                                      requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": False})

        w2_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2), requires_grad=False)
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(torch.empty(num_experts, 2), requires_grad=False)

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

        # Shared zero points for converting symmetric weights on HPU
        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            w2_scales_size = intermediate_size_per_partition
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_zeros = torch.full((num_groups_w13, 2 * intermediate_size_per_partition),
                               self.quant_type.bias,
                               dtype=torch.int32)
        w13_zeros = pack_quantized_values_into_int32(w13_zeros, self.quant_type, packed_dim=1)
        layer.register_parameter("w13_zero_point", torch.nn.Parameter(w13_zeros, requires_grad=False))
        w2_zeros = torch.full((num_groups_w2, hidden_size), self.quant_type.bias, dtype=torch.int32)
        w2_zeros = pack_quantized_values_into_int32(w2_zeros, self.quant_type, packed_dim=1)
        layer.register_parameter("w2_zero_point", torch.nn.Parameter(w2_zeros, requires_grad=False))

        layer.a13_scale = None
        layer.a2_scale = None

    def gptq_hpu_moe_repack(self, b_q_weight: torch.Tensor) -> torch.Tensor:
        num_experts = b_q_weight.shape[0]
        outputs = []
        for e in range(num_experts):
            weight = unpack_quantized_values_into_int32(b_q_weight[e].data.contiguous().transpose(0, 1),
                                                        self.quant_type,
                                                        packed_dim=0)
            q_weight = pack_quantized_values_into_int32(weight, self.quant_type, packed_dim=1)
            outputs.append(q_weight)

        return torch.stack(outputs, dim=0)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reconfigure packed weights and scales to match moe_wna16 format
        w13_weight_packed = self.gptq_hpu_moe_repack(layer.w13_weight_packed)
        w2_weight_packed = self.gptq_hpu_moe_repack(layer.w2_weight_packed)

        # for torch.compile
        layer.w13_weight_packed = torch.nn.Parameter(w13_weight_packed, requires_grad=False)
        layer.w2_weight_packed = torch.nn.Parameter(w2_weight_packed, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.data.transpose(1, 2).contiguous(),
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(layer.w2_weight_scale.data.transpose(1, 2).contiguous(),
                                                   requires_grad=False)

        # Initialize HPU MoE op
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        layer.moe_op = VllmMixtureOfExpertsOpWNA16(
            num_experts,
            experts_min,
            experts_max,
        )
        for expert_id in range(layer.local_num_experts):
            layer.moe_op.w13_list[expert_id].set_weight_packed(layer.w13_weight_packed.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight_packed(layer.w2_weight_packed.data[expert_id])
            layer.moe_op.w13_list[expert_id].set_weight_scale(layer.w13_weight_scale.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight_scale(layer.w2_weight_scale.data[expert_id])
            layer.moe_op.w13_list[expert_id].set_zero_point(layer.w13_zero_point.data)
            layer.moe_op.w2_list[expert_id].set_zero_point(layer.w2_zero_point.data)

            if self.actorder == "group":
                layer.moe_op.w13_list[expert_id].set_g_idx(layer.w13_weight_g_idx.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_g_idx(layer.w2_weight_g_idx.data[expert_id])

        htorch.core.mark_step()

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        input_shape = x.shape
        x = x.view(-1, x.shape[-1])

        if use_grouped_topk or custom_routing_function is not None:
            topk_weights, topk_ids, zero_expert_result = layer.select_experts(hidden_states=x,
                                                                              router_logits=router_logits)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=False,
            activation=activation,
        )
        return output.view(*input_shape)


compressed_tensors.CompressedTensorsLinearMethod = \
    HPUCompressedTensorsLinearMethod
compressed_tensors_moe.CompressedTensorsW8A8Fp8MoEMethod = \
    HPUCompressedTensorsW8A8Fp8MoEMethod
compressed_tensors_moe.CompressedTensorsWNA16MoEMethod = \
    HPUCompressedTensorsWNA16MoEMethod
compressed_tensors_moe.CompressedTensorsWNA16MarlinMoEMethod = \
    HPUCompressedTensorsWNA16MoEMethod # Override default WNA16 MoE method

# support weight_loader_v2
WEIGHT_LOADER_V2_SUPPORTED.append(HPUCompressedTensorsLinearMethod.__name__)
