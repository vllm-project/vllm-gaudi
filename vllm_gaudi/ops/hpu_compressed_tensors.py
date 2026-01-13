from typing import Callable, Optional, Union, Any
import habana_frameworks.torch as htorch
import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE, FusedMoEConfig)
from compressed_tensors.quantization import (QuantizationArgs, QuantizationStrategy)

from vllm.model_executor.layers.quantization.utils.w8a8_utils import convert_to_channelwise, all_close_1d
from vllm.model_executor.parameter import (ChannelQuantScaleParameter, ModelWeightParameter, PerTensorScaleParameter,
                                           BasevLLMParameter, GroupQuantScaleParameter, PackedColumnParameter,
                                           PackedvLLMParameter, RowvLLMParameter)
from vllm.model_executor.layers.quantization.compressed_tensors import (compressed_tensors)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod as OrigCompressedTensorsLinearMethod, CompressedTensorsConfig,
    CompressedTensorsMoEMethod, CompressedTensorsKVCacheMethod)
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
from vllm_gaudi.extension.scales import ConvertScaleToHwAligned
from vllm_gaudi.extension.ops import (VllmMixtureOfExpertsOpFP8PerChannel, VllmMixtureOfExpertsOpWNA16)
from vllm_gaudi.extension.runtime import get_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase, )
import vllm.model_executor.model_loader.weight_utils as vllm_weight_utils

logger = init_logger(__name__)
SUPPORTED_STRATEGIES = [QuantizationStrategy.CHANNEL, QuantizationStrategy.TENSOR]


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

        # If channelwise, scales are already lined up
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
            input_scale = layer.input_scale.max()
            # hw aligned
            if get_config().use_hpu_aligned_scale:
                input_scale = ConvertScaleToHwAligned().calc(input_scale)
            layer.input_scale = torch.nn.Parameter(input_scale, requires_grad=False)
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

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
    ):
        """
        Copied from CompressedTensorsW8A8Fp8MoEMethod.__init__: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py#L665
        The only differences are:
            - remove some useless code.
            - extend per-channel weight and per-tensor activation format
        """
        CompressedTensorsMoEMethod.__init__(self, moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        per_tensor = (self.weight_quant.strategy == QuantizationStrategy.TENSOR
                      and self.input_quant.strategy == QuantizationStrategy.TENSOR)
        per_channel_token = (self.weight_quant.strategy == QuantizationStrategy.CHANNEL
                             and self.input_quant.strategy == QuantizationStrategy.TOKEN)

        # extend format
        per_channel_tensor = (self.weight_quant.strategy == QuantizationStrategy.CHANNEL
                              and self.input_quant.strategy == QuantizationStrategy.TENSOR)

        if not (per_tensor or per_channel_token or per_channel_tensor):
            assert self.weight_quant.strategy == QuantizationStrategy.BLOCK
            self.weight_block_size = self.weight_quant.block_structure
            assert self.weight_quant.dynamic is not None
        else:
            self.weight_block_size = None
        self.block_quant = self.weight_block_size is not None

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel_token:
            raise ValueError("For FP8 Fused MoE layer, we require either per tensor or "
                             "channelwise, dynamic per token quantization.")

        self.use_marlin = False
        self.fp8_backend = False
        self.disable_expert_map = False

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
            layer.global_num_experts,
            num_experts,
            experts_min,
            experts_max,
        )

        if self.static_input_scales:
            assert self.input_quant.strategy == QuantizationStrategy.TENSOR
            if (layer.w13_input_scale is None or layer.w2_input_scale is None):
                raise ValueError("QuantConfig has static quantization, but found "
                                 "activation scales are None.")

            if (not all_close_1d(layer.w13_input_scale)):
                logger.warning_once("Found input_scales that are not equal for "
                                    "fp8 MoE layer. Using the maximum across experts "
                                    "for each layer.")
            layer.w13_input_scale = torch.nn.Parameter(layer.w13_input_scale.max(), requires_grad=False)

        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            assert layer.w13_weight_scale is not None
            # Convert per-tensor weight scales to per-channel format
            # by repeating scale values across the intermediate dimension.
            w13_s0 = layer.w13_weight_scale[:, :1]
            w13_s1 = layer.w13_weight_scale[:, 1:]
            w13_s0_exp = torch.repeat_interleave(w13_s0, repeats=layer.intermediate_size_per_partition, dim=1)
            w13_s1_exp = torch.repeat_interleave(w13_s1, repeats=layer.intermediate_size_per_partition, dim=1)
            w13_weight_scale_channel = torch.cat([w13_s0_exp, w13_s1_exp],
                                                 dim=1).unsqueeze(-1).to(device=layer.w13_weight_scale.device,
                                                                         dtype=torch.float32)
            layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale_channel, requires_grad=False)

            w2_weight_scale_channel = torch.empty((layer.local_num_experts, layer.hidden_size, 1),
                                                  dtype=torch.float32,
                                                  device=layer.w2_weight_scale.device)

            w2_weight_scale_channel[:, :, 0] = layer.w2_weight_scale.reshape(-1, 1)
            layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale_channel, requires_grad=False)

        layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)
        return

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = layer.select_experts(hidden_states=x, router_logits=router_logits)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=True,
            activation=layer.activation,
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
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        input_shape = x.shape
        x = x.view(-1, x.shape[-1])

        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = layer.select_experts(hidden_states=x, router_logits=router_logits)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=False,
            activation=layer.activation,
        )
        return output.view(*input_shape)


class HPUCompressedTensorsKVCacheMethodForMLA(CompressedTensorsKVCacheMethod):

    def _convert_all_scale_to_nn_param(self, module: torch.nn.Module, scale_attrs: list[str]) -> None:
        for scale_attr in scale_attrs:
            scale_value = getattr(module, scale_attr, None)
            if scale_value is not None and not isinstance(scale_value, torch.nn.Parameter):
                scale_param = torch.nn.Parameter(torch.tensor(scale_value, dtype=torch.float32), requires_grad=False)
                setattr(module, scale_attr, scale_param)

    def _remove_attrs(self, layer: torch.nn.Module, attrs_lst: list[str]) -> None:
        for attr_name in attrs_lst:
            if hasattr(layer, attr_name):
                delattr(layer, attr_name)
                logger.debug_once(f"Removed attribute {attr_name} from layer {layer}")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process KV cache scales for cross-platform FP8 quantization compatibility."""
        super().process_weights_after_loading(layer)
        # The `k_scale` and `v_scale` are loaded from checkpoint without any adjustment.
        # Compute KV scales based on quantization and deployment platforms
        fp8_max_original = 448.0 if get_config().scale_adjustment else 240.0
        max_k = layer._k_scale * fp8_max_original
        max_v = layer._v_scale * fp8_max_original
        max_kv = max(max_k, max_v)
        fp8_max_cur_platform = 240.0 if hpu_ops.is_hpu_gaudi2 else 448.0
        kv_scale = fp8_max_cur_platform / max_kv
        # Configure latent cache and matmul scales
        layer.impl.latent_cache_k.input_scale = kv_scale
        layer.impl.latent_cache_k.output_scale = 1.0 / kv_scale
        # TODO(yiliu30): Support loading q_scale from checkpoint
        layer.impl.matmul_qk.scale_input = 1.0
        layer.impl.matmul_qk.scale_other = kv_scale
        # For `a` in a@v, as `a` is the output of softmax, its max value is 1.0
        layer.impl.matmul_av.scale_input = 1.0
        layer.impl.matmul_av.scale_other = kv_scale

        # Note: The following steps are important to avoid compiling each decoding layer into a different gc recipe
        # Step 1: Remove deprecated scale attributes
        old_scale_attrs = ["_k_scale", "_v_scale", "_q_scale", "_k_scale_float", "_v_scale_float", "_q_scale_float"]
        self._remove_attrs(layer, attrs_lst=old_scale_attrs)

        # Step 2: Convert scales in submodules to nn.Parameter to avoid compiling each layer into different gc recipe
        submodules_to_check = ["latent_cache_k", "matmul_qk", "matmul_av"]
        scale_attrs = ["input_scale", "output_scale", "scale_input", "scale_other"]
        for submodule_name in submodules_to_check:
            submodule = getattr(layer.impl, submodule_name, None)
            if submodule is not None:
                self._convert_all_scale_to_nn_param(submodule, scale_attrs=scale_attrs)


class HPUCompressedTensorsConfig(CompressedTensorsConfig):

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import MLAAttention
        if isinstance(layer, MLAAttention):
            return HPUCompressedTensorsKVCacheMethodForMLA(self)
        else:
            return super().get_quant_method(layer, prefix)

    @classmethod
    def _update_scale_adjustment_if_needed(cls, config: dict[str, Any]) -> None:
        """Update scale adjustment setting based on quantization configuration.
        
        On G2, this method automatically disables scale adjustment when 
        the model is already calibrated/quantized with FP8 E4M3 FNUZ format.
        
        Background:
        -----------
        Scale adjustment is a mechanism to handle cross-platform FP8 quantization compatibility:
        - Gaudi2 (G2) uses FP8 E4M3 FNUZ with max value ~240
        - Gaudi3/Other GPUs use FP8 E4M3 FN with max value ~448
        
        When a model is quantized on G3 (max=448) but deployed on G2 (max=240), scale
        adjustment is needed to rescale the quantization parameters. However, if the model
        is already calibrated with FNUZ format, no adjustment is needed as it's already 
        in the target format.
        
        Scale Adjustment Scenarios:
        | No | Quant Plat   | Deply Plat | Scale Adj | FP8 Max Quant | FP8 Max Deploy |
        |----|--------------|------------|-----------|---------------|----------------|
        | 1  | G2           | G2         | OFF       | 240           | 240            | <- No adjustment needed
        | 2  | G3/Other GPU | G2         | ON        | 448           | 240            |
        | 3  | G3/Other GPU | G3         | OFF       | 448           | 448            |
        """
        # Early exit if scale adjustment is already disabled or not on Gaudi2
        if get_config().scale_adjustment is False or not hpu_ops.is_hpu_gaudi2:
            return

        # Check if model was calibrated with FNUZ format (Gaudi2 native format)
        fp8_dtype_flavor = config.get("fp8_dtype_flavor")

        GAUDI2_NATIVE_FP8_FORMAT = "float8_e4m3fnuz"

        if fp8_dtype_flavor == GAUDI2_NATIVE_FP8_FORMAT:
            # Disable scale adjustment since model was calibrated/quantized for Gaudi2 hardware
            get_config().scale_adjustment = False
            logger.warning_once(f"Detected model calibrated/quantized with {GAUDI2_NATIVE_FP8_FORMAT} format. "
                                "Disabling scale adjustment as the model is already compatible with Gaudi2 hardware.")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CompressedTensorsConfig":
        cls._update_scale_adjustment_if_needed(config)
        return super().from_config(config)

compressed_tensors.CompressedTensorsLinearMethod = \
    HPUCompressedTensorsLinearMethod
compressed_tensors_moe.CompressedTensorsW8A8Fp8MoEMethod = \
    HPUCompressedTensorsW8A8Fp8MoEMethod
compressed_tensors_moe.CompressedTensorsWNA16MoEMethod = \
    HPUCompressedTensorsWNA16MoEMethod
compressed_tensors_moe.CompressedTensorsWNA16MarlinMoEMethod = \
    HPUCompressedTensorsWNA16MoEMethod # Override default WNA16 MoE method
compressed_tensors.CompressedTensorsConfig = HPUCompressedTensorsConfig

# support weight_loader_v2
WEIGHT_LOADER_V2_SUPPORTED.append(HPUCompressedTensorsLinearMethod.__name__)


def oot_maybe_remap_kv_scale_name(name: str, params_dict: dict) -> str | None:
    """Remap the name of FP8 k/v_scale parameters.

    This function handles the remapping of FP8 k/v_scale parameter names.
    It detects if the given name ends with a suffix and attempts to remap
    it to the expected name format in the model. If the remapped name is not
    found in the params_dict, a warning is printed and None is returned.

    Args:
        name (str): The original loaded checkpoint parameter name.
        params_dict (dict): Dictionary containing the model's named parameters.

    Returns:
        str: The remapped parameter name if successful, or the original name
             if no remapping is needed.
        None: If the remapped name is not found in params_dict.
    """

    if name.endswith(".kv_scale"):
        logger.warning_once("DEPRECATED. Found kv_scale in the checkpoint. "
                            "This format is deprecated in favor of separate k_scale and "
                            "v_scale tensors and will be removed in a future release. "
                            "Functionally, we will remap kv_scale to k_scale and duplicate "
                            "k_scale to v_scale")
        # NOTE: we remap the deprecated kv_scale to k_scale
        remapped_name = name.replace(".kv_scale", ".attn.k_scale")
        if remapped_name not in params_dict:
            logger.warning_once(
                "Found kv_scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv_scale is not loaded.",  #  noqa: E501
                name,
                remapped_name,
            )
            return None
        return remapped_name

    if any("mla_attn" in key for key in params_dict):
        attn_str = "mla_attn.mla_attn"
        logger.debug_once(f"Found mla_attn with k_scale and v_scale in "
                          f"the checkpoint, using {attn_str} as attn_str")
    else:
        attn_str = "attn"
    # Define scale name mapping patterns in order of precedence
    scale_mapping_patterns = [
        # LLMC format:  .self_attn.{q,k,v}_scale ->
        #   .attn.{attn_str}.{q,k,v}_scale
        (
            r"\.self_attn\.([qkv])_scale$",
            rf".self_attn.{attn_str}.\1_scale",
        ),
        (
            r"\.self_attn\.([kv])_proj\.([kv])_scale$",
            rf".self_attn.{attn_str}.\2_scale",
        ),
        # ModelOpt format: .self_attn.{k,v}_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (
            r"\.self_attn\.([kv])_proj\.([kv])_scale$",
            rf".self_attn.{attn_str}.\2_scale",
        ),
        # QKV proj format: .self_attn.qkv_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.qkv_proj\.([kv])_scale$", r".self_attn.attn.\1_scale"),
        # Qwen3 MoE format: .self_attn.qkqkv_proj.{k,v}_scale ->
        # .self_attn.attn.{k,v}_scale
        (r"\.self_attn\.qkqkv_proj\.([kv])_scale$", r".self_attn.attn.\1_scale"),
        # Default format: .{k,v}_scale -> .attn.{k,v}_scale
        (r"\.([kv])_scale$", r".attn.\1_scale"),
    ]

    # Check if name ends with k_scale or v_scale
    if name.endswith((".k_scale", ".v_scale", ".q_scale")):
        import regex as re

        for pattern, replacement in scale_mapping_patterns:
            if re.search(pattern, name):
                remapped_name = re.sub(pattern, replacement, name)
                if remapped_name not in params_dict:
                    scale_type = name.split(".")[-1]
                    logger.warning_once(
                        "Found %s in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). %s is not loaded.",  # noqa: E501
                        scale_type,
                        name,
                        remapped_name,
                        scale_type,
                    )
                    return None
                return remapped_name

    # If there were no matches, return the untouched param name
    return name


# Patch the `maybe_remap_kv_scale_name` function to load k/v scale correctly
vllm_weight_utils.maybe_remap_kv_scale_name = oot_maybe_remap_kv_scale_name
