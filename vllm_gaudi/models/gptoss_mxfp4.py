from collections.abc import Iterable

import torch
from transformers import PretrainedConfig
from vllm.model_executor.models.gpt_oss import GptOssModel
from vllm.utils.math_utils import cdiv
from vllm.transformers_utils.model_arch_config_convertor import ModelArchConfigConvertorBase
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_BLOCK_SIZE
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

# Store original implementations before patching
_original_normalize_quantization_config = ModelArchConfigConvertorBase._normalize_quantization_config
_original_load_weights = GptOssModel.load_weights


def _patched_normalize_quantization_config(self, config: PretrainedConfig):
    # Skip mxfp4 quantization to use custom loading logic for gpt_oss
    if getattr(config, "model_type", None) == "gpt_oss":
        quant_cfg = getattr(config, "quantization_config", None)
        if quant_cfg is not None and quant_cfg.get("quant_method", "").lower() == "mxfp4":
            return None

    # For all other models, use the original vLLM implementation
    return _original_normalize_quantization_config(self, config)


def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    # Large default chosen to process many rows per kernel launch and reduce overhead;
    # lower this if you need to limit peak memory usage.
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """
    Convert the mxfp4 weights, dequantize and make them compatible with the forward
    pass of GPT_OSS.
    """
    import math

    FP4_VALUES = [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]

    # MxFP4 stores the scale as an unsigned 8-bit exponent with a bias of 127
    # (i.e. values 0–255 represent exponents in the range -127…128). Subtract 127
    # to recover the signed exponent that torch.ldexp expects.
    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp, sub

    out = out.reshape(*prefix_shape, G * B * 2).contiguous()
    del blocks, scales, lut
    return out


def _load_weights_mxfp4_dequantize_hpu(
    self,
    ep_rank_end: int,
    ep_rank_start: int,
    heads_per_rank: int,
    head_start: int,
    weights: Iterable[tuple[str, torch.Tensor]],
    stacked_params_mapping: list[tuple[str, ...]],
) -> set[str]:
    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    use_ep = self.parallel_config.enable_expert_parallel

    # In MoE, we need to flatten the tensor parallel size across the data
    # parallel size when EP is disabled.
    tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
        tp_size=get_tensor_model_parallel_world_size(),
        dp_size=get_dp_group().world_size,
        dp_rank=get_dp_group().rank_in_group,
        pcp_size=get_pcp_group().world_size,
        pcp_rank=get_pcp_group().rank_in_group,
    )
    intermediate_size = self.config.intermediate_size
    intermediate_size_block = intermediate_size // OCP_MX_BLOCK_SIZE
    per_rank_intermediate_size_block = cdiv(intermediate_size_block, tp_size)
    per_rank_intermediate_size = per_rank_intermediate_size_block * OCP_MX_BLOCK_SIZE

    # Calculate common slicing bounds for current rank
    tp_rank_start = tp_rank * per_rank_intermediate_size
    tp_rank_end = min((tp_rank + 1) * per_rank_intermediate_size, intermediate_size)

    block_weight_dict = {}

    for name, weight in weights:
        # Skip layers on other devices.
        if is_pp_missing_parameter(name, self):
            continue

        if ".w13_weight_scale" in name:
            # Handle MLP gate and up projection weights
            # Extract gate and up projection parts
            if use_ep:
                narrow_weight_scale = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight_scale = weight[:, 2 * tp_rank_start:2 * tp_rank_end, :]

            narrow_weight_scale = narrow_weight_scale.contiguous()

            # Read block weight
            block_name = name.replace("weight_scale", "weight")
            if block_name not in block_weight_dict:
                raise ValueError(f"Expected block weight for {block_name} not found when processing {name}")
            block_weight = block_weight_dict[block_name]
            param = params_dict[block_name]

            weight = convert_moe_packed_tensors(block_weight, narrow_weight_scale)
            param[:, :2 * (tp_rank_end - tp_rank_start), :] = weight
            del block_weight_dict[block_name]
            loaded_params.add(name)
            continue
        elif ".w13_weight" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, 2 * tp_rank_start:2 * tp_rank_end, :, :]
            narrow_weight = narrow_weight.contiguous()
            block_weight_dict[name] = narrow_weight
            loaded_params.add(name)
            continue
        elif ".w2_weight_scale" in name:
            # Handle MLP down projection weights
            if use_ep:
                narrow_weight_scale = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight_scale = weight[..., tp_rank_start // OCP_MX_BLOCK_SIZE:tp_rank_end // OCP_MX_BLOCK_SIZE]
            narrow_weight_scale = narrow_weight_scale.contiguous()

            # Read block weight
            block_name = name.replace("weight_scale", "weight")
            if block_name not in block_weight_dict:
                raise ValueError(f"Expected block weight for {block_name} not found when processing {name}")
            block_weight = block_weight_dict[block_name]
            param = params_dict[block_name]

            weight = convert_moe_packed_tensors(block_weight, narrow_weight_scale)
            param[:, :, :(tp_rank_end - tp_rank_start)] = weight
            del block_weight_dict[block_name]
            loaded_params.add(name)
            continue
        elif ".w2_weight" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, :, tp_rank_start // OCP_MX_BLOCK_SIZE:tp_rank_end // OCP_MX_BLOCK_SIZE, :]
            narrow_weight = narrow_weight.contiguous()
            block_weight_dict[name] = narrow_weight
            loaded_params.add(name)
            continue
        elif ".w13_bias" in name:
            # Handle MLP gate and up projection biases
            # Extract gate and up projection bias parts
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, 2 * tp_rank_start:2 * tp_rank_end]
            narrow_weight = narrow_weight.contiguous()

            param = params_dict[name]
            param[:, :2 * (tp_rank_end - tp_rank_start)] = narrow_weight
            loaded_params.add(name)
            continue
        elif ".w2_bias" in name:
            # Handle MLP down projection bias
            if use_ep:
                weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                # (only load on rank 0 to avoid duplication)
                if tp_rank != 0:
                    weight.zero_()
            param = params_dict[name]
            param.copy_(weight)
            loaded_params.add(name)
            continue
        elif "sinks" in name:
            # Handle attention sinks (distributed across ranks)
            param = params_dict[name]
            narrow_weight = weight.narrow(0, head_start, heads_per_rank)
            param.data.copy_(narrow_weight)
            loaded_params.add(name)
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if weight_loader == default_weight_loader:
                weight_loader(param, weight)
            else:
                weight_loader(param, weight, shard_id)
            break
        else:
            # Handle all other weights with potential renaming
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
        loaded_params.add(name)
    return loaded_params


def patched_load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    # Check if this is gpt_oss model with mxfp4 quantization
    quant_cfg = getattr(self.config, "quantization_config", None)
    quant_method = quant_cfg.get("quant_method") if quant_cfg else None

    # Only use custom loading for gpt_oss + mxfp4
    if quant_method == "mxfp4":
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        ep_size = get_ep_group().world_size
        ep_rank = get_ep_group().rank
        num_experts = self.config.num_local_experts
        experts_per_rank = num_experts // ep_size
        ep_rank_start = ep_rank * experts_per_rank
        ep_rank_end = (ep_rank + 1) * experts_per_rank

        return self._load_weights_mxfp4_dequantize_hpu(
            ep_rank_end,
            ep_rank_start,
            heads_per_rank,
            head_start,
            weights,
            stacked_params_mapping,
        )

    # For all other models, use the original vLLM implementation
    return _original_load_weights(self, weights)


# Apply monkey patches unconditionally
# The wrappers check at runtime whether to use custom logic or delegate to original
ModelArchConfigConvertorBase._normalize_quantization_config = _patched_normalize_quantization_config
GptOssModel._load_weights_mxfp4_dequantize_hpu = _load_weights_mxfp4_dequantize_hpu
GptOssModel.load_weights = patched_load_weights
