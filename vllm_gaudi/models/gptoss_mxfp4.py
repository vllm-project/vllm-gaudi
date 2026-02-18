from typing import Iterable

import torch
from transformers import PretrainedConfig
from vllm.model_executor.models.gpt_oss import GptOssModel
from vllm.utils.math_utils import cdiv
from vllm.transformers_utils.model_arch_config_convertor import ModelArchConfigConvertorBase
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.platforms import current_platform

def _normalize_quantization_config(self, config: PretrainedConfig):

    # Skip mxfp4 quantization to use custom loading logic
    if getattr(config, "model_type", None) == "gpt_oss":
        quant_cfg = getattr(config, "quantization_config", None)
        if quant_cfg is not None and quant_cfg.get("quant_method", "").lower() == "mxfp4":
            return None

    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg is None:
        # compressed-tensors uses a "compression_config" key
        quant_cfg = getattr(config, "compression_config", None)

    else:
        # Set quant_method for ModelOpt models.
        producer_name = quant_cfg.get("producer", {}).get("name")
        if producer_name == "modelopt":
            quant_algo = quant_cfg.get("quantization", {}).get("quant_algo")
            if quant_algo is not None:
                quant_algo_upper = str(quant_algo).upper()
                if quant_algo_upper in {
                    "FP8",
                    "FP8_PER_CHANNEL_PER_TOKEN",
                    "FP8_PB_WO",
                }:
                    quant_cfg["quant_method"] = "modelopt"
                elif quant_algo_upper == "NVFP4":
                    quant_cfg["quant_method"] = "modelopt_fp4"
                else:
                    raise ValueError(f"Unknown ModelOpt quant algo: {quant_algo}")

    if quant_cfg is not None:
        # Use the community standard 'quant_method'
        quant_method = quant_cfg.get("quant_method", "").lower()

        # Normalize library names
        quant_method = quant_method.replace(
            "compressed_tensors", "compressed-tensors"
        )

        quant_cfg["quant_method"] = quant_method

    return quant_cfg

def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,  # Large default chosen to process many rows per kernel launch and reduce overhead; lower this if you need to limit peak memory usage.
) -> torch.Tensor:
    """
    Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
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

    scales = scales.to(torch.int32) - 127  # TODO that's because 128=2**7

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

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    del blocks, scales, lut
    return out.transpose(1, 2).contiguous()


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

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    mxfp4_block = 32
    intermediate_size = self.config.intermediate_size
    intermediate_size_block = intermediate_size // mxfp4_block
    per_rank_intermediate_size_block = cdiv(intermediate_size_block, tp_size)
    per_rank_intermediate_size = per_rank_intermediate_size_block * mxfp4_block

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
                narrow_weight_scale = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, :]

            narrow_weight_scale = narrow_weight_scale.contiguous()

            # Read block weight
            block_name = name.replace("weight_scale", "weight")
            block_weight = block_weight_dict[block_name]
            param = params_dict[block_name]

            weight = convert_moe_packed_tensors(block_weight, narrow_weight_scale).permute(0, 2, 1).contiguous()
            param[:, :2 * (tp_rank_end - tp_rank_start), :] = weight
            loaded_params.add(name)
            continue
        elif ".w13_weight" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end, :, :]

            narrow_weight = narrow_weight.contiguous()
            block_weight_dict[name] = narrow_weight
            loaded_params.add(name)
            continue
        elif ".w2_weight_scale" in name:
            # Handle MLP down projection weights
            if use_ep:
                narrow_weight_scale = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight_scale = weight[..., tp_rank_start//mxfp4_block:tp_rank_end//mxfp4_block]
            narrow_weight_scale = narrow_weight_scale.contiguous()

            # Read block weight
            block_name = name.replace("weight_scale", "weight")
            block_weight = block_weight_dict[block_name]
            param = params_dict[block_name]

            weight = convert_moe_packed_tensors(block_weight, narrow_weight_scale).permute(0, 2, 1).contiguous()
            param[:, :, :(tp_rank_end - tp_rank_start)] = weight
            loaded_params.add(name)
            continue
        elif ".w2_weight" in name:
            if use_ep:
                narrow_weight = weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = weight[:, :, tp_rank_start//mxfp4_block:tp_rank_end//mxfp4_block, :]
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
                narrow_weight = weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
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

def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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

    quant_method = (
        self.config.quantization_config["quant_method"]
        if hasattr(self.config, "quantization_config")
        else None
    )
    if quant_method == "mxfp4":
        if current_platform.device_name == "hpu":
            return self._load_weights_mxfp4_dequantize_hpu(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )
        else:
            return self._load_weights_mxfp4(
                ep_rank_end,
                ep_rank_start,
                heads_per_rank,
                head_start,
                weights,
                stacked_params_mapping,
            )
    else:
        return self._load_weights_other(
            ep_rank_end,
            ep_rank_start,
            heads_per_rank,
            head_start,
            weights,
            stacked_params_mapping,
        )

ModelArchConfigConvertorBase._normalize_quantization_config = _normalize_quantization_config
GptOssModel.load_weights = load_weights
GptOssModel._load_weights_mxfp4_dequantize_hpu = _load_weights_mxfp4_dequantize_hpu