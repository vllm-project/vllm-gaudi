import math
import os
from functools import partial
from typing import Optional, Callable, Union
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm.logger import init_logger
from vllm.distributed import parallel_state

from transformers import BatchFeature
from vllm.transformers_utils.processor import (cached_image_processor_from_config)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention, Qwen2_5_VisionBlock, Qwen2_5_VisionTransformer, Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLMultiModalProcessor, Qwen2_5_VLProcessingInfo, Qwen2_5_VLVideoInputs,
    Qwen2_5_VLImageInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLImagePixelInputs, Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLProcessor)

from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm.config import MultiModalConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.model_executor.models.utils import maybe_prefix

from vllm.multimodal.inputs import MultiModalFieldConfig

import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.kernels import FusedSDPA

logger = init_logger(__name__)


class AttentionLongSequence:

    @staticmethod
    def forward(q, k, v, mask, q_block_size, softmax_mode):
        """
        Support long sequence at prompt phase
        """
        q_len = q.size(-2)
        assert q_len % q_block_size == 0
        q_tiles = (q_len // q_block_size) if (q_len % q_block_size == 0) else math.ceil(q_len / q_block_size)
        attn_output = torch.zeros_like(q)

        for i in range(q_tiles):
            s, e = i * q_block_size, (i + 1) * q_block_size
            row_q = q[:, :, s:e, :]
            row_mask = mask[:, :, s:e, :]
            attn_output[:, :, s:e, :] = FusedSDPA.apply(row_q, k, v, row_mask, 0.0, False, None, softmax_mode)
            # TODO: markstep after a couple of iterations
            # need to experiment the optimal number.
            if i % 75 == 0:
                htcore.mark_step()
        return attn_output


def create_block_diagonal_attention_mask_outerprod(indices):
    maxsize = indices[-1]
    range_to_max_for_each_img = torch.arange(maxsize,
                                             device=indices.device).unsqueeze(0).repeat(indices.shape[0] - 1, 1)
    lesser = range_to_max_for_each_img < indices[1:].unsqueeze(1)
    greater_eq = range_to_max_for_each_img >= indices[:-1].unsqueeze(1)
    range_indices = torch.logical_and(lesser, greater_eq).float()
    # can reduce sum externally or as batchmatmul
    if range_indices.shape[-1] > 40000:
        log_msg = "einsum running on CPU :" + str(range_indices.shape)
        logger.info(log_msg)
        range_indices = range_indices.to("cpu")
        res = torch.einsum('bi,bj->ij', range_indices, range_indices)
        res = res.to("hpu")
    else:
        res = torch.einsum('bi,bj->ij', range_indices, range_indices)
    return res.bool()


def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int):
    """All-gather the input tensor interleavely across model parallel group."""
    import torch.distributed as dist
    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(tp_size)]
    dist.all_gather(gathered_tensors, local_tensor, group=parallel_state.get_tp_group().device_group)

    gathered_tensors_split = [torch.split(tensor, hidden_size // tp_size, -1) for tensor in gathered_tensors]
    ordered_tensors = [tensor for pair in zip(*gathered_tensors_split) for tensor in pair]
    result_tensor = torch.cat(ordered_tensors, dim=-1)
    return result_tensor


class HPUQwen2_5_VisionAttention(Qwen2_5_VisionAttention):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            projection_size=projection_size,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=prefix,
        )

        self.softmax_mode = 'fp32' if os.environ.get('VLLM_FP32_SOFTMAX_VISION', 'false').lower() in ['true', '1'
                                                                                                      ] else 'None'
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape

        qkv = rearrange(
            x,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_attention_heads_per_partition,
        )

        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk, v = qkv[:, :, :2], qkv[:, :, 2]

            qk_reshaped = rearrange(qk, "b s two head head_dim -> (two b) s head head_dim", two=2)
            qk_rotated = self.apply_rotary_emb(qk_reshaped, rotary_pos_emb_cos, rotary_pos_emb_sin)
            qk_rotated = qk_rotated.view(
                2,
                batch_size,
                seq_len,
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            q, k = qk_rotated.unbind(dim=0)
        else:
            q, k, v = qkv.unbind(dim=2)

        fullattn_mask = cu_seqlens

        if fullattn_mask is None:  # performs window attention
            # we assume image is 112 aligned in both h/w dims
            # in other words, x % 64 = 0
            # that simplifies the slicing of window attention
            # in patches of 64
            outputs = []
            cu_seqlens = list(range(0, x.shape[0] + 1, 64))
            for i in range(1, len(cu_seqlens)):
                # For large image, we add mark step here
                # for every 100th step to make compile time shorter
                if i % 100 == 0:
                    htcore.mark_step()
                start_idx = cu_seqlens[i - 1]
                end_idx = cu_seqlens[i]
                q_i = q[:, start_idx:end_idx]
                k_i = k[:, start_idx:end_idx]
                v_i = v[:, start_idx:end_idx]
                q_i, k_i, v_i = (rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i])
                output_i = FusedSDPA.apply(q_i, k_i, v_i, None, 0.0, False, None, self.softmax_mode)
                output_i = rearrange(output_i, "b h s d -> b s h d ")
                outputs.append(output_i)
            context_layer = torch.cat(outputs, dim=1)
        else:
            # performs full attention using the previous computed mask
            fullatt_block_attn_mask = fullattn_mask
            q1, k1, v1 = (rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
            (batch_size, _, seq_len_N_t, _) = q1.shape
            (batch_size, _, seq_len_N_s, _) = k1.shape
            mask_shape = (batch_size, 1, seq_len_N_t, seq_len_N_s)
            attn_mask = fullatt_block_attn_mask.reshape(batch_size, 1, seq_len_N_t, seq_len_N_s,
                                                        -1)[:, :, :, :, 0]  # reshapes the mask to be Bx1xNxN
            assert attn_mask.shape == mask_shape

            if q1.shape[2] <= 65536:  # need to investigate this crosspoint
                fused_out = FusedSDPA.apply(q1, k1, v1, attn_mask, 0.0, False, None, self.softmax_mode)
            else:
                fused_out = AttentionLongSequence.forward(q1, k1, v1, attn_mask, 64, self.softmax_mode)
            context_layer = rearrange(fused_out, "b h s d -> b s h d ")
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class HPUQwen2_5_VisionBlock(Qwen2_5_VisionBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            act_fn=act_fn,
            norm_layer=norm_layer,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=prefix,
        )
        self.attn = HPUQwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=maybe_prefix(prefix, "attn."),
        )

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb_cos: torch.Tensor,
            rotary_pos_emb_sin: torch.Tensor,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          rotary_pos_emb_cos=rotary_pos_emb_cos,
                          rotary_pos_emb_sin=rotary_pos_emb_sin)

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionTransformerStaticShape(Qwen2_5_VisionTransformer):
    """
    Here we overwrite some of the methods of Qwen2_5_VisionTransformer
    to make the model more friendly to static shapes. Specifically,
    we split the forward  method into:
      - pre_attn (dynamic)
      - forward (static shape)
      - post_attn (dynamic)
    and we should call get_image_embeds instead of forward, allowing
    the forward method to run with HPU_Graphs, whereas the
    pre_attn and post_attn methods are allow to be dynamic.
    """

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=prefix,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        depth = vision_config.depth
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("Qwen2_5_VisionBlock"):
            self.blocks = nn.ModuleList([
                HPUQwen2_5_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=get_act_and_mul_fn(vision_config.hidden_act),
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                ) for layer_idx in range(depth)
            ])

    def rot_pos_emb(self, grid_thw: torch.Tensor):  # -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        cos_combined = cos_combined.reshape(
            cos_combined.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        sin_combined = sin_combined.reshape(
            sin_combined.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )

        return cos_combined, sin_combined

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (self.window_size // self.spatial_merge_size // self.patch_size)

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = \
                vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = \
                vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
            index_padded = index_padded.reshape(grid_t, num_windows_h, vit_merger_window_size, num_windows_w,
                                                vit_merger_window_size)
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(grid_t, num_windows_h * num_windows_w,
                                                                       vit_merger_window_size, vit_merger_window_size)
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]

            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]

            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index.append(index_new + window_index_id)
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def pre_attn(self, x: torch.Tensor, grid_thw: torch.Tensor):
        # patchify
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        # compute position embedding
        cos_combined, sin_combined = self.rot_pos_emb(grid_thw)

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)

        # NOTE: unique_consecutive is a dynamic operation
        # we are using `remove_duplicates_cpu` instead
        def remove_duplicates_cpu(a):
            return [a[i] for i in range(len(a)) if i == 0 or a[i - 1] != a[i]]

        cu_window_seqlens = remove_duplicates_cpu(cu_window_seqlens)
        cu_window_seqlens = torch.tensor(cu_window_seqlens,
                                         device=hidden_states.device,
                                         dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        cos_combined = cos_combined[window_index, :, :]
        cos_combined = cos_combined.flatten(start_dim=0, end_dim=1)
        sin_combined = sin_combined[window_index, :, :]
        sin_combined = sin_combined.flatten(start_dim=0, end_dim=1)
        cos_combined = cos_combined.to(device=self.device, non_blocking=True)
        sin_combined = sin_combined.to(device=self.device, non_blocking=True)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0,
                                                                                                     dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        return (
            hidden_states,
            cos_combined,
            sin_combined,
            cu_seqlens,
            cu_window_seqlens,
            window_index,
        )

    def forward(self, x: torch.Tensor, fullattn_mask: Optional[torch.Tensor], rotary_pos_emb_cos: torch.Tensor,
                rotary_pos_emb_sin: torch.Tensor) -> torch.Tensor:
        assert_msg = ("Expect inputs to be 112x112 aligned. "
                      "Please align before sending image and "
                      "check PR #1163 description for more details")
        assert x.shape[0] % 64 == 0, assert_msg
        hidden_states = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            htcore.mark_step()
            hidden_states = blk(
                hidden_states,
                cu_seqlens=fullattn_mask if layer_num in self.fullatt_block_indexes else None,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )
        return hidden_states

    def post_attn(self, hidden_states: torch.Tensor, window_index: torch.Tensor):
        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)

        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def pad_multimodal_data(self, pixel_values, image_grid_thw, vision_buckets):
        assert pixel_values.shape[0] % 64 == 0, 'needs 64 aligned resolution'

        desired_number_of_pixels = vision_buckets.get_multimodal_bucket(pixel_values.shape[0])
        padding_len = desired_number_of_pixels - pixel_values.shape[0]
        if padding_len <= 0:
            return pixel_values, image_grid_thw

        logger_msg = "Padding current number pixel " \
            + str(pixel_values.shape[0]) \
            + " to " \
            + str(desired_number_of_pixels)
        logger.info(logger_msg)

        assert padding_len % 64 == 0, 'padding needs to be multiple of 64'

        constant_value = -100
        pixel_values = torch.cat([
            pixel_values,
            torch.ones((padding_len, pixel_values.shape[1]), device=pixel_values.device) * constant_value
        ])

        image_grid_thw = torch.cat(
            [image_grid_thw, torch.tensor([[1, 8, padding_len // 8]], device=image_grid_thw.device)])

        assert image_grid_thw.prod(-1).sum() == desired_number_of_pixels
        return pixel_values, image_grid_thw

    def get_image_embeds(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        vision_buckets,
    ) -> torch.Tensor:

        num_patches = pixel_values.shape[0]
        if num_patches % 64 != 0:
            assert num_patches > 64, "Image needs to be at least 112 x 112"
            logger_msg = ("QWEN 2.5VL for HPU is under development. "
                          "Image height and width need to be multiples of 112 pixels. "
                          "We are prunning the last visual tokens to comply with this "
                          "requirement but this leads to accuracy degradation. "
                          "Please, reshape the images or use this custom transformer "
                          "that does the resizing/alignment automatically: "
                          "pip install "
                          "git+https://github.com/malkomes/transformers.git"
                          "@ac372cd18f836c41f57cdce46094db00019d4280"
                          "See PR #1163 description, for more details")
            logger.warning_once(logger_msg)

            # reshape grid_thw with multiples of 8
            old_img_sizes = []
            new_img_sizes = []
            for img_idx in range(grid_thw.shape[0]):
                img_shape = grid_thw[img_idx, :].tolist()
                tt, hh, ww = img_shape
                hh_new = (hh // 8) * 8
                ww_new = (ww // 8) * 8
                old_img_sizes.append(tt * hh * ww)
                new_img_sizes.append(tt * hh_new * ww_new)
                grid_thw[img_idx, 1] = hh_new
                grid_thw[img_idx, 2] = ww_new

            # truncate pixel_values to new shapes
            copy_pointer = 0
            paste_pointer = 0
            for old_img_size, new_img_size in zip(old_img_sizes, new_img_sizes):
                pixel_values[paste_pointer:paste_pointer + new_img_size, :] = \
                    pixel_values[copy_pointer:copy_pointer + new_img_size, :]
                copy_pointer += old_img_size
                paste_pointer += new_img_size

            pixel_values = pixel_values[:paste_pointer, :]

        offset = 0
        results = []
        # process each image one by one
        for img_idx in range(grid_thw.shape[0]):
            img_shape = grid_thw[img_idx, :].unsqueeze(0)
            curr_img_size = img_shape.prod()

            pixel_values_curr_img = pixel_values[offset:offset + curr_img_size, :]

            offset += curr_img_size
            pixel_values_curr_img_padded, img_shape_padded = \
                self.pad_multimodal_data(
                    pixel_values_curr_img,
                    img_shape,
                    vision_buckets=vision_buckets
                )

            pixel_values_curr_img_padded, rot_pos_emb_cos, rot_pos_emb_sin, \
                cu_seqlens, cu_window_seqlens, window_index = self.pre_attn(
                    pixel_values_curr_img_padded, img_shape_padded)

            # Create full attention block mask before VisionTransformer
            # to save memory/time
            fullatt_block_attn_mask = \
                create_block_diagonal_attention_mask_outerprod(cu_seqlens)

            assert pixel_values_curr_img_padded.shape[0] == cu_seqlens[-1]
            assert pixel_values_curr_img_padded.shape[0] == rot_pos_emb_cos.shape[0] == rot_pos_emb_sin.shape[0]

            htcore.mark_step()
            hidden_states = self.forward(
                pixel_values_curr_img_padded,
                rotary_pos_emb_cos=rot_pos_emb_cos,
                rotary_pos_emb_sin=rot_pos_emb_sin,
                fullattn_mask=fullatt_block_attn_mask,
            )
            htcore.mark_step()

            image_embeds = self.post_attn(hidden_states, window_index)
            # slice image_embeds to remove the padded parts
            pad_index = img_shape_padded[0].prod() // self.spatial_merge_unit
            results += [image_embeds[:pad_index, :]]
        results_cat = torch.concat(results)
        image_embeds = results_cat
        return image_embeds


class HPUQwen2_5_VLProcessingInfo(Qwen2_5_VLProcessingInfo):

    def get_hf_processor(
        self,
        *,
        fps: Optional[Union[float, list[float]]] = None,
        **kwargs: object,
    ) -> Qwen2_5_VLProcessor:
        if fps is not None:
            kwargs["fps"] = fps

        min_pixels = 112 * 112
        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,
            image_processor=cached_image_processor_from_config(
                self.ctx.model_config,
                min_pixels=min_pixels,
            ),
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class HPUQwen2_5_VLMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            **super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video"),
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=HPUQwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class HpuQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        if hasattr(self, "visual") and self.visual is not None:
            self.visual = Qwen2_5_VisionTransformerStaticShape(
                self.config.vision_config,
                norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

    def _parse_and_validate_image_input_v1(self, **kwargs: object) -> Optional[Qwen2_5_VLImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(pixel_values, "image pixel values")
            image_grid_thw = self._validate_and_reshape_mm_tensor(image_grid_thw, "image grid_thw")

            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image pixel values. "
                                 f"Got type: {type(pixel_values)}")

            return Qwen2_5_VLImagePixelInputs(type="pixel_values",
                                              pixel_values=pixel_values,
                                              image_grid_thw=image_grid_thw)

        if image_embeds is not None:
            image_embeds = self._validate_and_reshape_mm_tensor(image_embeds, "image embeds")
            image_grid_thw = self._validate_and_reshape_mm_tensor(image_grid_thw, "image grid_thw")

            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")
            return Qwen2_5_VLImageEmbeddingInputs(type="image_embeds",
                                                  image_embeds=image_embeds,
                                                  image_grid_thw=image_grid_thw)

    def _parse_and_validate_video_input_v1(self, **kwargs: object) -> Optional[Qwen2_5_VLVideoInputs]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:

            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        if video_embeds is not None:
            video_embeds = self._validate_and_reshape_mm_tensor(video_embeds, "video embeds")
            video_grid_thw = self._validate_and_reshape_mm_tensor(video_grid_thw, "video grid_thw")

            if not isinstance(video_embeds, torch.Tensor):
                raise ValueError("Incorrect type of video embeddings. "
                                 f"Got type: {type(video_embeds)}")
            return Qwen2_5_VLVideoEmbeddingInputs(type="video_embeds",
                                                  video_embeds=video_embeds,
                                                  video_grid_thw=video_grid_thw)

    def _process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)

            image_embeds = self.visual.get_image_embeds(
                pixel_values,
                grid_thw=grid_thw,
                vision_buckets=self.vision_bucket_manager,
            )

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return image_embeds.split(sizes.tolist())

    def _process_video_input(self, video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            video_embeds = self.visual.get_image_embeds(
                pixel_values_videos,
                grid_thw=grid_thw,
                vision_buckets=self.vision_bucket_manager,
            )

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size

        return video_embeds.split(sizes.tolist())
