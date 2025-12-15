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
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig, )

from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention, Qwen2_5_VisionBlock, Qwen2_5_VisionTransformer, Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLMultiModalProcessor, Qwen2_5_VLProcessingInfo, Qwen2_5_VLVideoInputs,
    Qwen2_5_VLImageInputs, Qwen2_5_VLImageEmbeddingInputs, Qwen2_5_VLImagePixelInputs, Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLProcessor)

from vllm.model_executor.models.qwen2_vl import (apply_rotary_pos_emb_vision)

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm.config import MultiModalConfig, VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.model_executor.models.utils import (maybe_prefix, cast_overflow_tensors)

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
        q_tiles = (q_len // q_block_size)
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


class HPU_Attention:

    softmax_mode = 'fp32' if \
        os.environ.get('VLLM_FP32_SOFTMAX_VISION', 'false').lower() \
            in ['true', '1'] else 'None'

    @classmethod
    def forward(cls, q, k, v, mask, q_block_size=64):
        """
        Support long sequence at prompt phase
        """
        q_len = q.size(-2)
        if q_len <= 65536:  # need to investigate this crosspoint
            return FusedSDPA.apply(q, k, v, mask, 0.0, False, None, cls.softmax_mode)

        assert q_len % q_block_size == 0
        q_tiles = (q_len // q_block_size) if (q_len % q_block_size == 0) else math.ceil(q_len / q_block_size)
        attn_output = torch.zeros_like(q)

        for i in range(q_tiles):
            s, e = i * q_block_size, (i + 1) * q_block_size
            row_q = q[:, :, s:e, :]
            row_mask = mask[:, :, s:e, :]
            attn_output[:, :, s:e, :] = FusedSDPA.apply(row_q, k, v, row_mask, 0.0, False, None, cls.softmax_mode)
            # TODO: markstep after a couple of iterations
            # need to experiment the optimal number.
            if i % 75 == 0:
                htcore.mark_step()
        return attn_output


def create_block_diagonal_attention_mask(indices):
    max_size = indices[-1]
    range_to_max_for_each_img = torch.arange(max_size,
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

    def forward(
            self,
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb_cos: torch.Tensor,
            rotary_pos_emb_sin: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,  # Only used for HPU
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
            qk_rotated = apply_rotary_pos_emb_vision(qk_reshaped, cos=rotary_pos_emb_cos, sin=rotary_pos_emb_sin)
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

        # performs full attention using the previous computed mask
        q1, k1, v1 = (rearrange(x, "b s h d -> b h s d") for x in [q, k, v])
        output = HPU_Attention.forward(q1, k1, v1, attn_mask)
        context_layer = rearrange(output, "b h s d -> b s h d ")

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
            rotary_pos_emb_cos: torch.Tensor,
            rotary_pos_emb_sin: torch.Tensor,
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
            seqlens: Optional[list[int]] = None,  # Only used for xFormers
            attn_mask: Optional[torch.Tensor] = None,  # Only used for HPU
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x),
                          cu_seqlens=cu_seqlens,
                          rotary_pos_emb_cos=rotary_pos_emb_cos,
                          rotary_pos_emb_sin=rotary_pos_emb_sin,
                          attn_mask=attn_mask)

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

    def pre_attn(self, x: torch.Tensor, grid_thw: torch.Tensor):
        # patchify
        seq_len, _ = x.size()
        cos_list = []
        sin_list = []
        window_index: list = []
        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens: list = []

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                cos_thw,
                sin_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += (t * llm_h * llm_w)

            cu_seqlens_window_thw = (cu_seqlens_window_thw + cu_window_seqlens_last)
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            # accumulate RoPE and THW seqlens
            cos_list.append(cos_thw)
            sin_list.append(sin_thw)
            cu_seqlens.append(cu_seqlens_thw)

        # concatenate
        cos_combined = torch.cat(cos_list).to(self.device, non_blocking=True)
        sin_combined = torch.cat(sin_list).to(self.device, non_blocking=True)

        window_index = torch.cat(window_index).to(self.device, non_blocking=True)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=self.device, non_blocking=True)

        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        return (
            hidden_states,
            cos_combined,
            sin_combined,
            cu_seqlens,
            cu_window_seqlens,
            window_index,
        )

    def forward(self, hidden_states: torch.Tensor, rotary_pos_emb_cos: torch.Tensor, rotary_pos_emb_sin: torch.Tensor,
                padding_attn_mask_window: torch.Tensor, padding_attn_mask_full: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                padding_attn_mask_now = padding_attn_mask_full
            else:
                padding_attn_mask_now = padding_attn_mask_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=padding_attn_mask_now,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        # For Qwen2.5-VL-3B, float16 will overflow at last block
        # for long visual tokens sequences.
        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states

    def post_attn(self, hidden_states: torch.Tensor, window_index: torch.Tensor):
        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)

        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

    def get_image_embeds(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        vision_buckets,
    ) -> torch.Tensor:
        seq_len, _ = pixel_values.size()
        offset = 0
        results = []
        # process each image one by one
        for img_idx in range(grid_thw.shape[0]):
            img_shape = grid_thw[img_idx, :].unsqueeze(0)
            curr_img_size = img_shape.prod()

            pixel_values_curr_img = pixel_values[offset:offset + curr_img_size, :]

            offset += curr_img_size
            # pre-attention block
            hidden_states, rot_pos_emb_cos, rot_pos_emb_sin, \
                cu_seqlens, cu_window_seqlens, window_index = self.pre_attn(
                    pixel_values_curr_img, img_shape)

            # add padding
            bucket_size = vision_buckets.get_multimodal_bucket(curr_img_size)
            num_pad_tokens = bucket_size - curr_img_size
            if num_pad_tokens > 0:
                logger_msg = "Padding current image size " \
                    + str(curr_img_size.item()) \
                    + " to " \
                    + str(bucket_size)
                logger.info(logger_msg)
                cu_seqlens = F.pad(cu_seqlens, (0, 1), "constant", bucket_size)
                cu_window_seqlens = F.pad(cu_window_seqlens, (0, 1), "constant", bucket_size)
                hidden_states = F.pad(hidden_states, (0, 0, 0, num_pad_tokens), "constant", -100)
                rot_pos_emb_cos = F.pad(
                    rot_pos_emb_cos,  # [seq, dim]
                    (0, 0, 0, num_pad_tokens),
                    "constant",
                    0.0)
            rot_pos_emb_sin = F.pad(rot_pos_emb_sin, (0, 0, 0, num_pad_tokens), "constant", 0.0)

            padding_attn_mask_full = create_block_diagonal_attention_mask(cu_seqlens)
            padding_attn_mask_window = create_block_diagonal_attention_mask(cu_window_seqlens)

            # static part
            htcore.mark_step()
            hidden_states = self.forward(hidden_states,
                                         rotary_pos_emb_cos=rot_pos_emb_cos,
                                         rotary_pos_emb_sin=rot_pos_emb_sin,
                                         padding_attn_mask_window=padding_attn_mask_window,
                                         padding_attn_mask_full=padding_attn_mask_full)
            htcore.mark_step()

            # remove padding
            hidden_states = hidden_states[:curr_img_size, :, :]

            # after attention
            image_embeds = self.post_attn(hidden_states, window_index)
            results += [image_embeds]
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
