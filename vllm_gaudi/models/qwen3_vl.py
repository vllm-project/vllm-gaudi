import torch
import numpy as np
from .utils import _merge_multimodal_embeddings
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.interfaces import _require_is_multimodal

from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLImageInputs, )
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3_VisionTransformer,
    Qwen3_VisionBlock,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model

from vllm.model_executor.models.utils import maybe_prefix

from vllm_gaudi.models.qwen2_5_vl import HPUQwen2_5_VisionAttention


class HPUQwen3_VisionBlock(Qwen3_VisionBlock):

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn,
        norm_layer,
        quant_config=None,
        multimodal_config=None,
        prefix: str = "",
    ):
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
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        attn_mask=None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            attn_mask=attn_mask,
            max_seqlen=max_seqlen,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class HPUQwen3_VisionTransformer(Qwen3_VisionTransformer):

    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config=None,
        multimodal_config=None,
        prefix: str = "",
    ):
        super().__init__(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=prefix,
        )

        depth = vision_config.depth
        norm_layer = lambda d: torch.nn.LayerNorm(d, eps=norm_eps)

        self.blocks = torch.nn.ModuleList([
            HPUQwen3_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=get_act_fn(vision_config.hidden_act),
                norm_layer=norm_layer,
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
            ) for layer_idx in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.numpy()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        cu_seqlens = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(axis=0, dtype=np.int32)
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        cu_seqlens = torch.from_numpy(cu_seqlens)
        hidden_states = hidden_states.unsqueeze(1)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
                attn_mask=attn_mask,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat([hidden_states] + deepstack_feature_lists,
                                  dim=1)  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states


class HpuQwen3_VLForConditionalGeneration(Qwen3VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        quant_config = getattr(self, "quant_config", None)
        multimodal_config = getattr(vllm_config.model_config, "multimodal_config", None)

        if hasattr(self, "visual") and self.visual is not None:
            self.visual = HPUQwen3_VisionTransformer(
                self.config.vision_config,
                norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

    def create_block_diagonal_mask(self,
                                   cu_seqlens: torch.Tensor,
                                   grid_thw: list[int],
                                   device: torch.device = None,
                                   dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """
        Create block diagonal mask that excludes padded tokens for Qwen3VL attention.
        Args:
            cu_seqlens: Cumulative sequence lengths from grid dimensions
            grid_thw: The grid dimensions with merge_size=2 compatibility
            device: Target device for the mask
            dtype: Data type for the mask (typically torch.bool)

        Returns:
            Block diagonal attention mask with shape [total_seq_len, total_seq_len]
        """
        if device is None:
            device = cu_seqlens.device

        # Calculate total sequence length including padding
        total_patches = int(grid_thw.prod(-1).sum().item())
        # Create mask with total size including padding
        mask = torch.zeros(total_patches, total_patches, device=device, dtype=dtype)
        cu_seqlens = cu_seqlens.tolist()
        cu_seqlens = [0] + cu_seqlens
        starts = cu_seqlens[:-1]
        ends = cu_seqlens[1:]
        for start, end in zip(starts, ends):
            mask[start:end, start:end] = True
        return mask

    def _process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(self.visual,
                                                         pixel_values,
                                                         grid_thw.tolist(),
                                                         rope_type="rope_3d")
            else:
                SPLIT_THRESHOLD = 20480
                if pixel_values.shape[0] < SPLIT_THRESHOLD and grid_thw.shape[0] > 1:
                    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                         grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
                    #pixel_values, grid_thw = \
                    #    self.vision_bucket_manager.pad_multimodal_data(pixel_values, grid_thw)
                    attn_mask = self.create_block_diagonal_mask(cu_seqlens, grid_thw, pixel_values.device)
                else:
                    attn_mask = None
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw, attn_mask=attn_mask)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _compute_deepstack_embeds(
        self,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: torch.Tensor,
    ) -> tuple[torch.Tensor, MultiModalEmbeddings]:
        visual_lens = [len(x) for x in multimodal_embeddings]
        multimodal_embeddings_cat = torch.cat(multimodal_embeddings, dim=0)

        (
            multimodal_embeddings_main,
            multimodal_embeddings_multiscale,
        ) = torch.split(
            multimodal_embeddings_cat,
            [self.visual_dim, self.multiscale_dim],
            dim=-1,
        )

        multimodal_embeddings = torch.split(multimodal_embeddings_main, visual_lens, dim=0)
        multimodal_embeddings_multiscale = torch.split(multimodal_embeddings_multiscale, visual_lens, dim=0)

        deepstack_input_embeds = inputs_embeds.new_zeros(inputs_embeds.size(0),
                                                         self.deepstack_num_level * inputs_embeds.size(1))

        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multimodal_embeddings_multiscale,
            is_multimodal=is_multimodal,
        )
        deepstack_input_embeds = deepstack_input_embeds.view(inputs_embeds.shape[0], self.deepstack_num_level,
                                                             self.visual_dim)
        deepstack_input_embeds = deepstack_input_embeds.permute(1, 0, 2)

        return deepstack_input_embeds, multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        if self.use_deepstack:
            (
                deepstack_input_embeds,
                multimodal_embeddings,
            ) = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        return inputs_embeds
