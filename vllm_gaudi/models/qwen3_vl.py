import torch
from .utils import _merge_multimodal_embeddings
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.interfaces import _require_is_multimodal
from torch import nn
from vllm.model_executor.layers.activation import get_act_fn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3_VisionBlock,
)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_gaudi.models.qwen2_5_vl import (HPUQwen2_5_VisionAttention)


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


class HPUQwen3_VisionTransformerStaticShape(Qwen3_VisionTransformer):

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
        norm_layer = lambda d: nn.LayerNorm(d, eps=norm_eps)

        self.blocks = nn.ModuleList([
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


class HpuQwen3_VLForConditionalGeneration(Qwen3VLForConditionalGeneration):

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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        quant_config = getattr(self, "quant_config", None)
        multimodal_config = getattr(vllm_config.model_config, "multimodal_config", None)

        if hasattr(self, "visual") and self.visual is not None:
            self.visual = HPUQwen3_VisionTransformerStaticShape(
                self.config.vision_config,
                norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
