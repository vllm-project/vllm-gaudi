from torch import nn
from vllm.model_executor.layers.activation import get_act_fn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
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
        prefix: str = "",
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            act_fn=act_fn,
            norm_layer=norm_layer,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.attn = HPUQwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )


class HPUQwen3_VisionTransformer(Qwen3_VisionTransformer):

    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__(
            vision_config=vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
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
                prefix=f"{prefix}.blocks.{layer_idx}",
            ) for layer_idx in range(depth)
        ])


class HpuQwen3_VLForConditionalGeneration(Qwen3VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        if hasattr(self, "visual") and self.visual is not None:
            self.visual = HPUQwen3_VisionTransformer(
                self.config.vision_config,
                norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
                prefix=maybe_prefix(prefix, "visual"),
            )
