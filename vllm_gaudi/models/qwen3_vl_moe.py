from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from vllm_gaudi.models.qwen3_moe import upgrade_qwen3_moe_blocks_inplace


class HpuQwen3_VLMoeForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        quant_config = getattr(self, "quant_config", None)

        if hasattr(self, "visual") and self.visual is not None:
            self.visual = HPUQwen3_VisionTransformer(
                self.config.vision_config,
                norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        # qwen3 moe mlp blocks: make forward for 3d safe (b,s,h -> t,h)
        lm = getattr(self, "language_model", None)
        if lm is not None:
            _n = upgrade_qwen3_moe_blocks_inplace(lm)
