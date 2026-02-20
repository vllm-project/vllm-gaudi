from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from vllm_gaudi.models.qwen3_moe import upgrade_qwen3_moe_blocks_inplace


class HpuQwen3_VLMoeForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # qwen3 moe mlp blocks: make forward for 3d safe (b,s,h -> t,h)
        lm = getattr(self, "language_model", None)
        if lm is not None:
            _n = upgrade_qwen3_moe_blocks_inplace(lm)
