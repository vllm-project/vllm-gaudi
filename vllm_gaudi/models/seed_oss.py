from vllm.config import VllmConfig
from vllm.model_executor.models.seed_oss import SeedOssForCausalLM


class HpuSeedOssForCausalLM(SeedOssForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
