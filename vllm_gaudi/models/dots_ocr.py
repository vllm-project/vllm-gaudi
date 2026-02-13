from vllm.config import VllmConfig
from vllm.model_executor.models.dots_ocr import (DotsOCRForCausalLM, DotsOCRProcessingInfo, DotsOCRDummyInputsBuilder)
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2VLMultiModalProcessor,
    info=DotsOCRProcessingInfo,
    dummy_inputs=DotsOCRDummyInputsBuilder,
)
class HpuDotsOCRForCausalLM(DotsOCRForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
