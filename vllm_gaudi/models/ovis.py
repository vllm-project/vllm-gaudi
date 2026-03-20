# SPDX-License-Identifier: Apache-2.0

from vllm.config import VllmConfig
from vllm.model_executor.models.ovis import (Ovis, OvisMultiModalProcessor, OvisProcessingInfo, OvisDummyInputsBuilder)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(OvisMultiModalProcessor,
                                        info=OvisProcessingInfo,
                                        dummy_inputs=OvisDummyInputsBuilder)
class HpuOvis(Ovis):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
