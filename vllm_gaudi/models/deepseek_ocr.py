# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping

import torch

from vllm.distributed import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.model_executor.models.deepseek_ocr import (
    DeepseekOCRForCausalLM,
    DeepseekOCRMultiModalProcessor,
    DeepseekOCRProcessingInfo,
    DeepseekOCRDummyInputsBuilder,
)


class HpuDeepseekOCRDummyInputsBuilder(DeepseekOCRDummyInputsBuilder):

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None

        # All possible deepseek ocr input is always
        #    pixel_values: torch.Size([1, 3, 1024, 1024])
        #    images_crop: torch.Size([sub_image_num, 3, 640, 640])
        #    images_spatial_crop: torch.Size([1, 2])
        # Where sub_image_num is in [0, 2, 3, 4, 5, 6], decided by the
        # resolution and the aspect ratio.
        # The follow code can imitate all the possible sub_image_num.
        if image_overrides is not None and image_overrides.width is not None:
            if image_overrides.width > 1600:
                image_overrides.width = 1280
                image_overrides.height = 120
            elif image_overrides.width > 1280:
                image_overrides.width = 1280
                image_overrides.height = 260
            else:
                image_overrides.height = 300

        return {
            "image":
            self._get_dummy_images(
                num_images=num_images,
                width=max_image_size.width,
                height=max_image_size.height,
                overrides=image_overrides,
            )
        }


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCRMultiModalProcessor,
    info=DeepseekOCRProcessingInfo,
    dummy_inputs=HpuDeepseekOCRDummyInputsBuilder,
)
class HpuDeepseekOCRForCausalLM(DeepseekOCRForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        # The deepseek model only accepts (token, embed) tensor type,
        # We need to flatten (batch, token, embed) into (batch*token, embed)
        # here. The ForwardContext hold the current batch info and
        # HPUAttentionImpl will view it as (batch, token, embed) again.
        if get_pp_group().is_first_rank and inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1))

        hidden_states = self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states
