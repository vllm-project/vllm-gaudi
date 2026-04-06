# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4_mm import (Gemma4ForConditionalGeneration, Gemma4MultiModalProcessor,
                                                   Gemma4ProcessingInfo, Gemma4DummyInputsBuilder,
                                                   Gemma4ImageInputs)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(Gemma4MultiModalProcessor,
                                        info=Gemma4ProcessingInfo,
                                        dummy_inputs=Gemma4DummyInputsBuilder)
class HpuGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    # For HPU optimization, process images using buckets to reduce recipe recompilation overhead
    def _process_image_input(self, image_input: Gemma4ImageInputs) -> list[torch.Tensor]:
        assert self.vision_tower is not None
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]

        vt = self.vision_tower
        pooling_k2 = self.config.vision_config.pooling_kernel_size**2
        target_dtype = self.embed_vision.embedding_projection.weight.dtype

        if hasattr(self, 'vision_bucket_manager'):
            batch_breakdown = self.vision_bucket_manager.greedy_plan(pixel_values.shape[0],
                                                                     self.vision_bucket_manager.multimodal_buckets)
            start_idx = 0
            all_features = []

            for batch_size in batch_breakdown:
                end_idx = start_idx + batch_size
                indices = torch.arange(start_idx, end_idx).to(pixel_values.device)
                batch_pv = torch.index_select(pixel_values, dim=0, index=indices)
                batch_pp = torch.index_select(pixel_position_ids, dim=0, index=indices)

                for i in range(batch_pv.shape[0]):
                    pv = batch_pv[i].unsqueeze(0)
                    pp = batch_pp[i].unsqueeze(0)
                    max_patches = pv.shape[1]
                    output_length = max_patches // pooling_k2

                    vt_output = vt(pv, pp, output_length=output_length)
                    feat = self.embed_vision(
                        inputs_embeds=vt_output.last_hidden_state.unsqueeze(0).to(target_dtype)).squeeze(0)
                    all_features.append(feat)

                start_idx = end_idx

            return all_features
        else:
            return super()._process_image_input(image_input)
