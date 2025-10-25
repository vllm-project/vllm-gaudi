import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma3_mm import (Gemma3ForConditionalGeneration, Gemma3MultiModalProcessor,
                                                  Gemma3ProcessingInfo, Gemma3DummyInputsBuilder, Gemma3ImageInputs)
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(Gemma3MultiModalProcessor,
                                        info=Gemma3ProcessingInfo,
                                        dummy_inputs=Gemma3DummyInputsBuilder)
class HpuGemma3ForConditionalGeneration(Gemma3ForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    # For HPU optimization, process the vision tower using buckets to reduce recipe recompilation overhead
    def _process_image_input(self, image_input: Gemma3ImageInputs) -> list[torch.Tensor]:
        assert self.vision_tower is not None
        pixel_values = image_input["pixel_values"]
        num_patches = image_input["num_patches"]

        if hasattr(self, 'vision_bucket_manager'):
            batch_breakdown = self.vision_bucket_manager.greedy_plan(pixel_values.shape[0],
                                                                     self.vision_bucket_manager.multimodal_buckets)
            start_idx = 0
            image_embeds_multibatches = []

            for i in batch_breakdown:
                end_idx = start_idx + i
                indices = torch.arange(start_idx, end_idx).to(pixel_values.device)
                batch_sliced_pixel_values = torch.index_select(pixel_values, dim=0, index=indices)

                image_features = self._image_pixels_to_features(
                    self.vision_tower,
                    batch_sliced_pixel_values,
                )
                image_embeds = self.multi_modal_projector(image_features)
                image_embeds_multibatches += [image_embeds.clone()]
                start_idx = end_idx
            image_embeds = torch.cat(image_embeds_multibatches, dim=0)
        else:
            image_features = self._image_pixels_to_features(
                self.vision_tower,
                pixel_values,
            )
            image_embeds = self.multi_modal_projector(image_features)

        return [e.flatten(0, 1) for e in image_embeds.split(num_patches.tolist())]
