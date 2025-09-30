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

    def _process_image_input(self, image_input: Gemma3ImageInputs) -> list[torch.Tensor]:
        assert self.vision_tower is not None
        pixel_values = image_input["pixel_values"]
        num_patches = image_input["num_patches"]

        batch_breakdown = self.greedy_plan(pixel_values.shape[0],
                                           [1, 2, 4, 8])  ##self.vision_buckets.multimodal_buckets)
        start_idx = 0
        image_embeds_multibatches = []

        for i in batch_breakdown:
            end_idx = start_idx + i
            indices = torch.arange(start_idx, end_idx)
            batch_sliced_pixel_values = torch.index_select(pixel_values, dim=0, index=indices)

            image_features = self._image_pixels_to_features(
                self.vision_tower,
                batch_sliced_pixel_values,
            )
            image_embeds = self.multi_modal_projector(image_features)
            image_embeds_multibatches += [image_embeds.clone()]
            start_idx = end_idx
        image_embeds = torch.cat(image_embeds_multibatches, dim=0)

        return [e.flatten(0, 1) for e in image_embeds.split(num_patches.tolist())]

    def greedy_plan(self, batchsize, available_batchsizes):
        # sort descending
        available_batchsizes_sorted = sorted(available_batchsizes, key=lambda x: -x)
        idx = 0
        left_to_process = batchsize
        result = []
        while (left_to_process > 0 and idx < len(available_batchsizes_sorted)):
            if available_batchsizes_sorted[idx] <= left_to_process:
                result += [available_batchsizes_sorted[idx]]
                left_to_process -= available_batchsizes_sorted[idx]
            else:
                idx += 1
        if left_to_process > 0:
            result += [available_batchsizes_sorted[-1]]  # this will be padded
        return result
