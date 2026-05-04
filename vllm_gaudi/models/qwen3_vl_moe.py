from torch import Tensor
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from vllm.sequence import IntermediateTensors
from vllm_gaudi.models.qwen3_moe import upgrade_qwen3_moe_blocks_inplace


class HpuQwen3_VLMoeForConditionalGeneration(Qwen3VLMoeForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # qwen3 moe mlp blocks: make forward for 3d safe (b,s,h -> t,h)
        lm = getattr(self, "language_model", None)
        if lm is not None:
            _n = upgrade_qwen3_moe_blocks_inplace(lm)

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor | IntermediateTensors:
        # HPU 2D padded prefill produces 3D inputs_embeds (B, S, H).
        # Upstream deepstack uses inputs_embeds.size(0) to slice its
        # buffer, which yields (B, H) instead of (T, H) when 3D,
        # causing shape mismatch in the per-layer addition.
        if (self.use_deepstack and inputs_embeds is not None and inputs_embeds.ndim == 3):
            inputs_embeds = inputs_embeds.reshape(-1, inputs_embeds.size(-1))
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
