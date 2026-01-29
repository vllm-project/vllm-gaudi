from vllm.model_executor.models.registry import ModelRegistry


def register_model():
    from vllm_gaudi.models.gemma3_mm import HpuGemma3ForConditionalGeneration  # noqa: F401

    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",  # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3_mm:HpuGemma3ForConditionalGeneration")

    from vllm_gaudi.models.qwen2_5_vl import HpuQwen2_5_VLForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration",
                                 "vllm_gaudi.models.qwen2_5_vl:HpuQwen2_5_VLForConditionalGeneration")

    from vllm_gaudi.models.qwen3_vl import HpuQwen3_VLForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Qwen3VLForConditionalGeneration",
                                 "vllm_gaudi.models.qwen3_vl:HpuQwen3_VLForConditionalGeneration")

    from vllm_gaudi.models.ernie45_vl import HpuErnie4_5_VLMoeForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Ernie4_5_VLMoeForConditionalGeneration",
                                 "vllm_gaudi.models.ernie45_vl:HpuErnie4_5_VLMoeForConditionalGeneration")
