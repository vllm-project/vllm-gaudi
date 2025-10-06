from vllm.model_executor.models.registry import ModelRegistry


def register_model():
    from vllm_gaudi.models.gemma3_mm import HpuGemma3ForConditionalGeneration  # noqa: F401

    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",  # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3_mm:HpuGemma3ForConditionalGeneration")
