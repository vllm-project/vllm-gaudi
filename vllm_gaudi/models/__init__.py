from vllm.model_executor.models.registry import ModelRegistry


def register_model():
    from vllm_gaudi.models.gemma3_mm import HpuGemma3ForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",  # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3_mm:HpuGemma3ForConditionalGeneration")

    from vllm_gaudi.models.qwen2_5_vl import HpuQwen2_5_VLForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration",
                                 "vllm_gaudi.models.qwen2_5_vl:HpuQwen2_5_VLForConditionalGeneration")

    from vllm_gaudi.models.ernie45_vl import HpuErnie4_5_VLMoeForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Ernie4_5_VLMoeForConditionalGeneration",
                                 "vllm_gaudi.models.ernie45_vl:HpuErnie4_5_VLMoeForConditionalGeneration")

    from vllm_gaudi.models.ovis import HpuOvis  # noqa: F401
    ModelRegistry.register_model("Ovis", "vllm_gaudi.models.ovis:HpuOvis")

    from vllm_gaudi.models.qwen3_vl_moe import HpuQwen3_VLMoeForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Qwen3VLMoeForConditionalGeneration",
                                 "vllm_gaudi.models.qwen3_vl_moe:HpuQwen3_VLMoeForConditionalGeneration")

    from vllm_gaudi.models.hunyuan_v1 import HpuHunYuanDenseV1ForCausalLM  # noqa: F401
    ModelRegistry.register_model("HunYuanDenseV1ForCausalLM",
                                 "vllm_gaudi.models.hunyuan_v1:HpuHunYuanDenseV1ForCausalLM")

    from vllm_gaudi.models.hunyuan_v1 import HpuHunYuanMoEV1ForCausalLM  # noqa: F401
    ModelRegistry.register_model("HunYuanMoEV1ForCausalLM", "vllm_gaudi.models.hunyuan_v1:HpuHunYuanMoEV1ForCausalLM")

    from vllm_gaudi.models.minimax_m2 import HpuMiniMaxM2ForCausalLM  # noqa: F401
    ModelRegistry.register_model("MiniMaxM2ForCausalLM", "vllm_gaudi.models.minimax_m2:HpuMiniMaxM2ForCausalLM")
    from vllm_gaudi.models.pixtral import HPUPixtralForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("PixtralForConditionalGeneration",
                                 "vllm_gaudi.models.pixtral:HPUPixtralForConditionalGeneration")

    from vllm_gaudi.models.dots_ocr import HpuDotsOCRForCausalLM  # noqa: F401
    ModelRegistry.register_model("DotsOCRForCausalLM", "vllm_gaudi.models.dots_ocr:HpuDotsOCRForCausalLM")

    from vllm_gaudi.models.seed_oss import HpuSeedOssForCausalLM  # noqa: F401
    ModelRegistry.register_model("SeedOssForCausalLM", "vllm_gaudi.models.seed_oss:HpuSeedOssForCausalLM")

    import vllm_gaudi.models.deepseek_v2  # noqa: F401

    from vllm_gaudi.models.deepseek_ocr import HpuDeepseekOCRForCausalLM  # noqa: F401
    ModelRegistry.register_model("DeepseekOCRForCausalLM", "vllm_gaudi.models.deepseek_ocr:HpuDeepseekOCRForCausalLM")
