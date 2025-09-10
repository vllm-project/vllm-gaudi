from vllm_gaudi.platform import HpuPlatform
import os


def register():
    """Register the HPU platform."""
    HpuPlatform.set_torch_compile()
    if os.getenv("VLLM_WEIGHT_LOAD_FORCE_SYNC", "false").lower() in ("true", "1"):
        HpuPlatform.set_synchronized_weight_loader()
    return "vllm_gaudi.platform.HpuPlatform"


def register_ops():
    """Register custom ops for the HPU platform."""
    import vllm_gaudi.v1.sample.hpu_rejection_sampler  # noqa: F401
    import vllm_gaudi.ops.hpu_fused_moe  # noqa: F401
    import vllm_gaudi.ops.hpu_layernorm  # noqa: F401
    import vllm_gaudi.ops.hpu_lora  # noqa: F401
    import vllm_gaudi.ops.hpu_rotary_embedding  # noqa: F401
    import vllm_gaudi.ops.hpu_compressed_tensors  # noqa: F401
    import vllm_gaudi.ops.hpu_fp8  # noqa: F401
    import vllm_gaudi.ops.hpu_gptq  # noqa: F401
    import vllm_gaudi.ops.hpu_awq  # noqa: F401
