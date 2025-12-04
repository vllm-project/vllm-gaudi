from vllm_gaudi.platform import HpuPlatform


def register():
    """Register the HPU platform."""
    HpuPlatform.set_torch_compile()
    return "vllm_gaudi.platform.HpuPlatform"


def register_ops():
    """Register custom ops for the HPU platform."""
    import vllm_gaudi.v1.sample.hpu_rejection_sampler  # noqa: F401
    import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.hpu_nixl_connector  # noqa: F401
    import vllm_gaudi.ops.hpu_fused_moe  # noqa: F401
    import vllm_gaudi.ops.hpu_layernorm  # noqa: F401
    import vllm_gaudi.ops.hpu_lora  # noqa: F401
    import vllm_gaudi.ops.hpu_rotary_embedding  # noqa: F401
    import vllm_gaudi.ops.hpu_compressed_tensors  # noqa: F401
    import vllm_gaudi.ops.hpu_fp8  # noqa: F401
    import vllm_gaudi.ops.hpu_gptq  # noqa: F401
    import vllm_gaudi.ops.hpu_awq  # noqa: F401
    import vllm_gaudi.ops.hpu_multihead_attn  # noqa: F401
    import vllm_gaudi.ops.hpu_conv  # noqa: F401


def register_models():
    import vllm_gaudi.models.interfaces  # noqa: F401
    from .models import register_model
    register_model()
