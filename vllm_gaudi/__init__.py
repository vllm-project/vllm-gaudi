from vllm_gaudi.platform import HpuPlatform


def register():
    """Register the HPU platform."""
    HpuPlatform.set_torch_compile()
    return "vllm_gaudi.platform.HpuPlatform"


def register_ops():
    """Register custom ops for the HPU platform."""
    import vllm_gaudi.ops  # noqa: F401
