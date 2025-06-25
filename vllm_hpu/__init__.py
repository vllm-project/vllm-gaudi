from vllm_hpu.platform import HpuPlatform


def register():
    """Register the HPU platform."""
    HpuPlatform.set_torch_compile()
    return "vllm_hpu.platform.HpuPlatform"


def register_ops():
    """Register custom ops for the HPU platform."""
    import vllm_hpu.ops  # noqa: F401
