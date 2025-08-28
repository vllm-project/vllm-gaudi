from vllm_gaudi.platform import HpuPlatform
import os


def register():
    """Register the HPU platform."""
    HpuPlatform.set_torch_compile()
    if os.getenv("VLLM_WEIGHT_LOAD_FORCE_SYNC",
                 "false").lower() in ("true", "1"):
        HpuPlatform.set_synchronized_weight_loader()
    return "vllm_gaudi.platform.HpuPlatform"


def register_ops():
    """Register custom ops for the HPU platform."""
    import vllm_gaudi.ops  # noqa: F401
    vllm_gaudi.ops.register_ops()
