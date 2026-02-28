import os
import sys
import types
import importlib.machinery

from vllm_gaudi.platform import HpuPlatform

# Mock torchaudio if not available (HPU torch doesn't ship with torchaudio
# and the upstream wheel's C extension is incompatible). Several vLLM
# processors (e.g. FunASRProcessor) eagerly import torchaudio at module
# level, so we inject a lightweight stub to prevent ImportError.
if "torchaudio" not in sys.modules:
    try:
        import torchaudio  # noqa: F401
    except (ImportError, OSError):
        _ta = types.ModuleType("torchaudio")
        _ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
        _ta.__version__ = "0.0.0"  # type: ignore[attr-defined]
        _ta_compliance = types.ModuleType("torchaudio.compliance")
        _ta_compliance.__spec__ = importlib.machinery.ModuleSpec("torchaudio.compliance", None)
        _ta_kaldi = types.ModuleType("torchaudio.compliance.kaldi")
        _ta_kaldi.__spec__ = importlib.machinery.ModuleSpec("torchaudio.compliance.kaldi", None)
        _ta.compliance = _ta_compliance  # type: ignore[attr-defined]
        _ta_compliance.kaldi = _ta_kaldi  # type: ignore[attr-defined]
        sys.modules["torchaudio"] = _ta
        sys.modules["torchaudio.compliance"] = _ta_compliance
        sys.modules["torchaudio.compliance.kaldi"] = _ta_kaldi


def register():
    """Register the HPU platform."""
    HpuPlatform.set_torch_compile()
    return "vllm_gaudi.platform.HpuPlatform"


def register_utils():
    """Register utility functions for the HPU platform."""
    import vllm_gaudi.utils  # noqa: F401


def register_ops():
    """Register custom PluggableLayers for the HPU platform"""
    import vllm_gaudi.attention.oot_mla  # noqa: F401
    """Register custom ops for the HPU platform."""
    import vllm_gaudi.v1.sample.hpu_rejection_sampler  # noqa: F401
    import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.hpu_nixl_connector  # noqa: F401
    if os.getenv('VLLM_HPU_HETERO_KV_LAYOUT', 'false').lower() == 'true':
        import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.hetero_hpu_nixl_connector  # noqa: F401
    import vllm_gaudi.v1.kv_offload.worker.cpu_hpu  # noqa: F401
    import vllm_gaudi.ops.hpu_attention  # noqa: F401
    import vllm_gaudi.ops.hpu_fused_moe  # noqa: F401
    import vllm_gaudi.ops.hpu_grouped_topk_router  # noqa: F401
    import vllm_gaudi.ops.hpu_layernorm  # noqa: F401
    import vllm_gaudi.ops.hpu_lora  # noqa: F401
    import vllm_gaudi.ops.hpu_mamba_mixer2  # noqa: F401
    import vllm_gaudi.ops.hpu_rotary_embedding  # noqa: F401
    import vllm_gaudi.ops.hpu_modelopt  # noqa: F401
    import vllm_gaudi.ops.hpu_compressed_tensors  # noqa: F401
    import vllm_gaudi.ops.hpu_fp8  # noqa: F401
    import vllm_gaudi.ops.hpu_gptq  # noqa: F401
    import vllm_gaudi.ops.hpu_awq  # noqa: F401
    import vllm_gaudi.ops.hpu_conv  # noqa: F401
    import vllm_gaudi.ops.hpu_mm_encoder_attention  # noqa: F401


def register_models():
    import vllm_gaudi.models.utils  # noqa: F401
    import vllm_gaudi.models.interfaces  # noqa: F401
    import vllm_gaudi.models.bert  # noqa: F401
    import vllm_gaudi.models.roberta  # noqa: F401
    from .models import register_model
    register_model()
