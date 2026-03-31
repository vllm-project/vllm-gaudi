import os
import json
import sys

from vllm_gaudi.platform import HpuPlatform


def _uses_lmcache_connector() -> bool:
    """Check if lmcache is configured as the KV connector.

    Detection is based on:
    - Environment variables (for programmatic usage), and
    - CLI args (for command-line usage via --kv-transfer-config).
    """

    def _is_lmcache_connector(connector_value: str) -> bool:
        """Return True if the given connector string represents an LMCache connector."""
        if not isinstance(connector_value, str):
            return False
        return "LMCache" in connector_value

    # 1. Check env var that may mirror --kv-transfer-config JSON.
    #    This supports programmatic workflows that configure KVTransferConfig
    #    and then expose it via environment instead of CLI.
    env_kv_config = os.getenv("VLLM_KV_TRANSFER_CONFIG")
    if env_kv_config:
        try:
            config = json.loads(env_kv_config)
            connector = config.get("kv_connector", "")
            if _is_lmcache_connector(connector):
                return True
        except (json.JSONDecodeError, TypeError):
            # Fall through to other detection mechanisms.
            pass

    # 2. Check a simple env var that may directly specify the connector name.
    env_kv_connector = os.getenv("VLLM_KV_CONNECTOR")
    if env_kv_connector and _is_lmcache_connector(env_kv_connector):
        return True

    # 3. Fallback: inspect CLI args for --kv-transfer-config as before.
    for i, arg in enumerate(sys.argv):
        if arg == "--kv-transfer-config" and i + 1 < len(sys.argv):
            try:
                config = json.loads(sys.argv[i + 1])
                connector = config.get("kv_connector", "")
                return _is_lmcache_connector(connector)
            except (json.JSONDecodeError, TypeError):
                return False
    return False


def register():
    """Register the HPU platform."""
    HpuPlatform.set_torch_compile()
    # Monkey patch for LMCache
    # LMCache requires PT_HPU_GPU_MIGRATION=1
    # However, hooking torch.cuda.is_available() by
    # migration introduces a lot of issues in LMCache + Gaudi
    # Remove torch.cuda.is_available hook here as an alternative solution
    if _uses_lmcache_connector():
        HpuPlatform.adjust_cuda_hooks()
    return "vllm_gaudi.platform.HpuPlatform"


def register_utils():
    """Register utility functions for the HPU platform."""
    import vllm_gaudi.utils  # noqa: F401
    # Install the in-process EngineCore reconfigure hook only when
    # multi-model mode is requested, to avoid heavy imports for all users.
    import os
    if os.environ.get("VLLM_HPU_MULTI_MODEL_CONFIG"):
        from vllm_gaudi.v1.engine.core_patch import install_engine_core_patch
        install_engine_core_patch()


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
    import vllm_gaudi.ops.hpu_weights  # noqa: F401

    # Conditionally register HPURowParallelLinear only when chunking is enabled
    from vllm_gaudi.ops.hpu_row_parallel_linear import register as register_row_parallel
    register_row_parallel()

    # Register HPU LoRA layers only when row parallel chunking is active
    env_value = os.environ.get('VLLM_ROW_PARALLEL_CHUNKS', '1')
    try:
        row_parallel_chunks = int(env_value)
    except ValueError:
        row_parallel_chunks = 1
    if row_parallel_chunks > 1:
        from vllm_gaudi.lora.layers.hpu_row_parallel_linear import register_hpu_lora_layers
        register_hpu_lora_layers()


def register_models():
    import vllm_gaudi.models.utils  # noqa: F401
    import vllm_gaudi.models.interfaces  # noqa: F401
    import vllm_gaudi.models.bert  # noqa: F401
    from .models import register_model
    register_model()
