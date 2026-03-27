import os
from vllm_gaudi.platform import HpuPlatform


def _patch_triton_compat():
    """Patch triton.next_power_of_2 for triton >= 3.6 compatibility.

    vLLM's fp8_utils.py calls triton.next_power_of_2() which was moved to
    triton.runtime in triton 3.6+.  On HPU triton kernels are not executed,
    but the function is still called during FP8 quantization setup.
    """
    try:
        import triton
        if not hasattr(triton, 'next_power_of_2'):
            try:
                from triton.runtime import next_power_of_2
                triton.next_power_of_2 = next_power_of_2
            except ImportError:
                triton.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    except ImportError:
        pass


# Apply triton compatibility patch early, before any vLLM model code runs.
_patch_triton_compat()


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

    import vllm_gaudi.ops.hpu_sparse_attn_indexer  # noqa: F401


def register_models():
    import vllm_gaudi.models.utils  # noqa: F401
    import vllm_gaudi.models.interfaces  # noqa: F401
    import vllm_gaudi.models.bert  # noqa: F401
    import vllm_gaudi.models.roberta  # noqa: F401
    from .models import register_model
    register_model()
