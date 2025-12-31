import torch
from vllm.v1.kv_cache_interface import MambaSpec, KVCacheConfig, KVCacheTensor, KVCacheGroupSpec
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.attention.layer import Attention
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner
from vllm.config import VllmConfig
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


def get_vllm_config():
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        is_encoder_decoder=False,  # Added required field
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="generate",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=128,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
    )
    return vllm_config


def test_get_kv_cache_spec_with_mamba():

    class FakeMamba(MambaBase):

        def get_state_shape(self):
            return [(2, 3), (4, 5)]

        @property
        def mamba_type(self):
            return "fake_mamba"

        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)

    vllm_config = get_vllm_config()
    vllm_config.cache_config.mamba_block_size = 42
    vllm_config.cache_config.mamba_page_size_padded = 128
    vllm_config.compilation_config.static_forward_context = {"layer.mamba": FakeMamba()}
    runner = HPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    assert "layer.mamba" in kv_cache_spec
    spec = kv_cache_spec["layer.mamba"]
    assert isinstance(spec, MambaSpec)
    assert spec.shapes == [(2, 3), (4, 5)]
    assert spec.dtypes == (torch.bfloat16, torch.bfloat16)
    assert spec.block_size == 42
    assert spec.page_size_padded == 128
    assert spec.mamba_type == "fake_mamba"
    assert spec.num_speculative_blocks == 0


def test_kv_cache_tensor_shape_with_mamba():

    class FakeMamba(MambaBase):

        def get_state_shape(self):
            return ((2, ), (3, ))

        @property
        def mamba_type(self):
            return "mamba1"

        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)

    vllm_config = get_vllm_config()
    vllm_config.cache_config.mamba_block_size = 5
    vllm_config.cache_config.mamba_page_size_padded = 128
    # Use 'layer.0' as the mamba layer name to match main code expectations
    mamba_layer_name = "layer.0"
    vllm_config.compilation_config.static_forward_context = {mamba_layer_name: FakeMamba()}
    runner = HPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    spec = kv_cache_spec[mamba_layer_name]
    num_blocks = 7
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=[mamba_layer_name])],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[mamba_layer_name], kv_cache_spec=spec)],
    )
    runner.initialize_kv_cache(kv_cache_config)

    kv_caches = runner.kv_caches
    if isinstance(kv_caches, (tuple, list)):
        # If nested list (e.g., [[tensor1, tensor2]])
        tensors = kv_caches[0] if len(kv_caches) == 1 and isinstance(kv_caches[0], (tuple, list)) else kv_caches
        assert tensors[0].shape == (num_blocks, 2)
        assert tensors[1].shape == (num_blocks, 3)
    else:
        shape = kv_caches.shape
        assert (shape == (num_blocks, 2)) or (shape == (num_blocks, 3)) or (shape[:3] == (2, 3, num_blocks))


def test_kv_cache_tensor_shape_with_two_mamba_layers():

    class FakeMamba(MambaBase):

        def __init__(self, shape):
            self._shape = shape

        def get_state_shape(self):
            return self._shape

        @property
        def mamba_type(self):
            return "mamba1"

        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)

    vllm_config = get_vllm_config()
    vllm_config.cache_config.mamba_block_size = 5
    vllm_config.cache_config.mamba_page_size_padded = 128
    # Two mamba layers with different shapes
    mamba_layer_names = ["layer.0", "layer.1"]
    shapes = [((2, ), (3, )), ((4, ), (5, ))]
    vllm_config.compilation_config.static_forward_context = {
        mamba_layer_names[0]: FakeMamba(shapes[0]),
        mamba_layer_names[1]: FakeMamba(shapes[1]),
    }
    runner = HPUModelRunner(vllm_config, DEVICE)
    kv_cache_spec = runner.get_kv_cache_spec()
    num_blocks = 7
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=kv_cache_spec[mamba_layer_names[0]].page_size_bytes * num_blocks,
                          shared_by=[mamba_layer_names[0]]),
            KVCacheTensor(size=kv_cache_spec[mamba_layer_names[1]].page_size_bytes * num_blocks,
                          shared_by=[mamba_layer_names[1]]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[mamba_layer_names[0]], kv_cache_spec=kv_cache_spec[mamba_layer_names[0]]),
            KVCacheGroupSpec(layer_names=[mamba_layer_names[1]], kv_cache_spec=kv_cache_spec[mamba_layer_names[1]]),
        ],
    )
    runner.initialize_kv_cache(kv_cache_config)
    kv_caches = runner.kv_caches
    for i, lname in enumerate(mamba_layer_names):
        print(f"kv_cache_spec for {lname}: {kv_cache_spec[lname].shapes}")
        tensors = kv_caches[i]
        print(f"FakeMamba.get_state_shape() for {lname}: {tensors[0].shape}, {tensors[1].shape}")
        assert tensors[0].shape == (num_blocks, shapes[i][0][0])
        assert tensors[1].shape == (num_blocks, shapes[i][1][0])


def test_hpu_hybrid_attention_mamba_kv_cache_padding_and_shapes():
    """
    Hybrid on HPU means:
      - Attention and Mamba coexist in the same model
      - They may have different block sizes
      - Padding / kernel block alignment is applied
      - KV caches are per-layer (no shared backing tensor)

    This test verifies that:
      1) KV cache initialization succeeds with hybrid block sizes
      2) Mamba KV cache has exact logical blocks
      3) Attention KV cache has padded kernel blocks (>= num_blocks)
      4) Dtypes and basic layout are correct
    """

    # Fake Mamba layer
    class FakeMamba(MambaBase):

        def __init__(self, shapes, block_size=13107):
            self._shapes = shapes  # sum(shapes) should be 10 to get close to 2.69M
            self.block_size = block_size

        def get_state_shape(self):
            return self._shapes

        @property
        def mamba_type(self):
            return "mamba2"

        def get_state_dtype(self):
            return (torch.bfloat16, torch.bfloat16)

        @property
        def page_size_bytes(self):
            total_elements_per_block = sum([s[0] for s in self._shapes]) * self.block_size
            return total_elements_per_block * 2  # bfloat16

    vllm_config = get_vllm_config()

    # Hybrid Mamba settings (normally set by
    # HybridAttentionMambaModelConfig.verify_and_update_config)
    # vllm_config.cache_config.block_size == 128  Attention block size
    vllm_config.cache_config.mamba_block_size = 13107
    # mamba_obj.page_size_bytes == 131070 , attn_spec.page_size_bytes == 65536
    vllm_config.cache_config.mamba_page_size_padded = 131072  # mamba_spec.page_size_bytes

    # Layers
    mamba_layer = "layers.0.mamba"
    attn_layer = "layers.1.attn"

    mamba_shapes = ((2, ), (3, ))
    num_heads = 2
    head_size = 64
    num_blocks = 8  # logical number of blocks

    vllm_config.compilation_config.static_forward_context = {
        mamba_layer: FakeMamba(mamba_shapes),
        attn_layer: Attention(num_heads, head_size, 0.1),
    }

    runner = HPUModelRunner(vllm_config, DEVICE)

    kv_specs = runner.get_kv_cache_spec()
    mamba_spec = kv_specs[mamba_layer]

    attn_spec = FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )

    # KV cache config
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=mamba_spec.page_size_bytes * num_blocks,
                shared_by=[mamba_layer],
            ),
            KVCacheTensor(
                size=attn_spec.page_size_bytes * num_blocks,
                shared_by=[attn_layer],
            ),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec([mamba_layer], mamba_spec),
            KVCacheGroupSpec([attn_layer], attn_spec),
        ],
    )

    # Initialize KV cache
    runner.initialize_kv_cache(kv_cache_config)
    kv_caches = runner.kv_caches
    mamba_cache = kv_caches[0]
    attn_cache = kv_caches[1]

    # Mamba checks
    expected_mamba_blocks = num_blocks
    # padded_blocks should be  mamba_block_size * num_blocks = 13107 * 8 = 104,856

    assert mamba_cache[0].shape[0] == expected_mamba_blocks
    assert mamba_cache[0].shape[1] == mamba_shapes[0][0]
    assert mamba_cache[1].shape[0] == expected_mamba_blocks
    assert mamba_cache[1].shape[1] == mamba_shapes[1][0]
    assert mamba_cache[0].dtype == torch.bfloat16
    assert mamba_cache[1].dtype == torch.bfloat16

    # Attention checks
    assert attn_cache[0].shape[1] == 2  # K/V dimension

    # Compute Attention blocks
    logical_blocks = num_blocks + 1
    attention_block_size = vllm_config.cache_config.block_size
    expected_attn_blocks = logical_blocks * attention_block_size
    # padded_block_size should be  131072 / 65536 * attn_block_size = 256; padded_blocks = 256 * num_blocks = 2048

    assert attn_cache[0].shape[0] == expected_attn_blocks  # 1152
    assert attn_cache[0].dtype == torch.bfloat16

    # Hybrid padding sanity
    assert vllm_config.cache_config.mamba_page_size_padded >= vllm_config.cache_config.mamba_block_size
    assert vllm_config.cache_config.block_size != vllm_config.cache_config.mamba_block_size
