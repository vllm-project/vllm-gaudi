import gc

import torch
from vllm.distributed import (init_distributed_environment, initialize_model_parallel)
import pytest
import tempfile
from huggingface_hub import snapshot_download
from vllm import envs
from vllm.distributed import parallel_state
from vllm.platforms import current_platform
"""Ops-level monkey patches for Gaudi.

Provides the HPU-specific override of distributed cleanup so `_host_emptyCache`
runs when available. This module can be imported from either ops or
distributed; it updates `vllm.distributed.parallel_state.cleanup_dist_env_and_memory`
in place.
"""


def hpu_cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    # Reset environment variable cache
    envs.disable_envs_cache()
    # Ensure all objects are not frozen before cleanup
    gc.unfreeze()

    parallel_state.destroy_model_parallel()
    parallel_state.destroy_distributed_environment()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    gc.collect()

    empty_cache = current_platform.empty_cache
    if empty_cache is not None:
        empty_cache()
    try:
        if not current_platform.is_cpu():
            torch._C._host_emptyCache()
    except AttributeError:
        parallel_state.logger.warning("torch._C._host_emptyCache() only available in Pytorch >=2.5")


# Apply monkey-patch on import
#parallel_state.cleanup_dist_env_and_memory = _hpu_cleanup_dist_env_and_memory


@pytest.fixture
def dist_init():
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="hccl",
    )
    initialize_model_parallel(1, 1)
    yield
    hpu_cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def llama32_lora_files():
    return snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")


@pytest.fixture
def default_vllm_config():
    """Set a default VllmConfig for tests that directly test CustomOps or pathways
    that use get_current_vllm_config() outside of a full engine context.
    """
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield
