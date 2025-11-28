from vllm.distributed import (cleanup_dist_env_and_memory, init_distributed_environment, initialize_model_parallel)
import pytest
import tempfile
from huggingface_hub import snapshot_download


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
    cleanup_dist_env_and_memory()


@pytest.fixture(scope="session")
def llama32_lora_files():
    return snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")
