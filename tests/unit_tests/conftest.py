from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
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
def sql_lora_huggingface_id():
    # huggingface repo id is used to test lora runtime downloading.
    return "yard1/llama-2-7b-sql-lora-test"


@pytest.fixture(scope="session")
def sql_lora_files(sql_lora_huggingface_id):
    return snapshot_download(repo_id=sql_lora_huggingface_id)
