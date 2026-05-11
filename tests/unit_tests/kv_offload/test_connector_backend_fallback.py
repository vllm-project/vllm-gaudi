# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.unit_tests.kv_offload.utils import create_vllm_config
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends


def test_cpu_connector_backend_uses_cpu_backend():
    vllm_config = create_vllm_config()

    with set_current_vllm_config(vllm_config):
        backend = get_current_attn_backends(vllm_config)[0]

    assert backend.get_name() == "CPU_ATTN"
