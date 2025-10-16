# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import pytest
import torch
import habana_frameworks.torch as htorch
from unittest.mock import patch, MagicMock
from vllm_gaudi.utils import HPUCompileConfig
from vllm.attention.layer import MultiHeadAttention
from vllm_gaudi.ops.hpu_multihead_attn import HpuMultiHeadAttention


@pytest.mark.parametrize("num_heads", [2, 8])
@pytest.mark.parametrize("head_size", [32, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_multi_head_attention(num_heads, head_size, num_kv_heads) -> None:
    scale = 1.0 / math.sqrt(head_size)
    hidden_size = num_heads * head_size
    batch_size = 2
    seq_len = 32
    # prepare native MultiHeadAttention module
    native_attn = MultiHeadAttention(num_heads, head_size, scale, num_kv_heads)
    assert not isinstance(native_attn, HpuMultiHeadAttention)

    # Prepare oot HpuMultiHeadAttention module
    oot_attn = HpuMultiHeadAttention(num_heads, head_size, scale, num_kv_heads).to("hpu")
    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_attn = torch.compile(oot_attn, **compile_config.get_compile_args())

    # Prepare input data
    kv_hidden_size = num_kv_heads * head_size
    query = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device="cpu")
    key = torch.randn(batch_size, seq_len, kv_hidden_size, dtype=torch.float16, device="cpu")
    value = torch.randn(batch_size, seq_len, kv_hidden_size, dtype=torch.float16, device="cpu")

    # Execute layers
    ref_out = native_attn(query, key, value)

    mock_config = MagicMock()
    mock_config.prompt_attn_impl = 'fsdpa_impl'
    with patch('vllm_gaudi.extension.runtime.get_config', return_value=mock_config):
        out = oot_attn(query.to("hpu"), key.to("hpu"), value.to("hpu"))

    # Check correctness
    torch.testing.assert_close(out.cpu(), ref_out, atol=1e-2, rtol=1e-2)
