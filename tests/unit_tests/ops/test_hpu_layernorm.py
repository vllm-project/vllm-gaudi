# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import habana_frameworks.torch as htorch
from utils import temporary_op_registry_oot, register_op
from vllm_gaudi.ops.hpu_layernorm import HPURMSNorm
from vllm_gaudi.utils import HPUCompileConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.platforms import current_platform

DTYPES = [torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]
HIDDEN_SIZES = [8, 768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
ADD_RESIDUAL = [False, True]
DEVICE = [current_platform.device_type]
IS_STRIDED = [False, True]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICE)
@pytest.mark.parametrize("strided_input", IS_STRIDED)
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    device: str,
    strided_input: bool,
) -> None:
    with temporary_op_registry_oot():
        # prepare native RMSNorm module
        native_rms_norm = RMSNorm(hidden_size=hidden_size, eps=1e-05)
        native_rms_norm = native_rms_norm.to(dtype=dtype).to(device)
        native_rms_norm.weight.data.normal_(mean=1.0, std=0.1)
        assert isinstance(native_rms_norm, RMSNorm) and not isinstance(native_rms_norm, HPURMSNorm)

        # Prepare oot HPURMSNorm module
        register_op(RMSNorm, HPURMSNorm)
        oot_rms_norm = RMSNorm(hidden_size=hidden_size, eps=1e-05)
        oot_rms_norm = oot_rms_norm.to(dtype=dtype).to(device)
        oot_rms_norm.weight.data = native_rms_norm.weight.data.clone()
        assert isinstance(oot_rms_norm, RMSNorm) and isinstance(oot_rms_norm, HPURMSNorm)

        if not htorch.utils.internal.is_lazy():
            compile_config = HPUCompileConfig()
            oot_rms_norm = torch.compile(oot_rms_norm, **compile_config.get_compile_args())

        # Prepare input data
        scale = 1 / (2 * hidden_size)
        last_dim = 2 * hidden_size if strided_input else hidden_size
        x = torch.randn(num_tokens, last_dim, dtype=dtype, device=device)
        x = x[..., :hidden_size]
        assert x.is_contiguous() != strided_input
        x *= scale
        residual = torch.randn_like(x) * scale if add_residual else None

        # Execute layers
        ref_out = native_rms_norm(x, residual)
        out = oot_rms_norm(x, residual)

        # Check correctness
        if add_residual:
            torch.testing.assert_close(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
