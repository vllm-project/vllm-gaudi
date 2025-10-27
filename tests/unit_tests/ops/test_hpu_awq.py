# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import habana_frameworks.torch as htorch
from utils import get_data_path, create_row_parallel_linear
from vllm_gaudi.ops.hpu_awq import AWQHPULinearMethod, AWQHPUConfig
from vllm_gaudi.utils import HPUCompileConfig
from safetensors import safe_open


def test_awq_linear_method(dist_init):
    config = {"bits": 4, "group_size": 128, "zero_point": True}
    oot_quant_config = AWQHPUConfig.from_config(config)

    # Prepare linear layer with oot AWQHPULinearMethod
    oot_op = create_row_parallel_linear(input_size=256, output_size=128, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, AWQHPULinearMethod)

    # qweight, qzeros, scales were extracted from first RowParallelLinear of TheBloke/Llama-2-7B-Chat-AWQ
    # (with adjusted shape, to make tensors smaller)
    with safe_open(get_data_path("data/awq/linear.safetensors"), framework="pt", device="hpu") as f:
        oot_op.qweight.copy_(f.get_tensor("qweight"))
        oot_op.qzeros.copy_(f.get_tensor("qzeros"))
        oot_op.scales.copy_(f.get_tensor("scales"))
    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of AWQLinearMethod for given input
    # (AWQLinearMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/awq/linear.safetensors"), framework="pt", device="hpu") as f:
        input = f.get_tensor("input").to(torch.bfloat16)
        ref_output = f.get_tensor("ref_output").to(torch.bfloat16)

    # Execute layer
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)
