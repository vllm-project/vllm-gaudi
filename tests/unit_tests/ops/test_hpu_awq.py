import torch
import habana_frameworks.torch as htorch
from vllm_gaudi.ops.hpu_awq import AWQHPULinearMethod, AWQHPUConfig
from vllm_gaudi.utils import HPUCompileConfig
from vllm.model_executor.layers.linear import RowParallelLinear


def test_awq_linear_method(dist_init):
    config = {"bits": 4, "group_size": 128, "zero_point": True}
    oot_quant_config = AWQHPUConfig.from_config(config)

    # Prepare linear layer with oot AWQHPULinearMethod
    oot_op = RowParallelLinear(input_size=256,
                               output_size=128,
                               bias=False,
                               input_is_parallel=True,
                               skip_bias_add=False,
                               params_dtype=torch.bfloat16,
                               reduce_results=True,
                               quant_config=oot_quant_config,
                               return_bias=False,
                               disable_tp=False).to("hpu")
    assert isinstance(oot_op.quant_method, AWQHPULinearMethod)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # qweight, qzeros, scales extracted from first RowParallelLinear of TheBloke/Llama-2-7B-Chat-AWQ
    # (with adjusted shape, to make tensors smaller - output_size = 8 instead of 4096)
    qweight = torch.load("data/awq/qweight.pt", weights_only=False, map_location="hpu")
    oot_op.qweight.copy_(qweight)
    qzeros = torch.load("data/awq/qzeros.pt", weights_only=False, map_location="hpu")
    oot_op.qzeros.data.copy_(qzeros)
    scales = torch.load("data/awq/scales.pt", weights_only=False, map_location="hpu").to(torch.bfloat16)
    oot_op.scales.data.copy_(scales)

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of AWQLinearMethod for given input
    # (AWQLinearMethod was triggered offline with the same input as below to get the desired output)
    input = torch.load("data/awq/input.pt", weights_only=False, map_location="hpu").to(torch.bfloat16)
    ref_output = torch.load("data/awq/output.pt", weights_only=False, map_location="hpu").to(torch.bfloat16)

    # Execute layer
    oot_op.quant_method.process_weights_after_loading(oot_op)
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)
