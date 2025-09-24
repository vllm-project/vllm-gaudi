import torch
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod, GPTQConfig
from vllm.model_executor.layers.linear import RowParallelLinear
import tempfile
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)

temp_file = tempfile.mkstemp()[1]
init_distributed_environment(
    world_size=1,
    rank=0,
    distributed_init_method=f"file://{temp_file}",
    local_rank=0,
    backend="hccl",
)
initialize_model_parallel(1, 1)

def test_gptq_linear_method():
    config = {"bits": 4,
              "group_size": 128,
              "desc_act": False,
              "lm_head": False}
    native_quant_config = GPTQConfig.from_config(config)

    # prepare linear layer with native GPTQLinearMethod
    native_op = RowParallelLinear(input_size=256,
                                  output_size=8,
                                  bias=False,
                                  input_is_parallel=True,
                                  skip_bias_add=False,
                                  params_dtype=torch.float16,
                                  reduce_results=True,
                                  quant_config=native_quant_config,
                                  return_bias=True,
                                  disable_tp=False
                                  ).to("cuda")
    assert isinstance(native_op.quant_method, GPTQLinearMethod) 

    import pdb
    pdb.set_trace()

    qweight = torch.load("/software/users/kpietkun/gptq/cuda_qweight.pt", weights_only=False, map_location="cuda")
    native_op.qweight.copy_(qweight)
    qzeros = torch.load("/software/users/kpietkun/gptq/cuda_qzeros.pt", weights_only=False, map_location="cuda")
    native_op.qzeros.data.copy_(qzeros)
    scales = torch.load("/software/users/kpietkun/gptq/cuda_scales.pt", weights_only=False, map_location="cuda")
    native_op.scales.data.copy_(scales)


    # Input and expected output
    # Output tensor holds the data that was returned by GPTQLinearMethod for given input
    # (GPTQLinearMethod was triggered offline with the same input as below to get the desired output)
    input = torch.load("/software/users/kpietkun/gptq/cuda_input.pt", weights_only=False, map_location="cuda")
    ref_output = torch.load("/software/users/kpietkun/gptq/cuda_output.pt", weights_only=False, map_location="cuda")

    # Execute layers
    native_op.quant_method.process_weights_after_loading(native_op)
    out = native_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out[0], atol=1e-3, rtol=1e-3)

    print(out)
    pdb.set_trace()


test_gptq_linear_method()
