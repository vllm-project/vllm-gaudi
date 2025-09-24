import torch
import habana_frameworks.torch as htorch
from vllm_gaudi.ops.hpu_compressed_tensors import HPUCompressedTensorsLinearMethod, CompressedTensorsConfig, HPUCompressedTensorsW8A8Fp8
from vllm_gaudi.utils import HPUCompileConfig
from vllm.model_executor.layers.linear import RowParallelLinear


def test_compressed_tensors_linear_method(dist_init):
    """
    HPUCompressedTensorsLinearMethod with HPUCompressedTensorsW8A8Fp8 scheme
    """
    config = {'config_groups': {'group_0': {'input_activations': {'actorder': None, 'block_structure': None, 'dynamic': True, 'group_size': None, 'num_bits': 8, 'observer': None, 'observer_kwargs': {}, 'strategy': 'token', 'symmetric': True, 'type': 'float'}, 'output_activations': None, 'targets': ['Linear'], 'weights': {'actorder': None, 'block_structure': None, 'dynamic': False, 'group_size': None, 'num_bits': 8, 'observer': 'mse', 'observer_kwargs': {}, 'strategy': 'channel', 'symmetric': True, 'type': 'float'}}}, 'format': 'float-quantized', 'global_compression_ratio': None, 'ignore': ['lm_head'], 'kv_cache_scheme': None, 'quant_method': 'compressed-tensors', 'quantization_status': 'compressed'}
    oot_quant_config = CompressedTensorsConfig.from_config(config)

    # Prepare linear layer with oot AWQHPULinearMethod
    oot_op = RowParallelLinear(input_size=4096,
                               output_size=4096,
                               bias=False,
                               input_is_parallel=True,
                               skip_bias_add=False,
                               params_dtype=torch.bfloat16,
                               reduce_results=True,
                               quant_config=oot_quant_config,
                               return_bias=False,
                               disable_tp=False).to("hpu")
    import pdb
    pdb.set_trace()
    assert isinstance(oot_op.quant_method, HPUCompressedTensorsLinearMethod)
    assert isinstance(oot_op.scheme, HPUCompressedTensorsW8A8Fp8)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # weight and weight_scale extracted from first RowParallelLinear of RedHatAI/Qwen3-8B-FP8-dynamic
    # (with adjusted shape, to make tensors smaller)
    weight = torch.load("data/compressed_tensors/weight.pt", weights_only=False, map_location="hpu")
    oot_op.weight.copy_(weight)
    weight_scale = torch.load("data/compressed_tensors/weight_scale.pt", weights_only=False, map_location="hpu")
    oot_op.weight_scale.data.copy_(weight_scale)

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of CompressedTensorsLinearMethod for given input
    # (CompressedTensorsLinearMethod was triggered offline with the same input as below to get the desired output)
    input = torch.load("data/compressed_tensors/input.pt", weights_only=False, map_location="hpu")
    ref_output = torch.load("data/compressed_tensors/output.pt", weights_only=False, map_location="hpu")

    # Execute layer
    oot_op.quant_method.process_weights_after_loading(oot_op)
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)
