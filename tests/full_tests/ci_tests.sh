# basic model
echo "Testing basic model with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model facebook/opt-125m
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model facebook/opt-125m
if [ $? -ne 0 ]; then
    echo "Error: Test failed for basic model" >&2
    exit -1
fi
echo "Test with basic model passed"

# tp=2
echo "Testing tensor parallel size 2 with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model facebook/opt-125m --tensor-parallel-size 2
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model facebook/opt-125m --tensor-parallel-size 2
if [ $? -ne 0 ]; then
    echo "Error: Test failed for tensor parallel size 2" >&2
    exit -1
fi
echo "Test with tensor parallel size 2 passed"

# mla and moe
echo "Testing MLA and MoE with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code
if [ $? -ne 0 ]; then
    echo "Error: Test failed for deepseek v2 lite passed" >&2
    exit -1
fi
echo "Test with deepseek v2 lite passed"

# deepseek v2 + inc + dynamic quantization + tp2
echo "Testing deepseek_v2 + inc dynamic quantization + tp2"
echo QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_dynamic_quant.json HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code  --quantization inc --kv_cache_dtype fp8_inc
QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_dynamic_quant.json \
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --quantization inc --tensor-parallel-size 2
if [ $? -ne 0 ]; then
    echo "Error: Test failed for deepseek_v2 + inc dynamic quantization + tp2" >&2
    exit -1
fi
echo "Test with deepseek_v2 + inc dynamic quantization + tp 2 successful"

# structured output
echo "Testing structured output"
echo HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/structured_outputs.py 
HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/structured_outputs.py 
if [ $? -ne 0 ]; then
    echo "Error: Test failed for structured outputs" >&2
    exit -1
fi
echo "Test with structured outputs passed"
