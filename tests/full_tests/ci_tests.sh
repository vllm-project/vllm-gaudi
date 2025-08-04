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

# awq
echo "Testing awq inference with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model TheBloke/Llama-2-7B-Chat-AWQ --dtype bfloat16 --quantization awq_hpu
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model TheBloke/Llama-2-7B-Chat-AWQ --dtype bfloat16 --quantization awq_hpu
if [ $? -ne 0 ]; then
    echo "Error: Test failed for awq" >&2
    exit -1
fi
echo "Test with awq passed"

# gptq
echo "Testing gptq inference with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model TheBloke/Llama-2-7B-Chat-GPTQ --dtype bfloat16 --quantization gptq_hpu
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model TheBloke/Llama-2-7B-Chat-GPTQ --dtype bfloat16 --quantization gptq_hpu
if [ $? -ne 0 ]; then
    echo "Error: Test failed for gptq" >&2
    exit -1
fi
echo "Test with gptq passed"

