# basic model
echo "Testing basic model with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/upstream_tests/generate.py --model facebook/opt-125m
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/upstream_tests/generate.py --model facebook/opt-125m
echo "Test with basic model passed"

# tp=2
echo "Testing tensor parallel size 2 with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/upstream_tests/generate.py --model facebook/opt-125m --tensor-parallel-size 2
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/upstream_tests/generate.py --model facebook/opt-125m --tensor-parallel-size 2
echo "Test with tensor parallel size 2 passed"

# mla and moe
echo "Testing MLA and MoE with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/upstream_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/upstream_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code
echo "Test with deepseek v2 lite passed"