# basic model
echo "Testing basic model with vllm-hpu plugin v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model facebook/opt-125m
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model facebook/opt-125m
if [ $? -ne 0 ]; then
    echo "Error: Test failed for basic model" >&2
    exit -1
fi
echo "Test with basic model passed"

# multimodal-support with qwen2.5-vl
##echo "Testing Qwen2.5-VL-7B"
##echo "VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
##python -u vllm-gaudi/tests/models/language/generation/generation_mm.py --model-card-path vllm-gaudi/tests/full_tests/model_cards/qwen2.5-vl-7b.yaml"
##VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
##python -u vllm-gaudi/tests/models/language/generation/generation_mm.py --model-card-path vllm-gaudi/tests/full_tests/model_cards/qwen2.5-vl-7b.yaml
##if [ $? -ne 0 ]; then
##    echo "Error: Test failed for multimodal-support with qwen2.5-vl-7b" >&2
##    exit -1
##fi
##echo "Test with multimodal-support with qwen2.5-vl-7b passed"

echo "VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
python -u vllm-gaudi/tests/models/language/generation/generation_mm.py --model-card-path vllm-gaudi/tests/full_tests/model_cards/gemma-3-27b-it.yaml"
VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
python -u vllm-gaudi/tests/models/language/generation/generation_mm.py --model-card-path vllm-gaudi/tests/full_tests/model_cards/gemma-3-27b-it.yaml
if [ $? -ne 0 ]; then
    echo "Error: Test failed for multimodal-support with gemma-3-27b-it" >&2
    exit -1
fi
echo "Test with multimodal-support with gemma-3-27b-it passed"
