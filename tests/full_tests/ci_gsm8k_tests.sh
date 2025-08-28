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

# granite + inc
echo "Testing granite-8b + inc with vllm-hpu plugin v1"
echo QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_unit_scale_quant.json HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model ibm-granite/granite-3.3-2b-instruct --trust-remote-code  --quantization inc --kv_cache_dtype fp8_inc
QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_unit_scale_quant.json \
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model ibm-granite/granite-3.3-2b-instruct --trust-remote-code --quantization inc --kv_cache_dtype fp8_inc
if [ $? -ne 0 ]; then
    echo "Error: Test failed for granite + inc" >&2
    exit -1
fi
echo "Test with granite + inc passed"

# deepseek v2 + inc
echo "Testing deepseek_v2 + inc with vllm-hpu plugin v1"
echo QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_unit_scale_quant.json HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code  --quantization inc --kv_cache_dtype fp8_inc
QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_unit_scale_quant.json \
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --quantization inc --kv_cache_dtype fp8_inc
if [ $? -ne 0 ]; then
    echo "Error: Test failed for deepseek_v2 + inc" >&2
    exit -1
fi
echo "Test with deepseek_v2 + inc passed"

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
echo HABANA_VISIBLE_DEVICES=all VLLM_MERGED_PREFILL=True VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/structured_outputs.py 
HABANA_VISIBLE_DEVICES=all VLLM_MERGED_PREFILL=True VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/structured_outputs.py 
if [ $? -ne 0 ]; then
    echo "Error: Test failed for structured outputs with merged prefill" >&2
    exit -1
fi
echo "Test with structured outputs passed"
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

# gsm8k test
# used to check HPUattn + MLP
echo "Testing GSM8K on ganite-8b"
echo VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/granite-8b.yaml
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/granite-8b.yaml
if [ $? -ne 0 ]; then
    echo "Error: Test failed for granite-8b" >&2
    exit -1
fi
echo "Test with granite-8b passed"

# used to check MLA + MOE
echo "Testing GSM8K on deepseek v2 lite"
# deepseek-R1
echo VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/DeepSeek-V2-Lite-chat.yaml
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/DeepSeek-V2-Lite-chat.yaml
if [ $? -ne 0 ]; then
    echo "Error: Test failed for deepseek R1" >&2
    exit -1
fi
echo "Test with deepseek R1 passed"

# used to check HPUATTN + MOE + ExpertParallel
echo "Testing GSM8K on QWEN3-30B-A3B"
echo VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=2 \
pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/Qwen3-30B-A3B.yaml
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=2 \
pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/Qwen3-30B-A3B.yaml
if [ $? -ne 0 ]; then
    echo "Error: Test failed for QWEN3-30B-A3B" >&2
    exit -1
fi
echo "Test with QWEN3-30B-A3B passed"

# multimodal-support with qwen2.5-vl
echo "Testing Qwen2.5-VL-7B"
echo "VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
python -u vllm-gaudi/tests/models/language/generation/generation_mm.py --model-card-path vllm-gaudi/tests/full_tests/model_cards/qwen2.5-vl-7b.yaml"
VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
python -u vllm-gaudi/tests/models/language/generation/generation_mm.py --model-card-path vllm-gaudi/tests/full_tests/model_cards/qwen2.5-vl-7b.yaml
if [ $? -ne 0 ]; then
    echo "Error: Test failed for multimodal-support with qwen2.5-vl-7b" >&2
    exit -1
fi
echo "Test with multimodal-support with qwen2.5-vl-7b passed"