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
echo QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_dynamic_quant.json HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code  --quantization inc --tensor-parallel-size 2
QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_dynamic_quant.json \
HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --quantization inc --tensor-parallel-size 2
if [ $? -ne 0 ]; then
    echo "Error: Test failed for deepseek_v2 + inc dynamic quantization + tp2" >&2
    exit -1
fi
echo "Test with deepseek_v2 + inc dynamic quantization + tp 2 successful"

echo "Testing Qwen3-8B-FP8 + inc requant FP8 model + dynamic quant"
echo VLLM_HPU_FORCE_CHANNEL_FP8=false QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_dynamic_quant.json HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model Qwen/Qwen3-8B-FP8 --trust-remote-code 
QUANT_CONFIG=vllm-gaudi/tests/models/language/generation/inc_dynamic_quant.json VLLM_HPU_FORCE_CHANNEL_FP8=false  \
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
    python -u vllm-gaudi/tests/full_tests/generate.py --model Qwen/Qwen3-8B-FP8 --trust-remote-code 
if [ $? -ne 0 ]; then
    echo "Error: Test failed for Qwen3-8B-FP8 + inc requant FP8 model + dynamic quant" >&2
    exit -1
fi
echo "Test with Qwen3-8B-FP8 + inc requant FP8 model + dynamic quant passed"

# QWEN3 + blockfp8 + dynamic scaling
echo "Testing Qwen3-8B-FP8 + blockfp8 + dynamic scaling"
echo HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model Qwen/Qwen3-8B-FP8 --trust-remote-code
HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model Qwen/Qwen3-8B-FP8 --trust-remote-code
if [ $? -ne 0 ]; then
    echo "Error: Test failed for Qwen3-8B-FP8 + blockfp8 + dynamic scaling" >&2
    exit -1
fi
echo "Test with Qwen3-8B-FP8 + blockfp8 + dynamic scaling successful"

# QWEN3 compressed tensor + dynamic scaling
echo "Testing Qwen3-8B-FP8-dynamic + compressed-tensor + dynamic scaling"
echo HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model RedHatAI/Qwen3-8B-FP8-dynamic --trust-remote-code
HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/generate.py --model RedHatAI/Qwen3-8B-FP8-dynamic --trust-remote-code
if [ $? -ne 0 ]; then
    echo "Error: Test failed for Qwen3-8B-FP8-dynamic + compressed-tensor + dynamic scaling" >&2
    exit -1
fi
echo "Test with Qwen3-8B-FP8-dynamic + compressed-tensor + dynamic scaling successful"

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

# used to check asynchronous scheduling
echo "Testing GSM8K on ganite-8b with async scheduling"
echo VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 ASYNC_SCHEDULING=1 \
pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/granite-8b.yaml
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 ASYNC_SCHEDULING=1 \
pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/granite-8b.yaml
if [ $? -ne 0 ]; then
    echo "Error: Test failed for granite-8b + async_scheduling" >&2
    exit -1
fi
echo "Test with granite-8b + async_scheduling passed"

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
# echo "Testing GSM8K on QWEN3-30B-A3B"
# echo VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=2 \
# pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/Qwen3-30B-A3B.yaml
# VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=2 \
# pytest -v -s vllm-gaudi/tests/models/language/generation/test_common.py --model_card_path vllm-gaudi/tests/full_tests/model_cards/Qwen3-30B-A3B.yaml
# if [ $? -ne 0 ]; then
#     echo "Error: Test failed for QWEN3-30B-A3B" >&2
#     exit -1
# fi
# echo "Test with QWEN3-30B-A3B passed"

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

# spec decode with ngram
# For G3, acc rate is 0.18, but for G2, it is 0.09
echo "Testing Spec-decode with ngram"
echo VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 python vllm-gaudi/tests/full_tests/spec_decode.py --task ngram --assert_acc_rate 0.25 --osl 1024
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 python vllm-gaudi/tests/full_tests/spec_decode.py --task ngram --assert_acc_rate 0.25 --osl 1024
if [ $? -ne 0 ]; then
    echo "Error: Test failed for spec decode with ngram" >&2
    exit -1
fi
echo "Test with spec decode with ngram passed"

# Embedding-model-support for v1
echo "Testing Embedding-model-support for v1"
echo HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/pooling.py --model intfloat/e5-mistral-7b-instruct --trust-remote-code
HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/tests/full_tests/pooling.py --model intfloat/e5-mistral-7b-instruct --trust-remote-code
if [ $? -ne 0 ]; then
    echo "Error: Test failed for Embedding-model-support for v1" >&2
    exit -1
fi
echo "Embedding-model-support for v1 successful"

# Data Parallel failed with recent upstream changes
# # DP2
# echo "Testing data parallel size 2 with vllm-hpu plugin v1"
# echo HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/examples/data_parallel.py --dp-size 2 --tp-size 2
# HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u vllm-gaudi/examples/data_parallel.py --dp-size 2 --tp-size 2
# if [ $? -ne 0 ]; then
#     echo "Error: Test failed for data parallel size 2" >&2
#     exit -1
# fi
# echo "Test with data parallel size 2 passed"
