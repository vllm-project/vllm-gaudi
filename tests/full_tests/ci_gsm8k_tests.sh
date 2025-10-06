#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# This ensures that if any test fails, the script will stop.
set -e

# --- Configuration ---
# Defines the path to the vllm-gaudi directory.
# All test functions will use this variable.
VLLM_GAUDI_PREFIX=${VLLM_GAUDI_PREFIX:-"vllm-gaudi"}
echo $VLLM_GAUDI_PREFIX

# Gemma3 with image input
run_gemma3_test() {
    echo "➡️ Testing gemma-3-4b-it..."
    VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/generation_mm.py" --model-card-path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/gemma-3-4b-it.yaml"
    echo "✅ Test with multimodal-support with gemma-3-4b-it passed."
}

# Basic model test
run_basic_model_test() {
    echo "➡️ Testing basic model with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model facebook/opt-125m
    echo "✅ Test with basic model passed."
}

# Tensor parallel size 2
run_tp2_test() {
    echo "➡️ Testing tensor parallel size 2 with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model facebook/opt-125m --tensor-parallel-size 2
    echo "✅ Test with tensor parallel size 2 passed."
}

# MLA and MoE test
run_mla_moe_test() {
    echo "➡️ Testing MLA and MoE with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code
    echo "✅ Test with deepseek v2 lite passed."
}

# Granite + INC test
run_granite_inc_test() {
    echo "➡️ Testing granite-8b + inc with vllm-hpu plugin v1..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_unit_scale_quant.json" \
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model ibm-granite/granite-3.3-2b-instruct --trust-remote-code --quantization inc --kv_cache_dtype fp8_inc
    echo "✅ Test with granite + inc passed."
}

# Deepseek v2 + INC test
run_deepseek_v2_inc_test() {
    echo "➡️ Testing deepseek_v2 + inc with vllm-hpu plugin v1..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_unit_scale_quant.json" \
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --quantization inc --kv_cache_dtype fp8_inc
    echo "✅ Test with deepseek_v2 + inc passed."
}

# Deepseek v2 + INC + dynamic quantization + TP2
run_deepseek_v2_inc_dynamic_tp2_test() {
    echo "➡️ Testing deepseek_v2 + inc dynamic quantization + tp2..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_dynamic_quant.json" \
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --quantization inc --tensor-parallel-size 2
    echo "✅ Test with deepseek_v2 + inc dynamic quantization + tp2 successful."
}

# Qwen3-8B-FP8 + INC requant
run_qwen3_inc_dynamic_test() {
    echo "➡️ Testing Qwen3-8B-FP8 + inc requant FP8 model + dynamic quant..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_dynamic_quant.json" VLLM_HPU_FORCE_CHANNEL_FP8=false \
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
    python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Qwen/Qwen3-8B-FP8 --trust-remote-code
    echo "✅ Test with Qwen3-8B-FP8 + inc requant FP8 model + dynamic quant passed."
}

# QWEN3 + blockfp8 + dynamic scaling
run_qwen3_blockfp8_dynamic_scaling_test() {
    echo "➡️ Testing Qwen3-8B-FP8 + blockfp8 + dynamic scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Qwen/Qwen3-8B-FP8 --trust-remote-code
    echo "✅ Test with Qwen3-8B-FP8 + blockfp8 + dynamic scaling successful."
}

# QWEN3 compressed tensor + dynamic scaling
run_qwen3_compressed_tensor_dynamic_scaling_test() {
    echo "➡️ Testing Qwen3-8B-FP8-dynamic + compressed-tensor + dynamic scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model RedHatAI/Qwen3-8B-FP8-dynamic --trust-remote-code
    echo "✅ Test with Qwen3-8B-FP8-dynamic + compressed-tensor + dynamic scaling successful."
}

# QWEN3 FP8 + MOE compressed tensor + dynamic scaling
run_qwen3_moe_compressed_tensor_dynamic_scaling_test() {
    echo "➡️ Testing Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 + moe + compressed-tensor + dynamic scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --trust-remote-code
    echo "✅ Test with Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 + moe + compressed-tensor + dynamic scaling successful."
}

# inc calibration and quantization of granite.
# quantization test must run after calibration test as it is using files generated by calibration test.
run_granite_inc_calibration_and_quantization_test() {
    echo "Testing inc calibration on granite"
    QUANT_CONFIG=${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_measure.json VLLM_CONTIGUOUS_PA=False HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 \
    VLLM_USE_V1=1 python -u ${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py --model ibm-granite/granite-3.3-2b-instruct --trust-remote-code --quantization inc
    if [ $? -ne 0 ]; then
        echo "Error: Test failed for inc calibration on granite" >&2
        exit -1
    fi
    echo "Test with inc calibration on granite passed"

    echo "Testing inc quantization with hw aligned scales on granite"
    QUANT_CONFIG=${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_maxabs_hw_quant.json VLLM_CONTIGUOUS_PA=False HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 \
    VLLM_USE_V1=1 python -u ${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py --model ibm-granite/granite-3.3-2b-instruct --trust-remote-code --quantization inc --kv_cache_dtype fp8_inc
    if [ $? -ne 0 ]; then
        echo "Error: Test failed for inc quantization with hw aligned scales on granite" >&2
        exit -1
    fi
    echo "Test with inc calibration and quantization with hw aligned scales on granite passed"
}

# Structured output
run_structured_output_test() {
    echo "➡️ Testing structured output..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/structured_outputs.py"
    HABANA_VISIBLE_DEVICES=all VLLM_MERGED_PREFILL=True VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/structured_outputs.py"
    echo "✅ Test with structured outputs passed."
}

# AWQ test
run_awq_test() {
    echo "➡️ Testing awq inference with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model TheBloke/Llama-2-7B-Chat-AWQ --dtype bfloat16 --quantization awq_hpu
    echo "✅ Test with awq passed."
}

# GPTQ test
run_gptq_test() {
    echo "➡️ Testing gptq inference with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model TheBloke/Llama-2-7B-Chat-GPTQ --dtype bfloat16 --quantization gptq_hpu
    echo "✅ Test with gptq passed."
}

# Compressed w4a16 channelwise
run_compressed_w4a16_channelwise_test() {
    echo "➡️ Testing compressed w4a16 (channelwise) inference..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model nm-testing/tinyllama-oneshot-w4a16-channel-v2 --dtype bfloat16
    echo "✅ Test with compressed w4a16 (channelwise) passed."
}

# Compressed w4a16 MoE with g_idx
run_compressed_w4a16_moe_gidx_test() {
    echo "➡️ Testing compressed w4a16 MoE with g_idx inference..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model nm-testing/test-w4a16-mixtral-actorder-group --dtype bfloat16
    echo "✅ Test with compressed w4a16 MoE with g_idx passed."
}

# GSM8K on granite-8b
run_gsm8k_granite_test() {
    echo "➡️ Testing GSM8K on granite-8b..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/granite-8b.yaml"
    echo "✅ Test with granite-8b passed."
}

# GSM8K on granite-8b with async scheduling
run_gsm8k_granite_async_test() {
    echo "➡️ Testing GSM8K on granite-8b with async scheduling..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 ASYNC_SCHEDULING=1 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/granite-8b.yaml"
    echo "✅ Test with granite-8b + async_scheduling passed."
}

# GSM8K on deepseek v2 lite
run_gsm8k_deepseek_test() {
    echo "➡️ Testing GSM8K on deepseek v2 lite..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/DeepSeek-V2-Lite-chat.yaml"
    echo "✅ Test with deepseek R1 passed."
}

# GSM8K on QWEN3-30B-A3B
run_gsm8k_qwen3_30b_test() {
    echo "➡️ Testing GSM8K on QWEN3-30B-A3B..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=2 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/Qwen3-30B-A3B.yaml"
    echo "✅ Test with QWEN3-30B-A3B passed."
}

# Multimodal-support with qwen2.5-vl
run_qwen2_5_vl_test() {
    echo "➡️ Testing Qwen2.5-VL-7B..."
    VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
    python -u "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/generation_mm.py" --model-card-path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/qwen2.5-vl-7b.yaml"
    echo "✅ Test with multimodal-support with qwen2.5-vl-7b passed."
}

# Spec decode with ngram
run_spec_decode_ngram_test() {
    echo "➡️ Testing Spec-decode with ngram..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task ngram --assert_acc_rate 0.25 --osl 1024
    echo "✅ Test with spec decode with ngram passed."
}

# NOTE(Chendi): Failed due upstream, expect fix by SW-241408
# Embedding-model-support for v1
run_embedding_model_test() {
    echo "➡️ Testing Embedding-model-support for v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/pooling.py" --model intfloat/e5-mistral-7b-instruct --trust-remote-code
    echo "✅ Embedding-model-support for v1 successful."
}


# --- Script Entry Point ---

# --- Utility Functions ---

# Function to run all tests sequentially
launch_all_tests() {
    echo "🚀 Starting all test suites..."
    run_gemma3_test
    run_basic_model_test
    run_tp2_test
    run_mla_moe_test
    run_granite_inc_test
    run_deepseek_v2_inc_test
    run_deepseek_v2_inc_dynamic_tp2_test
    run_qwen3_inc_dynamic_test
    run_qwen3_blockfp8_dynamic_scaling_test
    run_qwen3_compressed_tensor_dynamic_scaling_test
    run_qwen3_moe_compressed_tensor_dynamic_scaling_test
    run_structured_output_test
    run_awq_test
    run_gptq_test
    run_compressed_w4a16_channelwise_test
    run_compressed_w4a16_moe_gidx_test
    run_gsm8k_granite_test
    run_gsm8k_granite_async_test
    run_gsm8k_deepseek_test
    run_gsm8k_qwen3_30b_test
    run_qwen2_5_vl_test
    run_spec_decode_ngram_test
    #run_embedding_model_test
    echo "🎉 All test suites passed successfully!"
}


# A simple usage function to guide the user
usage() {
  echo "Usage: $0 [function_name]"
  echo "If no function_name is provided, all tests will be run."
  echo ""
  echo "Available functions:"
  # This command lists all functions in this script that start with 'run_'
  declare -F | awk '{print "  - " $3}' | grep --color=never "run_"
}


# --- Script Entry Point ---

# Default to 'run_all_tests' if no function name is provided as an argument.
# The ${1:-run_all_tests} syntax means "use $1 if it exists, otherwise use 'run_all_tests'".
FUNCTION_TO_RUN=${1:-launch_all_tests}

# Check if the provided argument corresponds to a declared function in this script.
if declare -f "$FUNCTION_TO_RUN" > /dev/null
then
  # If the function exists, call it.
  "$FUNCTION_TO_RUN"
else
  # If the function doesn't exist, show an error and the usage guide.
  echo "❌ Error: Function '${FUNCTION_TO_RUN}' is not defined."
  echo ""
  usage
  exit 1
fi


