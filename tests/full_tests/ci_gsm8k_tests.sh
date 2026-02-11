#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
# This ensures that if any test fails, the script will stop.
set -e

# --- Configuration ---
# Defines the path to the vllm-gaudi directory.
# All test functions will use this variable.
VLLM_GAUDI_PREFIX=${VLLM_GAUDI_PREFIX:-"vllm-gaudi"}

# --- Execution Mode ---
# PT_HPU_LAZY_MODE controls the HPU execution mode:
#   1 = lazy mode (graph-based compilation)
#   0 = eager mode (immediate execution, default)
#
# The mode is inherited from the environment. If not set, defaults to eager (0).
# CI wrappers and workflows set this variable before invoking the script.
export PT_HPU_LAZY_MODE="${PT_HPU_LAZY_MODE:-0}"

# --- Mode-Specific Test Declarations ---
# By default, all tests run in BOTH lazy and eager modes.
# Tests that can ONLY run in a specific mode must be listed in the
# corresponding associative array below.
#
# When adding a new test:
#   - If it works in both modes: no action needed (default behavior).
#   - If it only works in eager: add it to EAGER_ONLY_TESTS.
#   - If it only works in lazy:  add it to LAZY_ONLY_TESTS.
declare -A EAGER_ONLY_TESTS=(
    ["run_dsv2_blockfp8_static_scaling_fp8kv_test"]=1
    ["run_qwen3_8b_fp8_attn_static_scaling_fp8kv_test"]=1
    ["run_dsv2_blockfp8_static_scaling_fp8qkv_test"]=1
    ["run_qwen3_vl_test"]=1
    ["run_llama3_70b_inc_dynamic_quant_test"]=1
    ["run_sleep_mode_test"]=1
)

declare -A LAZY_ONLY_TESTS=(
    # Currently none. Available for future use if a test only works in lazy mode.
)

# --- Mode Utility Functions ---

# Returns the current execution mode as a human-readable string.
get_current_mode() {
    if [[ "${PT_HPU_LAZY_MODE}" == "1" ]]; then
        echo "lazy"
    else
        echo "eager"
    fi
}

# Returns the required mode for a given test function: "eager", "lazy", or "both".
get_test_required_mode() {
    local test_name="$1"
    if [[ -n "${EAGER_ONLY_TESTS[$test_name]+_}" ]]; then
        echo "eager"
    elif [[ -n "${LAZY_ONLY_TESTS[$test_name]+_}" ]]; then
        echo "lazy"
    else
        echo "both"
    fi
}

# Returns 0 (true) if the test is compatible with the current execution mode.
is_test_compatible() {
    local test_name="$1"
    local current_mode required_mode
    current_mode=$(get_current_mode)
    required_mode=$(get_test_required_mode "$test_name")
    [[ "$required_mode" == "both" ]] || [[ "$required_mode" == "$current_mode" ]]
}

# Print execution mode info only when running tests (not during discovery).
# The entry point at the bottom of this script handles this.
_print_mode_info() {
    echo "Execution mode: $(get_current_mode) (PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE})"
    echo "VLLM_GAUDI_PREFIX: ${VLLM_GAUDI_PREFIX}"
}

# Gemma3 with image input
run_gemma3_test() {
    echo "➡️ Testing gemma-3-4b-it..."
    VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/generation_mm.py" --model-card-path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/gemma-3-4b-it.yaml"
    echo "✅ Test with multimodal-support with gemma-3-4b-it passed."
    echo "➡️ Testing gemma-3-4b-it with multiple images(applying sliding_window)..."
    VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/generation_mm_multi.py" --model-card-path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/gemma-3-27b-it.yaml"
    echo "✅ Test with multimodal-support with multiple images gemma-3-27b-it passed."
}

# Basic model test
run_basic_model_test() {
    echo "➡️ Testing basic model with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model facebook/opt-125m
    echo "✅ Test with basic model passed."
}

# Tensor parallel size 2
run_tp2_test() {
    echo "➡️ Testing tensor parallel size 2 with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model facebook/opt-125m --tensor-parallel-size 2
    echo "✅ Test with tensor parallel size 2 passed."
}

# MLA and MoE test
run_mla_moe_test() {
    echo "➡️ Testing MLA and MoE with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code
    echo "✅ Test with deepseek v2 lite passed."
}

# Granite + INC test
run_granite_inc_test() {
    echo "➡️ Testing granite-8b + inc with vllm-hpu plugin v1..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_unit_scale_quant.json" \
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model ibm-granite/granite-3.3-2b-instruct --trust-remote-code --quantization inc --kv_cache_dtype fp8_inc
    echo "✅ Test with granite + inc passed."
}

# Deepseek v2 + INC test
run_deepseek_v2_inc_test() {
    echo "➡️ Testing deepseek_v2 + inc with vllm-hpu plugin v1..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_unit_scale_quant.json" \
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --quantization inc --kv_cache_dtype fp8_inc
    echo "✅ Test with deepseek_v2 + inc passed."
}

# Deepseek v2 + INC + dynamic quantization + TP2
run_deepseek_v2_inc_dynamic_tp2_test() {
    echo "➡️ Testing deepseek_v2 + inc dynamic quantization + tp2..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_dynamic_quant.json" \
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code --quantization inc --tensor-parallel-size 2
    echo "✅ Test with deepseek_v2 + inc dynamic quantization + tp2 successful."
}

# Qwen3-8B-FP8 + INC requant
run_qwen3_inc_dynamic_test() {
    echo "➡️ Testing Qwen3-8B-FP8 + inc requant FP8 model + dynamic quant..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_dynamic_quant.json" VLLM_HPU_FORCE_CHANNEL_FP8=false \
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true \
    python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Qwen/Qwen3-8B-FP8 --trust-remote-code
    echo "✅ Test with Qwen3-8B-FP8 + inc requant FP8 model + dynamic quant passed."
}

# DS + blockfp8 + static scaling + FP8 KV
# The lazy mode works on 1.24.0-272
run_dsv2_blockfp8_static_scaling_fp8kv_test() {
    echo "➡️ Testing Deepseek-V2-Lite-Chat-FP8 + blockfp8 + static scaling + FP8 KV..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model INC4AI/DeepSeek-V2-Lite-Chat-BF16-FP8-STATIC-FP8-KV-TEST-ONLY --trust-remote-code
    echo "✅ Test with Deepseek-V2-Lite-Chat-FP8 + blockfp8 + static scaling + FP8 KV successful."
}

# QWEN3 + FP8 Attn(FP8 QGA test)
# The lazy mode works on 1.24.0-272
run_qwen3_8b_fp8_attn_static_scaling_fp8kv_test() {
    echo "➡️ Testing Qwen3-8B + static scaling + FP8 Attn..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model INC4AI/Qwen3-8B-FP8_STATIC-FP8-Attn-LLMC-Test-Only --trust-remote-code --kv_cache_dtype fp8_inc
    echo "✅ Test with Qwen3-8B + static scaling + FP8 Attn successful."
}

# DS + blockfp8 + static scaling + FP8 QKV
# The lazy mode works on 1.24.0-272
run_dsv2_blockfp8_static_scaling_fp8qkv_test() {
    echo "➡️ Testing Deepseek-V2-Lite-Chat-FP8 + blockfp8 + static scaling + FP8 QKV..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Intel/DeepSeek-V2-Lite-Chat-BF16-FP8-STATIC-FP8-QKV-TEST-ONLY --trust-remote-code --kv_cache_dtype fp8_inc
    echo "✅ Test with Deepseek-V2-Lite-Chat-FP8 + blockfp8 + static scaling + FP8 QKV successful."
}

# QWEN3 + blockfp8 + dynamic scaling
run_qwen3_blockfp8_dynamic_scaling_test() {
    echo "➡️ Testing Qwen3-8B-FP8 + blockfp8 + dynamic scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Qwen/Qwen3-8B-FP8 --trust-remote-code
    echo "✅ Test with Qwen3-8B-FP8 + blockfp8 + dynamic scaling successful."
}

# QWEN3 compressed tensor + dynamic scaling
run_qwen3_compressed_tensor_dynamic_scaling_test() {
    echo "➡️ Testing Qwen3-8B-FP8-dynamic + compressed-tensor + dynamic scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model RedHatAI/Qwen3-8B-FP8-dynamic --trust-remote-code
    echo "✅ Test with Qwen3-8B-FP8-dynamic + compressed-tensor + dynamic scaling successful."
}

# QWEN3 FP8 + MOE compressed tensor + dynamic scaling
run_qwen3_moe_compressed_tensor_dynamic_scaling_test() {
    echo "➡️ Testing Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 + moe + compressed-tensor + dynamic scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --trust-remote-code --max-model-len 131072
    echo "✅ Test with Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 + moe + compressed-tensor + dynamic scaling successful."
}

# QWEN3 FP8 + MOE compressed tensor + static scaling (weight per-tensor, activation per-tensor)
run_qwen3_moe_compressed_tensor_static_per_tensor_scaling_test() {
    echo "▒~^▒▒~O Testing Intel/Qwen3-30B-A3B-FP8-Test-Only + moe + compressed-tensor + static scaling..."
    #HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Intel/Qwen3-30B-A3B-FP8-Test-Only --trust-remote-code --no-enforce-eager --enable-expert-parallel
    echo "▒~\~E Test with Intel/Qwen3-30B-A3B-FP8-Test-Only + moe + compressed-tensor + static scaling successful."
}

# QWEN3 FP8 + MOE compressed tensor + static scaling (weight per-channel, activation per-tensor)
run_qwen3_moe_compressed_tensor_static_scaling_test() {
    echo "▒~^▒▒~O Testing Intel/Qwen3-30B-A3B-FP8-Static-Test-Only + moe + compressed-tensor + static scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model Intel/Qwen3-30B-A3B-FP8-Static-Test-Only --trust-remote-code --no-enforce-eager --enable-expert-parallel
    echo "▒~\~E Test with Intel/Qwen3-30B-A3B-FP8-Static-Test-Only + moe + compressed-tensor + static scaling successful."
}

# RedHatAI/Meta-Llama-3-8B-Instruct-FP8 Per-tensor F8 static scales
run_llama3_per_tensor_scaling_test() {
    echo "➡️ Testing RedHatAI/Meta-Llama-3-8B-Instruct-FP8 + per tensor scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model RedHatAI/Meta-Llama-3-8B-Instruct-FP8 --trust-remote-code
    echo "✅ Test with RedHatAI/Meta-Llama-3-8B-Instruct-FP8 + per tensor scaling successful."
}

# nvidia/Llama-3.1-8B-Instruct-FP8 Per-tensor F8 static scales
run_llama3_modelopt_per_tensor_scaling_test() {
    echo "➡️ Testing nvidia/Llama-3.1-8B-Instruct-FP8 + per tensor scaling..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model nvidia/Llama-3.1-8B-Instruct-FP8 --trust-remote-code --kv_cache_dtype fp8_inc
    echo "✅ Test with nvidia/Llama-3.1-8B-Instruct-FP8 + per tensor scaling successful."
}


# inc calibration and quantization of granite.
# quantization test must run after calibration test as it is using files generated by calibration test.
run_granite_inc_calibration_and_quantization_test() {
    echo "Testing inc calibration on granite"
    QUANT_CONFIG=${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_measure.json VLLM_CONTIGUOUS_PA=False HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=True \
    python -u ${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py --model ibm-granite/granite-3.3-2b-instruct --trust-remote-code --quantization inc
    if [ $? -ne 0 ]; then
        echo "Error: Test failed for inc calibration on granite" >&2
        exit -1
    fi
    echo "Test with inc calibration on granite passed"

    echo "Testing inc quantization with hw aligned scales on granite"
    QUANT_CONFIG=${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_maxabs_hw_quant.json VLLM_CONTIGUOUS_PA=False HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=True \
    python -u ${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py --model ibm-granite/granite-3.3-2b-instruct --trust-remote-code --quantization inc --kv_cache_dtype fp8_inc
    if [ $? -ne 0 ]; then
        echo "Error: Test failed for inc quantization with hw aligned scales on granite" >&2
        exit -1
    fi
    echo "Test with inc calibration and quantization with hw aligned scales on granite passed"
}

# Structured output
run_structured_output_test() {
    echo "➡️ Testing structured output..."
    HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/structured_outputs.py"
    HABANA_VISIBLE_DEVICES=all VLLM_MERGED_PREFILL=True VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/structured_outputs.py"
    echo "✅ Test with structured outputs passed."
}

# AWQ test
run_awq_test() {
    echo "➡️ Testing awq inference with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model TheBloke/Llama-2-7B-Chat-AWQ --dtype bfloat16 --quantization awq_hpu
    echo "✅ Test with awq passed."
}

# GPTQ test
run_gptq_test() {
    echo "➡️ Testing gptq inference with vllm-hpu plugin v1..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model TheBloke/Llama-2-7B-Chat-GPTQ --dtype bfloat16 --quantization gptq_hpu
    echo "✅ Test with gptq passed."
}

# Compressed w4a16 channelwise
run_compressed_w4a16_channelwise_test() {
    echo "➡️ Testing compressed w4a16 (channelwise) inference..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model nm-testing/tinyllama-oneshot-w4a16-channel-v2 --dtype bfloat16
    echo "✅ Test with compressed w4a16 (channelwise) passed."
}

# Compressed w4a16 MoE with g_idx
run_compressed_w4a16_moe_gidx_test() {
    echo "➡️ Testing compressed w4a16 MoE with g_idx inference..."
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model nm-testing/test-w4a16-mixtral-actorder-group --dtype bfloat16
    echo "✅ Test with compressed w4a16 MoE with g_idx passed."
}

# Llama-3.3-70B-Instruct-FP8-dynamic + INC dynamic quant
run_llama3_70b_inc_dynamic_quant_test() {
    echo "➡️ Testing Llama-3.3-70B-Instruct-FP8-dynamic + inc dynamic quant in torch.compile mode ..."
    QUANT_CONFIG="${VLLM_GAUDI_PREFIX}/tests/models/language/generation/inc_maxabs_dynamic_quant.json" \
    HABANA_VISIBLE_DEVICES=all RUNTIME_SCALE_PATCHING=0 VLLM_SKIP_WARMUP=true \
    python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/generate.py" --model RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic --max-model-len 2048
    echo "✅ Test with Llama-3.3-70B-Instruct-FP8-dynamic + inc dynamic quant in torch.compile mode passed."
}

# GSM8K on granite-8b
run_gsm8k_granite_test() {
    echo "➡️ Testing GSM8K on granite-8b..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/granite-8b.yaml"
    echo "✅ Test with granite-8b passed."
}

# GSM8K on granite-8b (unified attn)
run_gsm8k_granite_test_unified_attn() {
    echo "➡️ Testing GSM8K on granite-8b with unified attention..."
    VLLM_UNIFIED_ATTN=True VLLM_SKIP_WARMUP=True \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/granite-8b.yaml"
    echo "✅ Test with granite-8b unified attention passed."
}

# GSM8K on granite-8b with async scheduling
run_gsm8k_granite_async_test() {
    echo "➡️ Testing GSM8K on granite-8b with async scheduling..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True ASYNC_SCHEDULING=1 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/granite-8b.yaml"
    echo "✅ Test with granite-8b + async_scheduling passed."
}

# GSM8K on granite-8b (unified attn + async scheduling)
run_gsm8k_granite_test_unified_attn_async() {
    echo "➡️ Testing GSM8K on granite-8b with unified attention + async scheduling..."
    VLLM_UNIFIED_ATTN=True VLLM_SKIP_WARMUP=True VLLM_USE_V1=1 ASYNC_SCHEDULING=1 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/granite-8b.yaml"
    echo "✅ Test with granite-8b unified attention + async scheduling passed."
}

# GSM8K on deepseek v2 lite
run_gsm8k_deepseek_test() {
    echo "➡️ Testing GSM8K on deepseek v2 lite..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/DeepSeek-V2-Lite-chat.yaml"
    echo "✅ GSM8K Test with deepseek v2 lite passed."
}


# GSM8K on deepseek v2 lite + unified attn
run_gsm8k_deepseek_unified_mla_test() {
    echo "➡️ Testing GSM8K on deepseek v2 lite + Unified MLA..."
    VLLM_UNIFIED_ATTN=true VLLM_SKIP_WARMUP=True \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/DeepSeek-V2-Lite-chat.yaml"
    echo "✅ GSM8K Test with deepseek v2 lite + Unified MLA passed."
}

# GSM8K on QWEN3-30B-A3B
run_gsm8k_qwen3_30b_test() {
    echo "➡️ Testing GSM8K on QWEN3-30B-A3B..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True TP_SIZE=2 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/test_common.py" --model_card_path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/Qwen3-30B-A3B.yaml"
    echo "✅ Test with QWEN3-30B-A3B passed."
}

# Multimodal-support with qwen2.5-vl
run_qwen2_5_vl_test() {
    echo "➡️ Testing Qwen2.5-VL-7B..."
    VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False \
    python -u "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/generation_mm.py" --model-card-path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/qwen2.5-vl-7b.yaml"
    echo "✅ Test with multimodal-support with qwen2.5-vl-7b passed."
}

# Multimodal-support + unified attention with qwen2.5-vl
run_qwen2_5_vl_unified_attn_test() {
    echo "➡️ Testing Qwen2.5-VL-7B with unified attention..."
    VLLM_SKIP_WARMUP=true VLLM_UNIFIED_ATTN=True VLLM_USE_V1=1 \
    python -u "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/generation_mm.py" --model-card-path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/qwen2.5-vl-7b.yaml"
    echo "✅ Test multimodal-support + unified attention with qwen2.5-vl-7b passed."
}

# Multimodal-support with qwen3-vl
run_qwen3_vl_test() {
    echo "➡️ Testing Qwen3-VL-32B..."
    VLLM_SKIP_WARMUP=true VLLM_CONTIGUOUS_PA=False \
    python -u "${VLLM_GAUDI_PREFIX}/tests/models/language/generation/generation_mm.py" --model-card-path "${VLLM_GAUDI_PREFIX}/tests/full_tests/model_cards/qwen3-vl-32b.yaml"
    echo "✅ Test with multimodal-support with qwen3-vl-32b passed."
}

# Spec decode with ngram
run_spec_decode_ngram_test() {
    echo "➡️ Testing Spec-decode with ngram..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task ngram --assert_accept_rate 0.25 --osl 1024
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task ngram --accuracy_rate 0.75
    echo "✅ Test with spec decode with ngram passed."
}

# Spec decode with eagle3
run_spec_decode_eagle3_test() {
    echo "➡️ Testing Spec-decode with eagle3..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task eagle3 --assert_accept_rate 0.70 --osl 2048
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task eagle3 --accuracy_rate 0.65
    echo "✅ Test with spec decode with eagle3 passed."
}

# Spec decode with eagle3 and num_speculative_tokens = 2
run_spec_decode_eagle3_num_spec_2_test() {
    echo "➡️ Testing Spec-decode with eagle3 and num_speculative_tokens = 2..."
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task eagle3 --assert_accept_rate 0.59 --osl 2048 --num_spec_tokens 2
    VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task eagle3 --accuracy_rate 0.59 --num_spec_tokens 2
    echo "✅ Test with spec decode with eagle3 and num_speculative_tokens = 2 passed."
}

# Spec decode with ngram with UA
run_UA_spec_decode_ngram_test() {
    echo "➡️ Testing Spec-decode with ngram..."
    VLLM_UNIFIED_ATTN=True VLLM_SKIP_WARMUP=True python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task ngram --assert_accept_rate 0.25 --osl 512
    echo "✅ Test with spec decode with ngram passed."
}

# Spec decode with eagle3 with UA
run_UA_spec_decode_eagle3_test() {
    echo "➡️ Testing Spec-decode with eagle3..."
    VLLM_UNIFIED_ATTN=True VLLM_SKIP_WARMUP=True python "${VLLM_GAUDI_PREFIX}/tests/full_tests/spec_decode.py" --task eagle3 --assert_accept_rate 0.50 --osl 1024
    echo "✅ Test with spec decode with eagle3 passed."
}

# Embedding-model-support for v1
run_embedding_model_test() {
   echo "➡️ Testing Embedding-model-support for v1..."
   HABANA_VISIBLE_DEVICES=all VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=false python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/pooling.py" --model intfloat/e5-mistral-7b-instruct --trust-remote-code
   echo "✅ Embedding-model-support for v1 successful."
}

# pd_disaggregate_nixl_libfabric
run_pd_disaggregate_nixl_libfabric_test() {
    echo "➡️ Testing PD disaggregate through NIXL libfabric."
    git clone https://github.com/intel-staging/nixl.git -b v0.6.0_OFI
    cp -r nixl /tmp/nixl_source
    cd nixl; WHEELS_CACHE_HOME=/workspace/hf_cache/wheels_cache_ofi python install_nixl.py; cd ..
    rm -rf nixl
    cd ${VLLM_GAUDI_PREFIX}/tests/unit_tests; DECODER_TP_SIZE=1 NIXL_BUFFER_DEVICE=hpu VLLM_NIXL_BACKEND=OFI bash run_accuracy_test.sh
    echo "✅ PD disaggregate through NIXL libfabric."
}

run_pd_disaggregate_nixl_ucx_test() {
    echo "➡️ Testing PD disaggregate through NIXL UCX."
    WHEELS_CACHE_HOME=/workspace/hf_cache/wheels_cache_ucx python "${VLLM_GAUDI_PREFIX}/install_nixl.py"
    cd ${VLLM_GAUDI_PREFIX}/tests/unit_tests; DECODER_TP_SIZE=1 NIXL_BUFFER_DEVICE=hpu VLLM_NIXL_BACKEND=UCX bash run_accuracy_test.sh
    echo "✅ PD disaggregate through NIXL UCX."
}

# CPU Offloading connector
run_cpu_offloading_test() {
    echo "➡️ Testing CPU offlading."
    VLLM_SKIP_WARMUP=True VLLM_USE_V1=1 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/unit_tests/kv_offload/test_cpu_offloading.py"
    echo "✅ Test CPU offlading passed."
}

run_offloading_connector_test() {
    echo "➡️ Testing OffloadingConnector."
    VLLM_SKIP_WARMUP=True VLLM_USE_V1=1 \
    pytest -v -s "${VLLM_GAUDI_PREFIX}/tests/unit_tests/kv_offload/test_offloading_connector.py"
    echo "✅ Test OffloadingConnector passed."
}

# sleep mode
run_sleep_mode_test() {
    echo "Testing basic model with sleep mode / wake up functionality"
    HABANA_VISIBLE_DEVICES=all VLLM_SKIP_WARMUP=true VLLM_ENABLE_V1_MULTIPROCESSING=0 python -u "${VLLM_GAUDI_PREFIX}/tests/full_tests/sleep_mode.py" --model facebook/opt-125m
    echo "✅ Test with sleep mode passed."
}

# --- Script Entry Point ---

# --- Utility Functions ---

# Function to run all tests sequentially, respecting mode compatibility.
# Tests incompatible with the current mode are automatically skipped.
launch_all_tests() {
    local current_mode
    current_mode=$(get_current_mode)
    echo "🚀 Starting all test suites in ${current_mode} mode (PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE})..."

    local all_tests=(
        run_gemma3_test
        run_basic_model_test
        run_tp2_test
        run_mla_moe_test
        run_granite_inc_test
        run_granite_inc_calibration_and_quantization_test
        run_deepseek_v2_inc_test
        run_deepseek_v2_inc_dynamic_tp2_test
        run_qwen3_inc_dynamic_test
        run_dsv2_blockfp8_static_scaling_fp8kv_test
        run_qwen3_8b_fp8_attn_static_scaling_fp8kv_test
        run_dsv2_blockfp8_static_scaling_fp8qkv_test
        run_qwen3_blockfp8_dynamic_scaling_test
        run_qwen3_compressed_tensor_dynamic_scaling_test
        run_qwen3_moe_compressed_tensor_dynamic_scaling_test
        run_qwen3_moe_compressed_tensor_static_scaling_test
        run_qwen3_moe_compressed_tensor_static_per_tensor_scaling_test
        run_llama3_per_tensor_scaling_test
        run_llama3_modelopt_per_tensor_scaling_test
        run_structured_output_test
        run_awq_test
        run_gptq_test
        run_compressed_w4a16_channelwise_test
        run_compressed_w4a16_moe_gidx_test
        run_llama3_70b_inc_dynamic_quant_test
        run_gsm8k_granite_test
        run_gsm8k_granite_test_unified_attn
        run_gsm8k_granite_async_test
        run_gsm8k_granite_test_unified_attn_async
        run_gsm8k_deepseek_test
        run_gsm8k_deepseek_unified_mla_test
        run_gsm8k_qwen3_30b_test
        run_qwen2_5_vl_test
        run_qwen2_5_vl_unified_attn_test
        run_qwen3_vl_test
        run_spec_decode_ngram_test
        run_spec_decode_eagle3_test
        run_spec_decode_eagle3_num_spec_2_test
        run_UA_spec_decode_ngram_test
        run_UA_spec_decode_eagle3_test
        run_cpu_offloading_test
        run_offloading_connector_test
        run_sleep_mode_test
        #run_embedding_model_test
    )

    local skipped=0
    local executed=0
    for func in "${all_tests[@]}"; do
        # Skip commented-out tests
        [[ "$func" == \#* ]] && continue

        if is_test_compatible "$func"; then
            echo "▶ Running ${func} [$(get_test_required_mode "$func") mode test]..."
            "$func"
            ((executed++))
        else
            echo "⏭ Skipping ${func} (requires $(get_test_required_mode "$func"), current: ${current_mode})"
            ((skipped++))
        fi
    done

    echo "🎉 Finished! Executed: ${executed}, Skipped (mode mismatch): ${skipped}"
}

# --- CI Discovery ---
# Outputs a JSON array of {test_function, mode} objects for GitHub Actions matrix.
# Each test is emitted for every mode it supports.
# Usage: ci_gsm8k_tests.sh discover_matrix
discover_matrix() {
    local script_path="${BASH_SOURCE[0]}"
    local json="["
    local first=true

    while IFS= read -r func; do
        local required_mode
        required_mode=$(get_test_required_mode "$func")

        local modes=()
        case "$required_mode" in
            eager) modes=("eager") ;;
            lazy)  modes=("lazy") ;;
            both)  modes=("lazy" "eager") ;;
        esac

        for mode in "${modes[@]}"; do
            if ! $first; then json+=","; fi
            json+="{\"test_function\":\"${func}\",\"mode\":\"${mode}\"}"
            first=false
        done
    done < <(grep '^run_' "$script_path" | awk '{print $1}' | sed 's/()//')

    json+="]"
    echo "$json"
}

# A simple usage function to guide the user
usage() {
    echo "Usage: $0 [function_name|discover_matrix|launch_all_tests]"
    echo ""
    echo "Commands:"
    echo "  <no argument>       Run all tests compatible with the current mode."
    echo "  <function_name>     Run a single test function (mode compatibility is checked)."
    echo "  discover_matrix     Output a JSON matrix of {test_function, mode} for CI."
    echo "  launch_all_tests    Run all compatible tests sequentially."
    echo ""
    echo "Environment:"
    echo "  PT_HPU_LAZY_MODE    Execution mode: 1=lazy, 0=eager (default: 0)"
    echo ""
    echo "Available test functions:"
    declare -F | awk '{print "  - " $3}' | grep --color=never "run_"
}


# --- Script Entry Point ---

# Default to 'launch_all_tests' if no function name is provided as an argument.
FUNCTION_TO_RUN=${1:-launch_all_tests}

# Check if the provided argument corresponds to a declared function in this script.
if declare -f "$FUNCTION_TO_RUN" > /dev/null; then
    # For run_* and launch_all_tests, print mode info before execution.
    if [[ "$FUNCTION_TO_RUN" == run_* ]] || [[ "$FUNCTION_TO_RUN" == launch_all_tests ]]; then
        _print_mode_info
    fi

    # For run_* functions, validate mode compatibility before execution.
    if [[ "$FUNCTION_TO_RUN" == run_* ]]; then
        if ! is_test_compatible "$FUNCTION_TO_RUN"; then
            echo "⏭ Skipping ${FUNCTION_TO_RUN}: requires $(get_test_required_mode "$FUNCTION_TO_RUN") mode, but current mode is $(get_current_mode)."
            echo "Set PT_HPU_LAZY_MODE appropriately and retry."
            exit 0
        fi
    fi
    "$FUNCTION_TO_RUN"
else
    echo "❌ Error: Function '${FUNCTION_TO_RUN}' is not defined."
    echo ""
    usage
    exit 1
fi
