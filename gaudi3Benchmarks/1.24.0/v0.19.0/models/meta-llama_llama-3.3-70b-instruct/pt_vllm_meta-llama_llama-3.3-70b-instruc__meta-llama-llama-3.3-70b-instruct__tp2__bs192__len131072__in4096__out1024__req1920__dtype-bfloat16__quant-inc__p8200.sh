#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_meta-llama_Llama-3.3-70B-Instruct_1_2
# model=meta-llama/Llama-3.3-70B-Instruct
# tensor_parallel_size=2
# batch_size=192
# max_model_len=131072
# dtype=bfloat16
# quant_config=/software/ae/fmwork/inc/1.21.0/llama-3.3-70b-instruct-2x/maxabs_quant_g3.json
# input_len=4096
# output_len=1024
# num_prompts=1920
# max_num_batched_tokens=na
# port=8200

export QUANT_CONFIG=/software/ae/fmwork/inc/1.21.0/llama-3.3-70b-instruct-2x/maxabs_quant_g3.json
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
VLLM_FUSED_BLOCK_SOFTMAX=true \
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_HANDLE_TOPK_DUPLICATES=true \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
vllm serve \
    --model=meta-llama/Llama-3.3-70B-Instruct \
    --port 8200 \
    --max-num-seqs=192 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.90 \
    --tensor-parallel-size=2 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --quantization inc \
    --kv-cache-dtype fp8_inc \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    llama3_json \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8200.json <<'JSON'
{
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": "What is 2+2? Use the tool and return the final answer."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
          "type": "object",
          "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
          },
          "required": ["a", "b"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0,
  "max_tokens": 256,
  "logprobs": true
}
JSON

curl -sS http://localhost:8200/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8200.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dataset-name random \
    --num-prompts 1920 \
    --max-concurrency 192 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --port 8200 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 600 \
    --ignore-eos \
    --trust-remote-code
