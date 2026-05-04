#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_mistralai_Mistral-Large-3-675B-Instruct-2512_0_1_2_3_4_5_6_7
# model=mistralai/Mistral-Large-3-675B-Instruct-2512
# tensor_parallel_size=8
# batch_size=8
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=4096
# output_len=1024
# num_prompts=240
# max_num_batched_tokens=na
# port=8260

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=false \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
VLLM_FUSED_BLOCK_SOFTMAX=true \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_PROMPT_BS_BUCKET_MAX=2 \
VLLM_PROMPT_CTX_BUCKET_STEP=32 \
VLLM_PROMPT_QUERY_BUCKET_MIN=1024 \
VLLM_PROMPT_QUERY_BUCKET_STEP=1024 \
VLLM_PROMPT_QUERY_BUCKET_MAX=4096 \
VLLM_DECODE_BS_BUCKET_MAX=8 \
VLLM_DECODE_BLOCK_BUCKET_MIN=256 \
VLLM_DECODE_BLOCK_BUCKET_STEP=256 \
VLLM_DECODE_BLOCK_BUCKET_MAX=8192 \
vllm serve \
    --model=mistralai/Mistral-Large-3-675B-Instruct-2512 \
    --port 8260 \
    --max-num-seqs=8 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.95 \
    --tensor-parallel-size=8 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser \
    mistral \
    --max-num-batched-tokens 4096 \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8260.json <<'JSON'
{
  "model": "mistralai/Mistral-Large-3-675B-Instruct-2512",
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

curl -sS http://localhost:8260/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8260.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model mistralai/Mistral-Large-3-675B-Instruct-2512 \
    --dataset-name random \
    --num-prompts 240 \
    --max-concurrency 8 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --endpoint /v1/completions \
    --host localhost \
    --port 8260 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 600 \
    --ignore-eos \
    --trust-remote-code
