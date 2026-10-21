#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_Qwen_Qwen3-Coder-Next_1_7
# model=Qwen/Qwen3-Coder-Next
# tensor_parallel_size=2
# batch_size=8
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=4096
# output_len=1024
# num_prompts=80
# max_num_batched_tokens=na
# port=8450

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=17200 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=true \
VLLM_BUCKETING_STRATEGY=exp \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=false \
VLLM_DEFRAG=false \
VLLM_FUSED_BLOCK_SOFTMAX=true \
PT_HPU_ENABLE_EAGER_CACHE=true \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
EXPERIMENTAL_WEIGHT_SHARING=0 \
VLLM_SKIP_WARMUP=false \
VLLM_GRAPH_PROMPT_RATIO=0.3 \
ENABLE_EXPERIMENTAL_FLAGS=true \
ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true \
vllm serve \
    --model=Qwen/Qwen3-Coder-Next \
    --port 8450 \
    --max-num-seqs=8 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.60 \
    --tensor-parallel-size=2 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    qwen3_coder \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --enable-chunked-prefill \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8450.json <<'JSON'
{
  "model": "Qwen/Qwen3-Coder-Next",
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
  "temperature": 0.7,
  "max_tokens": 256,
  "logprobs": true,
  "n": 2,
  "repetition_penalty": 1.1,
  "presence_penalty": 0.5,
  "frequency_penalty": 0.5
}
JSON

curl -sS http://localhost:8450/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8450.json




# =========================
# HS TICKET COMPLETIONS VALIDATION COMMAND
# =========================

# =========================
# HS TICKET COMPLETIONS VALIDATION
# =========================

set +e

cat > /tmp/hs_ticket_repro_8450.json <<'JSON'
{
  "model": "Qwen/Qwen3-Coder-Next",
  "prompt": "Please answer the following question.\nHow far is Paris from New York?",
  "decoding_method": "greedy",
  "min_tokens": 1,
  "max_tokens": 30,
  "repetition_penalty": 1.75
}
JSON

echo "===== HS TICKET COMPLETIONS VALIDATION ====="
cat /tmp/hs_ticket_repro_8450.json
echo
echo "============================================"

curl -sS -X POST "http://localhost:8450/v1/completions" \
  -H "Content-Type: application/json" \
  --data @/tmp/hs_ticket_repro_8450.json
echo



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model Qwen/Qwen3-Coder-Next \
    --dataset-name random \
    --num-prompts 80 \
    --max-concurrency 8 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --port 8450 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ignore-eos \
    --trust-remote-code
