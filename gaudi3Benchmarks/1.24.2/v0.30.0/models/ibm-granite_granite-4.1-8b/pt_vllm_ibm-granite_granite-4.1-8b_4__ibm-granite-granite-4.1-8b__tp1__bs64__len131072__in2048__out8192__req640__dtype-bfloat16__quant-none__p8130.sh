#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_ibm-granite_granite-4.1-8b_4
# model=ibm-granite/granite-4.1-8b
# tensor_parallel_size=1
# batch_size=64
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=2048
# output_len=8192
# num_prompts=640
# max_num_batched_tokens=na
# port=8130

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
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
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_BUCKET_FILENAME=/software/ae/fmwork/vllm_buckets_131072.txt \
VLLM_GRAPH_RESERVED_MEM=0.3 \
vllm serve \
    --model=ibm-granite/granite-4.1-8b \
    --port 8130 \
    --max-num-seqs=64 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.88 \
    --tensor-parallel-size=1 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    granite4 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --override-generation-config '{"temperature": 0}' \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8130.json <<'JSON'
{
  "model": "ibm-granite/granite-4.1-8b",
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

curl -sS http://localhost:8130/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8130.json




# =========================
# HS TICKET COMPLETIONS VALIDATION COMMAND
# =========================

# =========================
# HS TICKET COMPLETIONS VALIDATION
# =========================

set +e

cat > /tmp/hs_ticket_repro_8130.json <<'JSON'
{
  "model": "ibm-granite/granite-4.1-8b",
  "prompt": "Please answer the following question.\nHow far is Paris from New York?",
  "decoding_method": "greedy",
  "min_tokens": 1,
  "max_tokens": 30,
  "repetition_penalty": 1.75
}
JSON

echo "===== HS TICKET COMPLETIONS VALIDATION ====="
cat /tmp/hs_ticket_repro_8130.json
echo
echo "============================================"

curl -sS -X POST "http://localhost:8130/v1/completions" \
  -H "Content-Type: application/json" \
  --data @/tmp/hs_ticket_repro_8130.json
echo



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model ibm-granite/granite-4.1-8b \
    --dataset-name random \
    --num-prompts 640 \
    --max-concurrency 64 \
    --request-rate inf \
    --random-input-len 2048 \
    --random-output-len 8192 \
    --port 8130 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ignore-eos \
    --trust-remote-code
