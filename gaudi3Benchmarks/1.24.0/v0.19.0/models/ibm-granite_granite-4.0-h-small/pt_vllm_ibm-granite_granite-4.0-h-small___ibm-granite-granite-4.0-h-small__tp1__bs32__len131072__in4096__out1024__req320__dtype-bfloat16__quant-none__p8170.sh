#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_ibm-granite_granite-4.0-h-small_1
# model=ibm-granite/granite-4.0-h-small
# tensor_parallel_size=1
# batch_size=32
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=4096
# output_len=1024
# num_prompts=320
# max_num_batched_tokens=na
# port=8170

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=true \
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
    --model=ibm-granite/granite-4.0-h-small \
    --port 8170 \
    --max-num-seqs=32 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.45 \
    --tensor-parallel-size=1 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    hermes \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --tool-call-parser hermes \
    --override-generation-config '{"temperature": 0}' \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8170.json <<'JSON'
{
  "model": "ibm-granite/granite-4.0-h-small",
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

curl -sS http://localhost:8170/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8170.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model ibm-granite/granite-4.0-h-small \
    --dataset-name random \
    --num-prompts 320 \
    --max-concurrency 32 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --port 8170 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 600 \
    --ignore-eos \
    --trust-remote-code
