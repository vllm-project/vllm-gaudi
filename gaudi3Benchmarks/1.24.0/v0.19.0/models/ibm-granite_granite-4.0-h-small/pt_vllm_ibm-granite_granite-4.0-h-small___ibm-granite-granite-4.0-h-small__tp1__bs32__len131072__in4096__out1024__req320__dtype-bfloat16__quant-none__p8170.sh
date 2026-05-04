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

export VLLM_BUCKET_FILENAME=$(mktemp) && \
cat > "$VLLM_BUCKET_FILENAME" <<'BUCKETS'
(1, [256, 512, 1024, 2048, 4096, 8192], [0, 1, 2, 4, 8, 16])
(1, [512, 2048, 8192], [32, 64, 128, 192, 249])
(2, 1, [2, 4, 8, 16, 32, 64, 128, 256])
(8, 1, [8, 16, 32, 64, 128, 256, 512])
(16, 1, [16, 32, 64, 128, 256, 512, 1024])
(32, 1, [32, 64, 128, 256, 512, 1024, 2048])
BUCKETS
VLLM_CONTIGUOUS_PA=false \
VLLM_GRAPH_RESERVED_MEM=0.3 \
VLLM_BUCKETING_FROM_FILE="$VLLM_BUCKET_FILENAME" \
vllm serve ibm-granite/granite-4.0-h-small \
--override-generation-config '{"temperature": 0}' \
--dtype bfloat16 \
--tensor-parallel-size 1 \
--max_model_len 131072 \
--gpu_memory_util 0.9 \
--max-num-seqs 32 \
--max-num-batched-tokens 8192 \
--enable-chunked-prefill \
--no-enable-prefix-caching \
--tool-call-parser hermes \
--enable-auto-tool-choice \
--async-scheduling  


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
