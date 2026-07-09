#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_openai_gpt-oss-120b_0_1_2_3_4_5_6_7
# model=openai/gpt-oss-120b
# tensor_parallel_size=8
# batch_size=128
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=4096
# output_len=1024
# num_prompts=640
# max_num_batched_tokens=na
# port=8340

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
PT_HPU_WEIGHT_SHARING=0 \
VLLM_PROMPT_BS_BUCKET_MAX=1 \
VLLM_PROMPT_QUERY_BUCKET_MIN=1024 \
VLLM_PROMPT_QUERY_BUCKET_STEP=128 \
VLLM_PROMPT_QUERY_BUCKET_MAX=2048 \
VLLM_PROMPT_CTX_BUCKET_STEP=128 \
VLLM_DECODE_BS_BUCKET_MIN=128 \
VLLM_DECODE_BS_BUCKET_MAX=128 \
VLLM_DECODE_BLOCK_BUCKET_MIN=512 \
VLLM_DECODE_BLOCK_BUCKET_STEP=128 \
VLLM_DECODE_BLOCK_BUCKET_MAX=2048 \
VLLM_DECODE_CTX_BUCKET_STEP=128 \
RUNTIME_SCALE_PATCHING=1 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
VLLM_GRAPH_RESERVED_MEM=0.3 \
vllm serve \
    --model=openai/gpt-oss-120b \
    --port 8340 \
    --max-num-seqs=128 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.95 \
    --tensor-parallel-size=8 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    openai \
    --tool-call-parser openai \
    --disable-log-stats \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8340.json <<'JSON'
{
  "model": "openai/gpt-oss-120b",
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

curl -sS http://localhost:8340/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8340.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model openai/gpt-oss-120b \
    --dataset-name random \
    --num-prompts 640 \
    --max-concurrency 128 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --port 8340 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ignore-eos \
    --trust-remote-code
