#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_Qwen_Qwen3.6-35B-A3B_1_7
# model=Qwen/Qwen3.6-35B-A3B
# tensor_parallel_size=2
# batch_size=45
# max_model_len=262144
# dtype=bfloat16
# quant_config=none
# input_len=4096
# output_len=1024
# num_prompts=160
# max_num_batched_tokens=na
# port=8280

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=1000 \
VLLM_EXPONENTIAL_BUCKETING=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=1 \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=false \
VLLM_DEFRAG=false \
VLLM_FUSED_BLOCK_SOFTMAX=true \
PT_HPU_WEIGHT_SHARING=0 \
EXPERIMENTAL_WEIGHT_SHARING=0 \
PT_HPU_EAGER_PIPELINE_ENABLE=1 \
PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE=1 \
RUNTIME_SCALE_PATCHING=1 \
VLLM_SKIP_WARMUP=false \
VLLM_GRAPH_RESERVED_MEM=0.1 \
VLLM_GRAPH_PROMPT_RATIO=0.3 \
VLLM_USE_HYBRID_CACHE=true \
VLLM_COMPACT_GDN=1 \
VLLM_GDN_COMPUTE_FP32=1 \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=0 \
PT_HPU_ENABLE_EAGER_CACHE=true \
ENABLE_EXPERIMENTAL_FLAGS=true \
ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true \
vllm serve \
    --model=Qwen/Qwen3.6-35B-A3B \
    --port 8280 \
    --max-num-seqs=45 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.9 \
    --tensor-parallel-size=2 \
    --max-model-len=262144 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    qwen3_xml \
    --reasoning-parser \
    qwen3 \
    --served-model-name Qwen/Qwen3.6-35B-A3B \
    --enable-expert-parallel \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8280.json <<'JSON'
{
  "model": "Qwen/Qwen3.6-35B-A3B",
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

curl -sS http://localhost:8280/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8280.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model Qwen/Qwen3.6-35B-A3B \
    --dataset-name random \
    --num-prompts 160 \
    --max-concurrency 16 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --port 8280 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ignore-eos \
    --trust-remote-code
