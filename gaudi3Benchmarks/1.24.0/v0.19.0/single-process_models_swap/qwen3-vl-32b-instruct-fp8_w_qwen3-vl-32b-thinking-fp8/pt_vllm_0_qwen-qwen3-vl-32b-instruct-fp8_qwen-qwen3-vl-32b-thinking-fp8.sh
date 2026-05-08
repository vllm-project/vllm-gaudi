#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================

export VLLM_HPU_MULTI_MODEL_CONFIG=multi_models.yaml && \
cat > "$VLLM_HPU_MULTI_MODEL_CONFIG" << 'EOF'
default_model: Qwen/Qwen3-VL-32B-Instruct-FP8
models:
  Qwen/Qwen3-VL-32B-Instruct-FP8:
    model: Qwen/Qwen3-VL-32B-Instruct-FP8
    tensor_parallel_size: 1
    max_num_seqs: 8
    dtype: bfloat16
    block_size: 128
    max_model_len: 32768
    async_scheduling: True
    enable_prefix_caching: False
    gpu_memory_utilization: 0.65
    max_num_batched_tokens: 32768
    limit_mm_per_prompt:
      image:
        count: 20
        width: 864
        height: 480
    enable_chunked_prefill: False

  Qwen/Qwen3-VL-32B-Thinking-FP8:
    model: Qwen/Qwen3-VL-32B-Thinking-FP8
    tensor_parallel_size: 1
    max_num_seqs: 8
    dtype: bfloat16
    block_size: 128
    max_model_len: 32768
    async_scheduling: True
    enable_prefix_caching: False
    gpu_memory_utilization: 0.65
    reasoning_parser: qwen3
    max_num_batched_tokens: 32768
    limit_mm_per_prompt:
      image:
        count: 20
        width: 864
        height: 480 
    enable_chunked_prefill: False
EOF

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
VLLM_DELAYED_SAMPLING=true \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
EXPERIMENTAL_WEIGHT_SHARING=0 \
VLLM_SKIP_WARMUP=false \
VLLM_GRAPH_PROMPT_RATIO=0.3 \
VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
VLLM_HPU_MULTI_MODEL_CONFIG=multi_models.yaml \
python -m vllm_gaudi.entrypoints.openai.multi_model_api_server \
	--port 8220 \
	--enable-auto-tool-choice \
	--tool-call-parser \
	hermes \
	--disable-log-stats \
	--trust-remote-code

# ====================================
# TOOL VALIDATION CURL ONLY - Model 1
# ====================================

cat > /tmp/tool_probe_8220.json <<'JSON'
{
  "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
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

curl -sS http://localhost:8220/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8220.json
  
# =============================
# BENCHMARK COMMAND - Model 1
# =============================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --dataset-name random-mm \
    --base-url http://localhost:8220 \
    --num-prompts 20 \
    --max-concurrency 8 \
    --request-rate inf \
    --random-input-len 6300 \
    --random-output-len 380 \
    --endpoint /v1/chat/completions \
    --port 8220 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 600 \
    --backend openai-chat \
    --tokenizer Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --random-mm-base-items-per-request 20 \
    --random-mm-bucket-config "{(480, 864, 1): 1.0}" \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --ignore-eos \
    --trust-remote-code

# =========================
# ONLINE SWAP COMMAND 
# =========================
curl -s http://localhost:8220/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-32B-Thinking-FP8",
    "drain_timeout": 60
  }' | jq
 
# ====================================
# TOOL VALIDATION CURL ONLY - Model 2
# ====================================

cat > /tmp/tool_probe_8220_bis.json <<'JSON'
{
  "model": "Qwen/Qwen3-VL-32B-Thinking-FP8",
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

curl -sS http://localhost:8220/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8220_bis.json
  
# =========================
# BENCHMARK COMMAND - Model 2
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model Qwen/Qwen3-VL-32B-Thinking-FP8 \
    --dataset-name random-mm \
    --base-url http://localhost:8220 \
    --num-prompts 20 \
    --max-concurrency 8 \
    --request-rate inf \
    --random-input-len 6300 \
    --random-output-len 380 \
    --endpoint /v1/chat/completions \
    --port 8220 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 600 \
    --backend openai-chat \
    --tokenizer Qwen/Qwen3-VL-32B-Thinking-FP8 \
    --random-mm-base-items-per-request 20 \
    --random-mm-bucket-config "{(480, 864, 1): 1.0}" \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --ignore-eos \
    --trust-remote-code