#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=Qwen/Qwen3.5-397B-A17B-FP8
# model=Qwen/Qwen3.5-397B-A17B-FP8
# tensor_parallel_size=8
# batch_size=64
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=6300
# output_len=380
# num_prompts=20
# max_num_batched_tokens=na
# port=8230

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
PT_HPU_ENABLE_EAGER_CACHE=true \
VLLM_ENGINE_ITERATION_TIMEOUT_S=17200 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=true \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
EXPERIMENTAL_WEIGHT_SHARING=0 \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
VLLM_SKIP_WARMUP=false \
VLLM_GRAPH_PROMPT_RATIO=0.3 \
PT_HPU_LAZY_MODE=0 \
ENABLE_EXPERIMENTAL_FLAGS=true \
ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true \
VLLM_USE_HYBRID_CACHE=true VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=false \
VLLM_COMPACT_GDN=1 \
vllm  serve \
    --model="Qwen/Qwen3.5-397B-A17B-FP8" \
    --host localhost \
    --port 8230 \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --max_num_batched_tokens 8192 \
    --gpu-memory-util 0.85 \
    --limit-mm-per-prompt '{"image": {"count": 20, "width": 864, "height": 480}}' \
    --no-enable-prefix-caching \
    --max-num-seqs 64 \
    --block-size 128 \
    --async-scheduling \
    --enable-expert-parallel \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' 

# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8230.json <<'JSON'
{
  "model": "Qwen/Qwen3.5-397B-A17B-FP8",
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

curl -sS http://localhost:8230/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8230.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model "Qwen/Qwen3.5-397B-A17B-FP8" \
    --dataset-name random-mm \
    --base-url http://localhost:8230 \
    --num-prompts 20 \
    --max-concurrency 64 \
    --request-rate inf \
    --random-input-len 6300 \
    --random-output-len 380 \
    --endpoint /v1/chat/completions \
    --port 8230 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 600 \
    --backend openai-chat \
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \
    --random-mm-base-items-per-request 20 \
    --random-mm-bucket-config "{(480, 864, 1): 1.0}" \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --ignore-eos \
    --trust-remote-code
