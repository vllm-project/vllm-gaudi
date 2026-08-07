#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_zai-org_GLM-5.2-FP8_0_1_2_3_4_5_6_7
# model=zai-org/GLM-5.2-FP8
# tensor_parallel_size=8
# batch_size=32
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=4096
# output_len=1024
# num_prompts=320
# max_num_batched_tokens=na
# port=8320

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=36000 \
VLLM_RPC_TIMEOUT=10000000 \
VLLM_EXPONENTIAL_BUCKETING=true \
VLLM_BUCKETING_STRATEGY=exp \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
VLLM_FUSED_BLOCK_SOFTMAX=true \
ENABLE_EXPERIMENTAL_FLAGS=true \
EXPERIMENTAL_WEIGHT_SHARING=0 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
vllm serve \
    --model=zai-org/GLM-5.2-FP8 \
    --port 8320 \
    --max-num-seqs=32 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.90 \
    --tensor-parallel-size=8 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    glm47 \
    --reasoning-parser \
    glm45 \
    --host localhost \
    --max-num-batched-tokens 4096 \
    --enable-expert-parallel \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8320.json <<'JSON'
{
  "model": "zai-org/GLM-5.2-FP8",
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

curl -sS http://localhost:8320/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8320.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model zai-org/GLM-5.2-FP8 \
    --dataset-name random \
    --base-url http://localhost:8320 \
    --num-prompts 320 \
    --max-concurrency 32 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --endpoint /v1/chat/completions \
    --port 8320 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --backend openai-chat \
    --tokenizer zai-org/GLM-5.2-FP8 \
    --ignore-eos \
    --trust-remote-code
