#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_google_gemma-4-26B-A4B-it_1_2_5_7
# model=google/gemma-4-26B-A4B-it
# tensor_parallel_size=4
# batch_size=45
# max_model_len=262144
# dtype=bfloat16
# quant_config=none
# input_len=6300
# output_len=380
# num_prompts=450
# max_num_batched_tokens=na
# port=8290

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
VLLM_SKIP_WARMUP=true \
PT_HPU_ENABLE_EAGER_CACHE=true \
EXPERIMENTAL_WEIGHT_SHARING=0 \
ENABLE_EXPERIMENTAL_FLAGS=true \
ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
vllm serve \
    --model=google/gemma-4-26B-A4B-it \
    --port 8290 \
    --max-num-seqs=45 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.7 \
    --tensor-parallel-size=4 \
    --max-model-len=262144 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    gemma4 \
    --host localhost \
    --tokenizer google/gemma-4-26B-A4B-it \
    --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image": {"count": 20, "width": 864, "height": 480}}' \
    --enable-expert-parallel \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8290.json <<'JSON'
{
  "model": "google/gemma-4-26B-A4B-it",
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

curl -sS http://localhost:8290/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8290.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model google/gemma-4-26B-A4B-it \
    --dataset-name random-mm \
    --base-url http://localhost:8290 \
    --num-prompts 450 \
    --max-concurrency 45 \
    --request-rate inf \
    --random-input-len 6300 \
    --random-output-len 380 \
    --endpoint /v1/chat/completions \
    --port 8290 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --backend openai-chat \
    --tokenizer google/gemma-4-26B-A4B-it \
    --random-mm-base-items-per-request 10 \
    --random-mm-limit-mm-per-prompt "{\"image\":10}" \
    --random-mm-bucket-config "{(480, 864, 1): 1.0}" \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --ignore-eos \
    --trust-remote-code
