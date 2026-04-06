#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_Qwen_Qwen3-VL-32B-Instruct-FP8_3
# model=Qwen/Qwen3-VL-32B-Instruct-FP8
# tensor_parallel_size=1
# batch_size=8
# max_model_len=32768
# dtype=bfloat16
# quant_config=none
# input_len=6300
# output_len=380
# num_prompts=20
# max_num_batched_tokens=na
# port=8220

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
vllm serve \
    --model=Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --port 8220 \
    --max-num-seqs=8 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.65 \
    --tensor-parallel-size=1 \
    --max-model-len=32768 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    hermes \
    --max-num-batched-tokens 32768 \
    --limit-mm-per-prompt '{"image": {"count": 20, "width": 864, "height": 480}}' \
    --no-enable-chunked-prefill \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

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



# =========================
# BENCHMARK COMMAND
# =========================

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
