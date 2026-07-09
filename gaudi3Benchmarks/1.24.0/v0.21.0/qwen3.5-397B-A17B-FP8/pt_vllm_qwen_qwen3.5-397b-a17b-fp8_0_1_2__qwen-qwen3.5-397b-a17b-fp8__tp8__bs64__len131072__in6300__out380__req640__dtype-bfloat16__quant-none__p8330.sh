#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_Qwen_Qwen3.5-397B-A17B-FP8_0_1_2_3_4_5_6_7
# model=Qwen/Qwen3.5-397B-A17B-FP8
# tensor_parallel_size=8
# batch_size=64
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=6300
# output_len=380
# num_prompts=640
# max_num_batched_tokens=na
# port=8330

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=17200 \
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
PT_HPU_ENABLE_EAGER_CACHE=true \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
EXPERIMENTAL_WEIGHT_SHARING=0 \
VLLM_SKIP_WARMUP=false \
VLLM_GRAPH_PROMPT_RATIO=0.3 \
ENABLE_EXPERIMENTAL_FLAGS=true \
ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true \
VLLM_USE_HYBRID_CACHE=true \
VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=false \
VLLM_COMPACT_GDN=1 \
vllm serve \
    --model=Qwen/Qwen3.5-397B-A17B-FP8 \
    --port 8330 \
    --max-num-seqs=64 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.9 \
    --tensor-parallel-size=8 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser \
    qwen3_coder \
    --enable-expert-parallel \
    --max-num-batched-tokens 8192 \
    --limit-mm-per-prompt '{"image": {"count": 10, "width": 864, "height": 480}}' \
    --no-enable-prefix-caching \
    --async-scheduling \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --trust-remote-code


# =========================
# TOOL VALIDATION COMMAND
# =========================

# =========================
# TOOL VALIDATION CURL ONLY
# =========================

cat > /tmp/tool_probe_8330.json <<'JSON'
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
  "temperature": 0.7,
  "max_tokens": 256,
  "logprobs": false,
  "n": 2,
  "repetition_penalty": 1.1,
  "presence_penalty": 0.5,
  "frequency_penalty": 0.5
}
JSON

curl -sS http://localhost:8330/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data @/tmp/tool_probe_8330.json



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --dataset-name random-mm \
    --base-url http://localhost:8330 \
    --num-prompts 640 \
    --max-concurrency 64 \
    --request-rate inf \
    --random-input-len 6300 \
    --random-output-len 380 \
    --endpoint /v1/chat/completions \
    --port 8330 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --backend openai-chat \
    --tokenizer Qwen/Qwen3.5-397B-A17B-FP8 \
    --random-mm-base-items-per-request 10 \
    --random-mm-limit-mm-per-prompt "{\"image\":10}" \
    --random-mm-bucket-config "{(480, 864, 1): 1.0}" \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --ignore-eos \
    --trust-remote-code
