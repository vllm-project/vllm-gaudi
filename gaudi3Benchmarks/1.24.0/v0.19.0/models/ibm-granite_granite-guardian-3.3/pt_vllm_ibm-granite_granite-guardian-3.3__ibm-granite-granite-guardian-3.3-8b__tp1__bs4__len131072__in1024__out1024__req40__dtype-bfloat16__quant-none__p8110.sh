#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_ibm-granite_granite-guardian-3.3-8b_2
# model=ibm-granite/granite-guardian-3.3-8b
# tensor_parallel_size=1
# batch_size=4
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=1024
# output_len=1024
# num_prompts=40
# max_num_batched_tokens=na
# port=8110

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
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_WEIGHT_SHARING=0 \
vllm serve \
    --model=ibm-granite/granite-guardian-3.3-8b \
    --port 8110 \
    --max-num-seqs=4 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.9 \
    --tensor-parallel-size=1 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --override-generation-config '{"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512}' \
    --trust-remote-code



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model ibm-granite/granite-guardian-3.3-8b \
    --dataset-name custom \
    --dataset-path /tmp/aegis-benchmark-granite-guardian-3.3-8b.jsonl \
    --base-url http://localhost:8110 \
    --num-prompts 40 \
    --max-concurrency 4 \
    --request-rate inf \
    --port 8110 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --temperature 0 \
    --skip-chat-template \
    --trust-remote-code
