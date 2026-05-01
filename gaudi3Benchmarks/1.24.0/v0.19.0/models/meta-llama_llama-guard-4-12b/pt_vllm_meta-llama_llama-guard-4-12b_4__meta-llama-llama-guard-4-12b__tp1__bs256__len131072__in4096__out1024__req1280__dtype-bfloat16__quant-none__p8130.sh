#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_meta-llama_Llama-Guard-4-12B_4
# model=meta-llama/Llama-Guard-4-12B
# tensor_parallel_size=1
# batch_size=256
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=4096
# output_len=1024
# num_prompts=1280
# max_num_batched_tokens=na
# port=8130

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
VLLM_GRAPH_RESERVED_MEM=0.3 \
vllm serve \
    --model=meta-llama/Llama-Guard-4-12B \
    --port 8130 \
    --max-num-seqs=256 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.95 \
    --tensor-parallel-size=1 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 8192 \
    --trust-remote-code



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model meta-llama/Llama-Guard-4-12B \
    --dataset-name random \
    --num-prompts 1280 \
    --max-concurrency 256 \
    --request-rate inf \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --port 8130 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ready-check-timeout-sec 600 \
    --ignore-eos \
    --trust-remote-code
