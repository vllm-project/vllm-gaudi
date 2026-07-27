#!/bin/bash
set -e

#server command 
PT_HPU_ENABLE_EAGER_CACHE=true \
ENABLE_EXPERIMENTAL_FLAGS=true \
ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true \
PT_HPU_LAZY_MODE=0 \
VLLM_EXPONENTIAL_BUCKETING=true \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_STEP=1 \
VLLM_PROMPT_BS_BUCKET_MAX=4 \
VLLM_PROMPT_QUERY_BUCKET_MIN=128 \
VLLM_PROMPT_QUERY_BUCKET_STEP=512 \
VLLM_PROMPT_QUERY_BUCKET_MAX=4096 \
VLLM_PROMPT_CTX_BUCKET_MIN=0 \
VLLM_PROMPT_CTX_BUCKET_STEP=1 \
VLLM_PROMPT_CTX_BUCKET_MAX=64 \
VLLM_USE_HYBRID_CACHE=true \
VLLM_COMPACT_GDN=1 \
VLLM_GDN_COMPUTE_FP32=1 \
VLLM_GRAPH_RESERVED_MEM=0.05 \
VLLM_GRAPH_PROMPT_RATIO=0.1 \
VLLM_SKIP_WARMUP=false \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
EXPERIMENTAL_WEIGHT_SHARING=0 \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
VLLM_ENGINE_ITERATION_TIMEOUT_S=17200 \
VLLM_RPC_TIMEOUT=100000 \
ENABLE_PARALLEL_COMPILATION=false \
HABANA_GRAPH_COMPILATION_THREADS=1 \
vllm serve \
    --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
    --dtype bfloat16 \
    --port 8150 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.9 \
    --disable-log-stats \
    --max-model-len 262144 \
    --max-num-seqs 45 \
    --block-size 128 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --trust-remote-code 2>&1 | tee server.log 


#client command 
vllm bench serve \
    --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
    --dataset-name random \
    --random-input-len 204800 \
    --random-output-len 10240 \
    --ignore-eos \
    --num-prompts 45 \
    --max-concurrency 45 \
    --trust-remote-code \
    --request-rate inf \
    --backend vllm \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --port 8150 2>&1 | tee bench.log 
