#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================
# container=pt_vllm_moonshotai_Kimi-K2.6_0_1_2_3_4_5_6_7
# model=moonshotai/Kimi-K2.6
# tensor_parallel_size=8
# batch_size=45
# max_model_len=131072
# dtype=bfloat16
# quant_config=none
# input_len=6300
# output_len=380
# num_prompts=450
# max_num_batched_tokens=na
# port=8310

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
PT_HPU_ENABLE_EAGER_CACHE=true \
ENABLE_EXPERIMENTAL_FLAGS=true \
ENABLE_SKIP_REMOVAL_OF_GRAPH_INPUT_IDENTITY_NODES=true \
VLLM_DECODE_BS_BUCKET_MAX=45 \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
EXPERIMENTAL_WEIGHT_SHARING=0 \
VLLM_USE_HYBRID_CACHE=true \
VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=false \
VLLM_COMPACT_GDN=1 \
VLLM_GRAPH_RESERVED_MEM=0.3 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
HPU_FUSED_MOE=1 \
vllm serve \
    --model=moonshotai/Kimi-K2.6 \
    --port 8310 \
    --max-num-seqs=45 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.7 \
    --tensor-parallel-size=8 \
    --max-model-len=131072 \
    --block-size=128 \
    --async-scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --host localhost \
    --max-num-batched-tokens 16384 \
    --limit-mm-per-prompt '{"image": {"count": 20, "width": 864, "height": 480}}' \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    --trust-remote-code



# =========================
# BENCHMARK COMMAND
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model moonshotai/Kimi-K2.6 \
    --dataset-name random-mm \
    --base-url http://localhost:8310 \
    --num-prompts 450 \
    --max-concurrency 45 \
    --request-rate inf \
    --random-input-len 6300 \
    --random-output-len 380 \
    --endpoint /v1/chat/completions \
    --port 8310 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --backend openai-chat \
    --tokenizer moonshotai/Kimi-K2.6 \
    --random-mm-base-items-per-request 20 \
    --random-mm-limit-mm-per-prompt "{\"image\":20}" \
    --random-mm-bucket-config "{(480, 864, 1): 1.0}" \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --ignore-eos \
    --trust-remote-code
