#!/bin/bash
export MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export DTYPE=bfloat16
export TENSOR_PARALLEL_SIZE=1
export MAX_MODEL_LEN=16896
export BLOCK_SIZE=128
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=32
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=32
export VLLM_PROMPT_SEQ_BUCKET_MIN=128
export VLLM_PROMPT_SEQ_BUCKET_STEP=256
export VLLM_DECODE_BLOCK_BUCKET_MIN=128
export VLLM_DECODE_BLOCK_BUCKET_STEP=256
export MAX_NUM_PREFILL_SEQS=1
export PT_HPU_LAZY_MODE=1
export VLLM_DELAYED_SAMPLING=True
export VLLM_SKIP_WARMUP=False
export EXPERIMENTAL_WEIGHT_SHARING=0
export VLLM_EXPONENTIAL_BUCKETING=False
export MAX_NUM_BATCHED_TOKENS=2048
export PT_HPU_ENABLE_LAZY_COLLECTIVES=False
export GPU_MEM_UTILIZATION=0.95
export VLLM_GRAPH_PROMPT_RATIO=0.8
export VLLM_GRAPH_RESERVED_MEM=0.15
export MAX_NUM_SEQS=160
export VLLM_PROMPT_SEQ_BUCKET_MAX=16896
export VLLM_CONTIGUOUS_PA=False
export VLLM_DEFRAG=False
export ASYNC_SCHEDULING=1
export VLLM_WEIGHT_LOAD_FORCE_SYNC=0
export VLLM_SKIP_WARMUP=true
vllm serve $MODEL \
        --block-size $BLOCK_SIZE \
        --dtype $DTYPE \
        --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTILIZATION \
        --max-num-seqs $MAX_NUM_SEQS \
        --generation-config vllm \
        --max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS \
        --no-enable-prefix-caching \
        --disable-log-requests --async_scheduling \
        --limit-mm-per-prompt '{"image":{"count": 20, "width": 1024, "height": 576}}' \
2>&1 | tee -a  logs/vllm_server.log