#!/bin/bash

#@VARS

if [ $ASYNC_SCHEDULING -gt 0 ]; then # Checks if using async scheduling
    EXTRA_ARGS+=" --async_scheduling"
fi

## Start server
vllm serve $MODEL \
        --block-size $BLOCK_SIZE \
        --dtype $DTYPE \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --download_dir $HF_HOME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTILIZATION \
        --max-num-seqs $MAX_NUM_SEQS \
        --generation-config vllm \
        --max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS \
        --disable-log-requests ${EXTRA_ARGS} \
2>&1 | tee -a  logs/vllm_server.log
