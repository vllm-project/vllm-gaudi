#!/bin/bash

#@VARS

## Start server
vllm serve $MODEL \
        --block-size $BLOCK_SIZE \
        --dtype $DTYPE \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --download_dir $HF_HOME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEM_UTILIZATION \
        --max-num-seqs $MAX_NUM_SEQS \
        --disable-log-requests \
2>&1 | tee -a  logs/vllm_server.log
