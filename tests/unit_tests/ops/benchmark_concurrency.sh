#!/bin/bash

# Benchmark script for testing different concurrency levels
# Each run waits for the previous one to complete

MODEL_PATH="/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"
TOKENIZER_PATH="/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"
PORT=8080

# Array of max-concurrency values
CONCURRENCY_LEVELS=(4 8 16 32 64)

echo "Starting benchmark runs at $(date)"
echo "=================================="
LOG_FILE=''
for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"
do
    NUM_PROMPTS=$((10 * CONCURRENCY))
    LOG_FILE="concurrency_${CONCURRENCY}_prompts_${NUM_PROMPTS}_chunks4_8kCutOff_numaNode0.log"
    
    echo ""
    echo "Running benchmark with:"
    echo "  --max-concurrency: ${CONCURRENCY}"
    echo "  --num-prompts: ${NUM_PROMPTS}"
    echo "  Log file: ${LOG_FILE}"
    echo "  Started at: $(date)"
    echo ""
    
    vllm bench serve \
        --model "${MODEL_PATH}" \
        --dataset-name random \
        --request-rate inf \
        --num-prompts ${NUM_PROMPTS} \
        --max-concurrency ${CONCURRENCY} \
        --port ${PORT} \
        --tokenizer "${TOKENIZER_PATH}" \
        --random-input-len 8192 \
        --random-output-len 1024 \
        --random-range-ratio 0.8 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,90,95 \
        --trust-remote-code \
        --ignore-eos 2>&1 | tee "${LOG_FILE}"
    
    EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: Benchmark failed with exit code ${EXIT_CODE}"
        echo "Stopping further runs"
        exit ${EXIT_CODE}
    fi
    
    echo ""
    echo "Completed run with concurrency ${CONCURRENCY} at $(date)"
    echo "=================================="
done

echo ""
echo "All benchmark runs completed successfully at $(date)"
echo ""
echo "Generated log files:"
for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"
do
    NUM_PROMPTS=$((4 * CONCURRENCY))
    echo "  - ${LOG_FILE}"
done

