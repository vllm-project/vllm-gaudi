#!/bin/bash

#@VARS

# Wait for vLLM server to be ready
until curl -s http://localhost:8000/v1/models > /dev/null; do
    echo "Waiting for vLLM server to be ready..."
    sleep 15
done
echo "vLLM server is ready. Starting benchmark..."


# Total prompt length is PREFIX_LEN + random-input-len. INPUT_TOK is the
# full prompt length from the golden config, so the random context is the
# remainder. This keeps the previous 2048-token prompt with a 100-token prefix.
RANDOM_ARGS=""
if [[ "$DATASET_NAME" == "random" ]]; then
    PREFIX_LEN=${PREFIX_LEN:-0}
    RANDOM_INPUT_LEN=$((INPUT_TOK - PREFIX_LEN))
    if (( RANDOM_INPUT_LEN < 1 )); then
        echo "INPUT_TOK ($INPUT_TOK) must be greater than PREFIX_LEN ($PREFIX_LEN)" >&2
        exit 1
    fi
    RANDOM_ARGS="--random-prefix-len $PREFIX_LEN --random-input-len $RANDOM_INPUT_LEN --random-output-len $OUTPUT_TOK"
fi

HF_ARGS=""
if [[ "$DATASET_NAME" == "hf" ]]; then
    HF_ARGS="--hf-split train"
fi

## Executing command print

printf "\n---------------------Starting vLLM bench with the command-------------------------"
printf "\nvllm bench serve \
                --model $MODEL \
                --base-url http://localhost:8000 \
                --endpoint $ENDPOINT \
                --backend $BACKEND \
                --dataset-name $DATASET_NAME \
                ${DATASET:+--dataset-path $DATASET} \
                $RANDOM_ARGS \
                $HF_ARGS \
                --num-prompts $NUM_PROMPTS \
                --max-concurrency $CONCURRENT_REQ \
                --metric-percentiles 90 \
                --temperature 0 \
                --ignore-eos \
                --trust-remote-code \
                --save-result \
                --result-dir logs \
                --result-filename summary_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.json \
                $EXTRA_BENCH_ARGS"
printf "\n-----------------------------------------------------------------------------------\n"

## Start benchmarking vLLM serving
vllm bench serve \
                --model $MODEL \
                --base-url http://localhost:8000 \
                --endpoint $ENDPOINT \
                --backend $BACKEND \
                --dataset-name $DATASET_NAME \
                ${DATASET:+--dataset-path $DATASET} \
                $RANDOM_ARGS \
                $HF_ARGS \
                --num-prompts $NUM_PROMPTS \
                --max-concurrency $CONCURRENT_REQ \
                --metric-percentiles 90 \
                --temperature 0 \
                --ignore-eos \
                --trust-remote-code \
                --save-result \
                --result-dir logs \
                --result-filename summary_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.json \
                $EXTRA_BENCH_ARGS 2>&1 | stdbuf -o0 -e0 tr '\r' '\n' | tee -a logs/summary_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.log #save results to logs on a host
