#!/bin/bash

#@VARS

# Wait for vLLM server to be ready
until curl -s http://localhost:8000/v1/models > /dev/null; do
    echo "Waiting for vLLM server to be ready..."
    sleep 15
done
echo "vLLM server is ready. Starting benchmark..."


SONNET_ARGS=""
if [[ "$DATASET_NAME" == "sonnet" ]]; then
    SONNET_ARGS="--sonnet-prefix-len $PREFIX_LEN --sonnet-input-len $INPUT_TOK --sonnet-output-len $OUTPUT_TOK"
fi

HF_ARGS=""
if [[ "$DATASET_NAME" == "hf" ]]; then
    HF_ARGS="--hf-split train"
fi

SERVER_CMD=""
if [ -f logs/vllm_server.log ]; then
    SERVER_CMD=$(grep -m1 'vllm serve' logs/vllm_server.log || echo "")
fi

CLIENT_CMD="vllm bench serve --model $MODEL --base-url http://localhost:8000 --endpoint $ENDPOINT --backend $BACKEND --dataset-name $DATASET_NAME --dataset-path $DATASET $SONNET_ARGS $HF_ARGS --num-prompts $NUM_PROMPTS --max-concurrency $CONCURRENT_REQ --metric-percentiles 50 90 --temperature 0 --ignore-eos --trust-remote-code --save-result --result-dir logs --result-filename summary_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.json $EXTRA_BENCH_ARGS"

## Executing command print

printf "\n---------------------Starting vLLM bench with the command-------------------------"
printf "\nvllm bench serve \
                --model $MODEL \
                --base-url http://localhost:8000 \
                --endpoint $ENDPOINT \
                --backend $BACKEND \
                --dataset-name $DATASET_NAME \
                --dataset-path $DATASET\
                $SONNET_ARGS \
                $HF_ARGS \
                --num-prompts $NUM_PROMPTS \
                --max-concurrency $CONCURRENT_REQ \
                --metric-percentiles 50 90 \
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
                --dataset-path $DATASET\
                $SONNET_ARGS \
                $HF_ARGS \
                --num-prompts $NUM_PROMPTS \
                --max-concurrency $CONCURRENT_REQ \
                --metric-percentiles 50 90 \
                --temperature 0 \
                --ignore-eos \
                --trust-remote-code \
                --save-result \
                --result-dir logs \
                --result-filename summary_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.json \
                $EXTRA_BENCH_ARGS 2>&1 | stdbuf -o0 -e0 tr '\r' '\n' | tee -a logs/summary_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.log #save results to logs on a host

RESULT_JSON="logs/summary_inp${INPUT_TOK}_out${OUTPUT_TOK}_user${CONCURRENT_REQ}.json"

## Post-process benchmark results to add enriched metrics for Kibana
if [ -f "$RESULT_JSON" ]; then
    echo "Post-processing benchmark results..."
    python benchmark/postprocess_results.py \
        --result-json "$RESULT_JSON" \
        --server-cmd "$SERVER_CMD" \
        --client-cmd "$CLIENT_CMD"
    echo "Post-processing complete. Enriched results saved to $RESULT_JSON"
else
    echo "WARNING: Benchmark result file $RESULT_JSON not found. Skipping post-processing."
fi
