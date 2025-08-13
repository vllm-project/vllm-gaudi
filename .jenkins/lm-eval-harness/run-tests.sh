#!/bin/bash

# Trap Ctrl-C (SIGINT) to cancel running test and prevent further scheduling
TEST_PROCESS=""
trap 'echo "Interrupted by user. Killing running test and exiting."; if [[ -n "$TEST_PROCESS" ]]; then kill -9 $TEST_PROCESS 2>/dev/null || true; fi; exit 130' SIGINT

usage() {
    echo``
    echo "Runs lm eval harness on GSM8k using vllm and compares to "
    echo "precomputed baseline (measured by HF transformers.)"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -c    - path to the test data config (e.g. configs/small-models.txt)"
    echo "  -t    - tensor parallel size"
    echo "  -a    - enable automatic prefix caching"

    echo
}

SUCCESS=0
APC_ENABLED="false"
TIMEOUT_S=900 # 15 minutes timeout per test
TP_SIZE=1
while getopts "c:t:a" OPT; do
  case ${OPT} in
    c ) 
        CONFIG="$OPTARG"
        ;;
    t )
        TP_SIZE="$OPTARG"
        ;;
    a )
        APC_ENABLED="true"
        ;;
    \? )
        usage
        exit 1
        ;;
  esac
done

# Parse list of configs.
IFS=$'\n' read -d '' -r -a MODEL_CONFIGS < "$CONFIG"

PASSED_MODELS=()
FAILED_MODELS=()

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"
do
    LOCAL_SUCCESS=0
    
    echo "=== RUNNING MODEL: $MODEL_CONFIG WITH TP SIZE: $TP_SIZE==="

    export LM_EVAL_TEST_DATA_FILE=$PWD/configs/${MODEL_CONFIG}
    export LM_EVAL_TP_SIZE=$TP_SIZE
    export LM_EVAL_APC_ENABLED=$APC_ENABLED
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
    export VLLM_SKIP_WARMUP=true
    export TQDM_BAR_FORMAT="{desc}: {percentage:3.0f}% {bar:10} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" 
    RANDOM_SUFFIX=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 4; echo)
    JUNIT_FAMILY=""
    JUNIT_XML=""
    if [[ -n "$TEST_RESULTS_DIR" ]]; then
        LOG_DIR=$TEST_RESULTS_DIR
        LOG_FILENAME="test_${MODEL_CONFIG}_${RANDOM_SUFFIX}.xml"
        LOG_PATH="${LOG_DIR}/${LOG_FILENAME}"
        JUNIT_FAMILY="-o junit_family=xunit1"
        JUNIT_XML="--junitxml=${LOG_PATH}"
    fi
    echo "Executing command:"
    echo "LM_EVAL_TEST_DATA_FILE=$LM_EVAL_TEST_DATA_FILE LM_EVAL_TP_SIZE=$LM_EVAL_TP_SIZE LM_EVAL_APC_ENABLED=$LM_EVAL_APC_ENABLED PT_HPU_ENABLE_LAZY_COLLECTIVES=$PT_HPU_ENABLE_LAZY_COLLECTIVES VLLM_SKIP_WARMUP=$VLLM_SKIP_WARMUP timeout $TIMEOUT_S pytest -s test_lm_eval_correctness.py \"$JUNIT_FAMILY\" \"$JUNIT_XML\""
    LM_EVAL_TEST_DATA_FILE=$LM_EVAL_TEST_DATA_FILE LM_EVAL_TP_SIZE=$LM_EVAL_TP_SIZE LM_EVAL_APC_ENABLED=$LM_EVAL_APC_ENABLED PT_HPU_ENABLE_LAZY_COLLECTIVES=$PT_HPU_ENABLE_LAZY_COLLECTIVES VLLM_SKIP_WARMUP=$VLLM_SKIP_WARMUP timeout $TIMEOUT_S pytest -s test_lm_eval_correctness.py "$JUNIT_FAMILY" "$JUNIT_XML"
    TEST_PROCESS=$!
    wait $TEST_PROCESS
    LOCAL_SUCCESS=$?
    kill -9 $TEST_PROCESS 2> /dev/null || true
    if [[ $LOCAL_SUCCESS == 0 ]]; then
        echo "=== PASSED MODEL: ${MODEL_CONFIG} ==="
        PASSED_MODELS+=("$MODEL_CONFIG")
    else
        echo "=== FAILED MODEL: ${MODEL_CONFIG} ==="
        FAILED_MODELS+=("$MODEL_CONFIG")
    fi

    SUCCESS=$((SUCCESS + LOCAL_SUCCESS))

done

echo
if [ ${#PASSED_MODELS[@]} -gt 0 ]; then
    echo "PASSED MODELS:"
    for model in "${PASSED_MODELS[@]}"; do
        echo "  $model"
    done
else
    echo "No models passed."
fi

echo
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "FAILED MODELS:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  $model"
    done
else
    echo "No models failed."
fi

if [ "${SUCCESS}" -eq "0" ]; then
    exit 0
else
    exit 1
fi
