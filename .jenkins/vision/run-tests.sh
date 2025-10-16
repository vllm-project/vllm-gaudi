#!/bin/bash

usage() {
    echo``
    echo "Runs simple request check on multimodal models using vllm"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -c    - path to the test data config (e.g. configs/small-models.txt)"
    echo "  -t    - tensor parallel size"
    echo
}

SUCCESS=0

while getopts "c:t:" OPT; do
  case ${OPT} in
    c ) 
        CONFIG="$OPTARG"
        ;;
    t )
        TP_SIZE="$OPTARG"
        ;;
    \? )
        usage
        exit 1
        ;;
  esac
done

# Parse list of configs.
IFS=$'\n' read -d '' -r -a MODEL_CONFIGS < "$CONFIG"

for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"
do
    LOCAL_SUCCESS=0
    
    echo "=== RUNNING MODEL: $MODEL_CONFIG WITH TP SIZE: $TP_SIZE==="

    export TEST_DATA_FILE=$PWD/configs/${MODEL_CONFIG}
    export TP_SIZE=$TP_SIZE
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
    export VLLM_SKIP_WARMUP=true
    export TQDM_BAR_FORMAT="{desc}: {percentage:3.0f}% {bar:10} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]" 
    JUNIT_FAMILY=""
    JUNIT_XML=""
    TIMEOUT_S=900 # 15 minutes timeout per test
    if [[ -n "$TEST_RESULTS_DIR" ]]; then
        LOG_DIR=$TEST_RESULTS_DIR
        LOG_FILENAME="test_${MODEL_CONFIG}.xml"
        LOG_PATH="${LOG_DIR}/${LOG_FILENAME}"
        JUNIT_FAMILY="-o junit_family=xunit1"
        JUNIT_XML="--junitxml=${LOG_PATH}"
    fi
    timeout $TIMEOUT_S pytest -s test_enc_dec_model.py "$JUNIT_FAMILY" "$JUNIT_XML" &
    TEST_PROCESS=$!
    wait $TEST_PROCESS
    LOCAL_SUCCESS=$?
    kill -9 $TEST_PROCESS 2> /dev/null || true
    if [[ $LOCAL_SUCCESS == 0 ]]; then
        echo "=== PASSED MODEL: ${MODEL_CONFIG} ==="
    else
        echo "=== FAILED MODEL: ${MODEL_CONFIG} ==="
    fi

    SUCCESS=$((SUCCESS + LOCAL_SUCCESS))

done

if [ "${SUCCESS}" -eq "0" ]; then
    exit 0
else
    exit 1
fi
