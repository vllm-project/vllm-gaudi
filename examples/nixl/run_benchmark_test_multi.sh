#!/bin/bash
set -xe

MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
)

export VLLM_USE_V1=1
export VLLM_SKIP_WARMUP=True
export PT_HPU_LAZY_MODE=1
export VLLM_EXPONENTIAL_BUCKETING=False
#export VLLM_PROMPT_BS_BUCKET_MIN=1
#export VLLM_PROMPT_SEQ_BUCKET_MIN=1
export VLLM_PROMPT_SEQ_BUCKET_MIN=8192
export VLLM_PROMPT_SEQ_BUCKET_STEP=8192
export VLLM_PROMPT_SEQ_BUCKET_MAX=8192
export VLLM_DECODE_BLOCK_BUCKET_MIN=1024
export VLLM_DECODE_BLOCK_BUCKET_MAX=1184
export VLLM_USE_PADDING_AWARE_SCHEDULING=1

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}


# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

# Handle to get model-specific arguments for deepseek
get_model_args() {
  local model_name=$1
  local extra_args=""

  if [[ "$model_name" == "deepseek-ai/deepseek-vl2-tiny" ]]; then
    extra_args="--hf_overrides '{\"architectures\": [\"DeepseekVLV2ForCausalLM\"]}' --trust-remote-code"
  fi

  echo "$extra_args"
}

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    echo "$($SMI_BIN --query-gpu=name --format=csv,noheader | wc -l)"
  else
    echo "$($SMI_BIN -l | grep GPU | wc -l)"
  fi
}

# Function to run tests for a specific model
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Arrays to store all hosts and ports
  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()

  # Start prefill instances
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do

    # Calculate port number (base port + instance number)
    PORT=$((8300 + i))
    # Calculate side channel port. Avoid clash with with TP workers.
    SIDE_CHANNEL_PORT=$((5559 + i))

    echo "Starting prefill instance $i on port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="HABANA_VISIBLE_DEVICES=0 RANK=0 UCX_TLS=gaudi_gdr,rc,ud,ib VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --long_prefill_token_threshold 8192 \
    --max_num_batched_tokens 8192 \
    --disable-log-requests \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config '{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"hpu\"},{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":1000,\"block_size\":128}}]}}' "


    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
  done

  # Start decode instances
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    # Calculate port number (base port + instance number)
    PORT=$((8400 + i))
    # Calculate side channel port
    SIDE_CHANNEL_PORT=$((4659 + i * $DECODER_TP_SIZE))

    echo "Starting decode instance $i on port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="HABANA_VISIBLE_DEVICES=1 RANK=1 UCX_TLS=gaudi_gdr,rc,ud,ib VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --gpu-memory-utilization 0.3 \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --long_prefill_token_threshold 8192 \
    --max_num_batched_tokens 8192 \
    --disable-log-requests \
    --kv-transfer-config '{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"hpu\"},{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"num_cpu_blocks\":1000,\"block_size\":128}}]}}' "


    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
  done

  # Wait for all instances to start
  for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server $PORT
  done

  for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server $PORT
  done

  # Build the command for the proxy server with all the hosts and ports
  PROXY_CMD="python toy_proxy_server.py --port 9111"

  # Add all prefill hosts and ports
  PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOSTS[@]}"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORTS[@]}"

  # Add all decode hosts and ports
  PROXY_CMD+=" --decoder-hosts ${DECODE_HOSTS[@]}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORTS[@]}"

  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &

  # Wait for the proxy to start
  sleep 5

curl -X POST -s http://localhost:9111/v1/completions \
	-H "Content-Type: application/json" \
	-d '{
	"model": "meta-llama/Llama-3.2-1B-Instruct",
	"prompt": "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.[2]",
	"max_tokens": 50,
	"temperature": 0
	}'
  sleep 2

  # Run lm eval for this model
  echo "Running tests for $model_name"
  TEST_MODEL=$model_name python -m pytest -s -x test_accuracy.py

  sleep 100

  qps=(0.5) #(0.1 0.25 0.5 1 2 3 4) # 5)
  # explicit num_prompts mapping (must have same length as qps[])
  num_prompts=(32) #(32 64 128 256 256 256 256) # 256)
  input_len=8192
  output_len=256 #56

  # just sanity‐check lengths
  if [ "${#qps[@]}" -ne "${#num_prompts[@]}" ]; then
    echo "❌ qps[] and num_prompts[] must be the same length"
    exit 1
  fi

  for i in "${!qps[@]}"; do
    q=${qps[$i]}
    np=${num_prompts[$i]}

    ts=$(date +"%Y%m%d_%H%M%S")
    logf="./run_in${input_len}_out${output_len}_qps${q//./p}_$ts.log"

    echo "[$(date +"%Y-%m-%d %H:%M:%S")] input=${input_len}, output=${output_len}, qps=${q}, num_prompts=${np}" \
      | tee "$logf"

    python3 vllm serve \
      --port 9111 \
      --seed "$(date +%s)" \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --tokenizer meta-llama/Llama-3.2-1B-Instruct \
      --dataset-name random \
      --random-input-len "$input_len" \
      --random-output-len 256 \
      --num-prompts "$np" \
      --request-rate "$q" \
      --percentile-metrics ttft,tpot,itl,e2el \
      --burstiness 100 \
      --backend openai \
      --endpoint /v1/completions \
      --ignore-eos \
      2>&1 | tee -a "$logf"

  done

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All tests completed!"
