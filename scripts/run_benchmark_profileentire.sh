#!/bin/bash
model=/mnt/weka/data/pytorch/llama3/Meta-Llama-3-8B-Instruct/
QUANT_CONFIG=/software/users/chendixu/dev/vllm-gaudi/tests/models/language/generation/inc_unit_scale_quant.json
bs=128
num_prompts=512
max_concurrency=32

model_name=$(basename "${model}")
log_name="test1-online-${model_name}-kakao-profileentire"

hl-prof-config --use-template profile_api --hw-trace off 

mkdir -p benchmark_logs

# Define the directory and file path
DIRECTORY="benchmarks"
FILE_PATH="ShareGPT_V3_unfiltered_cleaned_split.json"

# --dataset-name hf --dataset-path mgoin/mlperf-inference-llama2-data

# Define the URL for the file to be downloaded
URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

# Check if the file already exists at the specified path
if [ -f "$FILE_PATH" ]; then
    # If the file exists, print a message and exit
    echo "File '$FILE_PATH' already exists. No download needed."
else
    # If the file does not exist, proceed with the download
    echo "File '$FILE_PATH' not found."

    # Download the file using wget
    echo "Downloading file from $URL..."
    wget -O "$FILE_PATH" "$URL"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful. File saved to '$FILE_PATH'."
    else
        echo "Error during download. Please check the URL or your network connection."
    fi
fi

#--max_num_batched_tokens 256 \
HABANA_PROFILE=1 \
VLLM_TORCH_PROFILER_DIR="benchmark_logs/${log_name}_profiler" \
VLLM_PROFILER_ENABLED=full \
VLLM_SOFTMAX_CONST_NORM=true \
QUANT_CONFIG=${QUANT_CONFIG} \
PT_HPU_LAZY_MODE=1 \
VLLM_CONFIG_HIDDEN_LAYERS=32 \
VLLM_DELAYED_SAMPLING=true \
VLLM_GRAPH_RESERVED_MEM=0.05 \
VLLM_EXPONENTIAL_BUCKETING=false \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=1 \
VLLM_PROMPT_SEQ_BUCKET_MIN=128 \
VLLM_PROMPT_SEQ_BUCKET_STEP=128 \
VLLM_PROMPT_SEQ_BUCKET_MAX=1280 \
VLLM_DECODE_BS_BUCKET_MIN=1 \
VLLM_DECODE_BS_BUCKET_MAX=32 \
VLLM_DECODE_BLOCK_BUCKET_MIN=32 \
VLLM_DECODE_BLOCK_BUCKET_STEP=8 \
VLLM_DECODE_BLOCK_BUCKET_MAX=128 \
python3 -m vllm.entrypoints.openai.api_server \
--port 18080 \
--model ${model} \
--tensor-parallel-size 1 \
--dtype bfloat16 \
--max-num-seqs 32 \
--block_size 128 \
--max-model-len 4096 \
--disable-log-requests \
--quantization inc \
--kv-cache-dtype fp8_inc 2>&1 | tee benchmark_logs/${log_name}_serving.log &
#Clients: 32, Prompt (mean): 233 tokens, Generation (mean): 169 tokens, Query throughput: 23.638 queries/s, Token throughput (total): 1662.074 tokens/s, Query latency: 1.354 s, Token generation latency: 0.008 s/token, First token received: 0.048 s

# now
#Clients: 32, Prompt (mean): 233 tokens, Generation (mean): 169 tokens, Query throughput: 26.811 queries/s, Token throughput (total): 1895.079 tokens/s, Query latency: 1.194 s, Token generation latency: 0.007 s/token, First token received: 0.048 s

# H200 took 0.004 s/token => 0.006 for G3

pid=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
sleep 10s
echo ${pid}


start_time=$(date +%s)
echo "Start to benchmark"


# How to run
# sharegpt dataset
curl -X POST http://localhost:18080/start_profile
python ./run_benchmark.py \
    --model_name llama3-8b-inst \
    --model_path ${model} \
    --port 18080 \
    --ip 127.0.0.1 \
    --tp_size 1 \
    --max_prompt_length 3072 \
    --use_chat_template \
    --num_clients 32 \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --out_json_dir benchmark_logs/${log_name}_json \
    --output_dir gaudi3 \
    --backend vllm \
    --overwrite_results 2>&1 | tee benchmark_logs/${log_name}_run1.log
curl -X POST http://localhost:18080/stop_profile

python ./run_benchmark.py \
    --model_name llama3-8b-inst \
    --model_path ${model} \
    --port 18080 \
    --ip 127.0.0.1 \
    --tp_size 1 \
    --max_prompt_length 3072 \
    --use_chat_template \
    --num_clients 32 \
    --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
    --out_json_dir benchmark_logs/${log_name}_json \
    --output_dir gaudi3 \
    --backend vllm \
    --overwrite_results 2>&1 | tee benchmark_logs/${log_name}_run2.log

end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

kill ${pid}