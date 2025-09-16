#!/bin/bash
model=ibm-granite/granite-3.3-2b-instruct

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

GRAPH_VISUALIZATION=true \
PT_HPU_METRICS_GC_DETAILS=1 \
PT_HPU_LAZY_MODE=1 \
vllm bench throughput \
  --model ${model} \
  --backend vllm \
  --dataset_path ShareGPT_V3_unfiltered_cleaned_split.json \
  --dataset_name sharegpt \
  --num-prompts 1000
