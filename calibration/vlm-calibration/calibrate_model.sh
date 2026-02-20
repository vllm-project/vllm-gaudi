#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

set -e
pushd "$(dirname "$0")" > /dev/null
trap 'popd > /dev/null' EXIT

ALLOWED_DEVICES=("g2" "g3")

usage() {
    echo
    echo "Calibrate given MODEL_PATH for FP8 inference"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -m    - [required] huggingface stub or local directory of the MODEL_PATH"
    echo "  -d    - [optional] path to source dataset (details in README). If not provided, the dataset will be downloaded from HuggingFace."
    echo "  -o    - [required] path to output directory for fp8 measurements"
    echo "  -b    - batch size to run the measurements at (default: 32)"
    echo "  -l    - limit number of samples in calibration dataset"
    echo "  -t    - tensor parallel size to run at (default: 1); NOTE: if t > 8 then we need a multi-node setup"
    echo "  -r    - rank of unified measurements, it should be smaller than original rank number and should be a factor of the original rank number"
    echo "  -u    - unify measurement results based on expert parallelism rules (default: False), expert parallelism unification rule is unique, card 1 expert measurement will be extended to card 0 if unified to x from 2x cards number"
    echo "  -e    - Turn on or off eager mode, default: off"
    echo
}

cleanup_tmp() {
	if [[ $(pwd) == *vlm-calibration ]]; then
		echo "Clearing temporary directory"
        mkdir -p inc_tmp nc_workspace
		rm -rf nc_workspace
		rm -rf inc_tmp
	else
		echo "Skipping temporary directory removal"
	fi
}

create_measure_config() {
    mkdir -p $1/$2/$3

    model_name_lower=$(echo "$2" | tr '[:upper:]' '[:lower:]')

    tmp_config="{\"method\": \"HOOKS\",\"mode\": \"MEASURE\",\"observer\": \"maxabs\",\"allowlist\": {\"types\": [], \"names\":  []},\"blocklist\": {\"types\": [], \"names\":  [\"lm_head\"]},\"quantize_weight\": false,\"dump_stats_path\": \"$1/$2/$3/inc_output\",\"calibration_sample_interval\": 1}"
    
    echo "$tmp_config" > $1/$2/maxabs_measure_$3.json
}

create_quant_config() {
    mkdir -p $1/$2/$3
    
    model_name_lower=$(echo "$2" | tr '[:upper:]' '[:lower:]')

    tmp_config="{\"mode\": \"QUANTIZE\",\"observer\": \"maxabs\",\"scale_method\": \"maxabs_hw\",\"allowlist\": {\"types\": [],\"names\": []},\"blocklist\": {\"types\": [],\"names\": [\"lm_head\"]},\"dump_stats_path\": \"$1/$2/$3/inc_output\"}"
    
    echo "$tmp_config" > $1/$2/maxabs_quant_$3.json
}

extract_last_folder_name() {
    local path="$1"

    path="${path%/}"
    last_folder="$(basename "$path")"
    last_folder="${last_folder,,}"

    echo "$last_folder"
}

cleanup_tmp

echo "downloading requirements..."
pip install -r requirements.txt 

EXTRA_FLAGS=""
BATCH_SIZE=32
TP_SIZE=1
eager_mode="off"
RANK=""
USE_EP=""
while getopts "m:b:l:t:d:h:o:r:u:e" OPT; do
    case ${OPT} in
        m )
            MODEL_PATH="$OPTARG"
            ;;
        d )
            DATASET_PATH="$OPTARG"
            ;;
        b )
            BATCH_SIZE="$OPTARG"
            ;;
        o )
            FP8_DIR=$(realpath -m "$OPTARG")
            ;;
        l )
            LIMIT="$OPTARG"
            ;;
        t )
            TP_SIZE="$OPTARG"
            ;;
        r )
            RANK="$OPTARG"
            ;;        
        u )
            USE_EP="--use_expert_paral"
            ;;
        h )
            usage
            ;;
        e )
            eager_mode="$OPTARG"
            ;;
        \? )
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" && -z "$FP8_DIR" ]]; then
    echo "Model stub and output path for fp8 measurements must be provided."
    usage
    exit 1
fi

if [[ -z "$DATASET_PATH" ]]; then
    echo "Local calibration dataset path not provided. Will download it from HuggingFace."
else
    echo "Using local calibration dataset path: $DATASET_PATH"
    if [[ -d "$DATASET_PATH/hub/datasets--MMMU--MMMU" && -d "$DATASET_PATH/datasets/MMMU___mmmu" ]]; then
        export HF_HOME="/root/.cache/huggingface"
        echo "copying local calibration dataset $DATASET_PATH to $HF_HOME"
        mkdir -p $HF_HOME "$HF_HOME/hub" "$HF_HOME/datasets"
        cp -rf "$DATASET_PATH/hub/datasets--MMMU--MMMU" "$HF_HOME/hub"
        cp -rf "$DATASET_PATH/datasets/MMMU___mmmu" "$HF_HOME/datasets"
    elif [[ -d "$DATASET_PATH/MMMU___mmmu" ]]; then
        export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
        echo "copying local calibration dataset $DATASET_PATH to $HF_DATASETS_CACHE"
        mkdir -p $HF_DATASETS_CACHE
        cp -rf "$DATASET_PATH/MMMU___mmmu" $HF_DATASETS_CACHE
    else
        echo "Your provided dataset path doesn't contain MMMU dataset. Please refer to README for details."
        exit 1
    fi
fi


if [[ $eager_mode == "on" ]]; then
    EXTRA_FLAGS+="--enforce-eager "
fi

if [[ -n $USE_EP ]]; then
    EXTRA_FLAGS+="--expert-parallel "
fi

# Store the provided MODEL_PATH name in a variable
MODEL_NAME=$(extract_last_folder_name "$MODEL_PATH")

echo ""
echo "Step 1/3 - detecting used device type ${ALLOWED_DEVICES[*]}"
python3 step-0-detect-device.py > /dev/null  || DEVICE_TYPE=$?
DEVICE_TYPE="g$DEVICE_TYPE"
# Check if the provided device type is valid
if [[ ! " ${ALLOWED_DEVICES[*]} " =~ " $DEVICE_TYPE " ]]; then
    echo "Invalid device type: $DEVICE_TYPE. Allowed devices: ${ALLOWED_DEVICES[*]}"
    exit 1
fi
echo "Detected device type: $DEVICE_TYPE"
echo "Step 1 done"

create_measure_config $FP8_DIR $MODEL_NAME $DEVICE_TYPE
create_quant_config $FP8_DIR $MODEL_NAME $DEVICE_TYPE

if [[ $TP_SIZE > 1 ]]; then
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
fi
export VLLM_SKIP_WARMUP=true
max_model_len=8192


echo ""
echo "2/3 Measuring scales"
export QUANT_CONFIG=$FP8_DIR/$MODEL_NAME/maxabs_measure_$DEVICE_TYPE.json
# quantization='None'
# kv_cache_dtype='auto'
quantization='inc'
kv_cache_dtype='auto'  # (afierka) TODO: we want to switch to fp8_inc for kv cache as well, but it causes instability for some models, need to investigate further

python3 vision_lm_eval.py \
    --max-model-len $max_model_len \
    --model-path $MODEL_PATH \
    --quantization $quantization \
    --kv-cache-dtype $kv_cache_dtype \
    --tensor-parallel-size $TP_SIZE \
    $EXTRA_FLAGS
echo "Step 2/3 done"


echo ""
echo "3/3 Quantize scales"
export QUANT_CONFIG=$FP8_DIR/$MODEL_NAME/maxabs_quant_$DEVICE_TYPE.json
quantization='inc'
kv_cache_dtype='fp8_inc'

python3 vision_lm_eval.py \
    --max-model-len $max_model_len \
    --model-path $MODEL_PATH \
    --quantization $quantization \
    --kv-cache-dtype $kv_cache_dtype \
    --tensor-parallel-size $TP_SIZE \
    $EXTRA_FLAGS

echo "Step 3/3 done"



if [[ -n $RANK ]]; then
    echo ""
    echo "Unify scales"
    QUANT_DIR=$FP8_DIR/$MODEL_NAME/$DEVICE_TYPE/
    python3 ../step-5-unify_measurements.py -r "$RANK" -m $QUANT_DIR -o $QUANT_DIR $USE_EP || (echo "Error in step 5" && exit 1)
    echo "Unify scales done"
fi
cleanup_tmp
echo "Calibration process done"