#Llama-3.3-70B-Instruct for interactive with batch size 128 

VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_FUSED_BLOCK_SOFTMAX=true \
VLLM_HANDLE_TOPK_DUPLICATES=true \
QUANT_CONFIG="/root/llama-3.3-70b-instruct-2x/maxabs_quant_g3.json" \
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --quantization inc \
    --kv-cache-dtype fp8_inc \
    --tensor_parallel_size 2 \
    --max-num-seqs 128 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.9 \
    --tensor-parallel-size=2 \
    --max-model-len=131072 \
    --block-size=128 \
    --disable-log-requests \
    --async_scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --trust_remote_code 2>&1 | tee Llama-3.3-70B-Instruct.txt


