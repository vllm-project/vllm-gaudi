# Llama-Guard-4-12B for batch mode  with batch size 80, gaurd model ensure the latency to be range

VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
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
vllm serve  meta-llama/Llama-Guard-4-12B \
    --port 8080 \
    --tensor-parallel-size 1 \
    --max-num-seqs 80 \
    --dtype=bfloat16 \
    --gpu-memory-util 0.95 \
    --tensor-parallel-size=1 \
    --max-model-len=131072 \
    --block-size=128 \
    --disable-log-requests \
    --async_scheduling \
    --disable-log-stats \
    --no-enable-prefix-caching \
    --trust_remote_code 2>&1 | tee Llama-Guard-4-12B_serverlog.txt


