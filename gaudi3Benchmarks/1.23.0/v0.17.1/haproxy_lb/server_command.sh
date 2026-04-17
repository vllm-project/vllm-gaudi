export HF_HUB_OFFLINE=0 \
export VLLM_BUILD=1.23.0.0 \
export VLLM_BUCKET_FILENAME=$(mktemp) && \
cat > "$VLLM_BUCKET_FILENAME" <<'BUCKETS'
(1, [256, 512, 1024, 2048, 4096, 8192], [0, 1, 2, 4, 8, 16])
(1, [512, 2048, 8192], [32, 48, 64])
(2, 1, [2, 4, 8, 16, 32, 64, 128])
(8, 1, [8, 16, 32, 64, 128, 256, 512])
(16, 1, [16, 32, 64, 128, 256, 512, 1024])
(32, 1, [32, 64, 128, 256, 512, 1024, 2048])
BUCKETS
VLLM_API_KEY=granite4.0h-g3key \
VLLM_CONTIGUOUS_PA=false \
VLLM_GRAPH_RESERVED_MEM=0.3 \
VLLM_BUCKETING_FROM_FILE="$VLLM_BUCKET_FILENAME" \
vllm serve ibm-granite/granite-4.0-h-small \
--override-generation-config '{"temperature": 0}' \
--block-size 128 \
--dtype bfloat16 \
--tensor-parallel-size 1 \
--max_model_len 43008 \
--gpu_memory_util 0.5 \
--max-num-seqs 32 \
--max-num-batched-tokens 8192 \
--enable-chunked-prefill \
--port 30360 \
--no-enable-prefix-caching \
--tool-call-parser hermes \
--enable-auto-tool-choice
