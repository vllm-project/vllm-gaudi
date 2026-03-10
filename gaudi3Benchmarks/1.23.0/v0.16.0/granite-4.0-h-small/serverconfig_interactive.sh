#granite-4.0-h-small for interactive with batch size 32


VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=false \
export VLLM_BUCKET_FILENAME=$(mktemp) && \
cat > "$VLLM_BUCKET_FILENAME" <<'BUCKETS'
(1, [256, 512, 1024, 2048, 4096, 8192], [0, 1, 2, 4, 8, 16])
(1, [512, 2048, 8192], [32, 48, 64])
(2, 1, [2, 4, 8, 16, 32, 64, 128])
(8, 1, [8, 16, 32, 64, 128, 256, 512])
(16, 1, [16, 32, 64, 128, 256, 512, 1024])
(32, 1, [32, 64, 128, 256, 512, 1024, 2048])
BUCKETS
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
 
vllm serve ibm-granite/granite-4.0-h-small \
    --block-size 128 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max_model_len 32768 \
    --gpu_memory_util 0.5 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
    --async-scheduling 2>&1 | tee granite-4.0-h-small_server.txt


#benchmark script 

vllm bench serve \
  --model ibm-granite/granite-4.0-h-small \
  --dataset-name random \
  --num-prompts 160 \
  --random-input-len 4096 \
  --random-output-len 1024 \
  --max-concurrency 32 \
  --port 8090 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --ignore-eos \
  --trust-remote-code

