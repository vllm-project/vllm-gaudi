#granite-4.0-h-small for interactive with batch size 32


VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=false \
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

