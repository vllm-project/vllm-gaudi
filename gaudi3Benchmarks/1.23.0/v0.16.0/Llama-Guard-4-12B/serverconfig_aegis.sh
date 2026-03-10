#meta-llama/Llama-Guard-4-12B with aegis example 
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=100000
export VLLM_EXPONENTIAL_BUCKETING=true
export VLLM_WEIGHT_LOAD_FORCE_SYNC=1
export VLLM_USE_V1=1
export PT_HPU_LAZY_MODE=0
export VLLM_CONTIGUOUS_PA=true
export VLLM_DEFRAG=true
export VLLM_FUSED_BLOCK_SOFTMAX=true

# HPU/Habana Specific Configurations
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true
export PT_HPU_WEIGHT_SHARING=0

# Start the model server
vllm serve \
  --model=meta-llama/Llama-Guard-4-12B \
  --port=8090 \
  --max-num-seqs=4 \
  --dtype=bfloat16 \
  --gpu-memory-util=0.9 \
  --tensor-parallel-size=1 \
  --max-model-len=131072 \
  --block-size=128 \
  --async-scheduling \
  --disable-log-requests \
  --disable-log-stats \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --override-generation-config '{"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 20}'

#benchmark
vllm bench serve \
  --model meta-llama/Llama-Guard-4-12B \
  --dataset-name custom \
  --dataset-path /tmp/aegis-benchmark.jsonl \
  --base-url http://localhost:8090 \
  --num-prompts 40 \
  --max-concurrency 4 \
  --request-rate inf \
  --save-result \
  --save-detailed \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --skip-chat-template
