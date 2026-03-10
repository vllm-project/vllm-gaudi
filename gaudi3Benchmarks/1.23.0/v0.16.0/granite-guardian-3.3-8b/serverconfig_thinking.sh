#ibm-granite/granite-guardian-3.3-8b with thinking= yes/no

# server
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=true \
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
VLLM_USE_V1=1 \
PT_HPU_LAZY_MODE=0 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
VLLM_FUSED_BLOCK_SOFTMAX=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_WEIGHT_SHARING=0 \
vllm serve \
  --model=ibm-granite/granite-guardian-3.3-8b \
  --port 8090 \
  --max-num-seqs=4 \
  --dtype=bfloat16 \
  --gpu-memory-util 0.9 \
  --tensor-parallel-size=1 \
  --max-model-len=131072 \
  --block-size=128 \
  --async-scheduling \
  --disable-log-requests \
  --disable-log-stats \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --override-generation-config '{"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512}'

#benchmark

vllm bench serve \
    --model ibm-granite/granite-guardian-3.3-8b \
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
    --temperature 0 \
    --save-result --save-detailed

