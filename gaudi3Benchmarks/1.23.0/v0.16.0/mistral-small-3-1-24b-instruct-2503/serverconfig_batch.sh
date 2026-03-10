#mistral-small-3-1-24b-instruct-2503 for interactive with batch size 6 - ttft <2s
export VLLM_SKIP_WARMUP=false
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPU_LAZY_MODE=0
export PT_HPU_WEIGHT_SHARING=0
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1

export VLLM_CONFIG_HIDDEN_LAYERS=1
export VLLM_WEIGHT_LOAD_FORCE_SYNC=0
export VLLM_USE_V1=1
export VLLM_CONTIGUOUS_PA=true
export VLLM_DEFRAG=true
export VLLM_FUSED_BLOCK_SOFTMAX=true
export VLLM_PROMPT_USE_FUSEDSDPA=1

export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=100000

export VLLM_EXPONENTIAL_BUCKETING=true
export VLLM_PROMPT_BS_BUCKET_MAX=2

vllm serve \
  --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --block-size 128 \
  --port 8090 \
  --max-num-seqs 6 \
  --max-model-len 131072 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-util 0.95 \
  --async-scheduling \
  --disable-log-requests \
  --no-enable-prefix-caching \
  --trust-remote-code

#benchmark script 

vllm bench serve \
  --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --dataset-name random \
  --num-prompts 180 \
  --random-input-len 4096 \
  --random-output-len 1024 \
  --max-concurrency 6 \
  --port 8090 \
  --request-rate inf \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --ignore-eos \
  --trust-remote-code
