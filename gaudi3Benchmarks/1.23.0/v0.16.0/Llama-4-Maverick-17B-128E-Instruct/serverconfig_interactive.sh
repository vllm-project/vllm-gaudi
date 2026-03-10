#Llama-4-Maverick-17B-128E-Instruct for interactive with batch size 16 with SLA TTFT <2s 

VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
RUNTIME_SCALE_PATCHING=1 \
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
VLLM_GRAPH_RESERVED_MEM=0.3 \
vllm serve \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --port 8090 \
  --max-num-seqs 1024 \
  --dtype bfloat16 \
  --gpu-memory-util 0.9 \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --block-size 128 \
  --disable-log-requests \
  --async-scheduling \
  --disable-log-stats \
  --quantization inc \
  --kv-cache-dtype fp8_inc \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --max-num-batched-tokens 8192 \
  --trust-remote-code 2>&1 | tee Llama-4-Maverick-17B-128E-Instruct.txt

# benchmark command
vllm bench serve \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --dataset-name random \
  --num-prompts 10240 \
  --random-input-len 4096 \
  --random-output-len 1024 \
  --max-concurrency 1024 \
  --port 8090 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --request-rate inf \
  --metric-percentiles 50,90,95,99 \
  --ready-check-timeout-sec 600 \
  --ignore-eos \
  --trust-remote-code
