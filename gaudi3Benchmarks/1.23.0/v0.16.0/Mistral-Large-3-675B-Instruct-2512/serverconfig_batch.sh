# mistralai/Mistral-Large-3-675B-Instruct-2512 example with batch size 8

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_FUSED_BLOCK_SOFTMAX=true \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
VLLM_EXPONENTIAL_BUCKETING=true \
vllm serve mistralai/Mistral-Large-3-675B-Instruct-2512 \
  --port 8110 \
  --max-num-seqs 2 \
  --max-model-len 131072 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --max_num_batched_tokens 4096 \
  --enable-expert-parallel \
  --disable-log-requests \
  --disable-log-stats 2>&1 | tee Mistral-Large-3-675B-Instruct-2512_server.txt


#benchmark command 
vllm bench serve \
--dataset-name random \
--model mistralai/Mistral-Large-3-675B-Instruct-2512  \
 --request-rate inf \
 --max-concurrency 2 \
 --endpoint /v1/completions \
 --host localhost \
 --port 9990 \
 --num-prompts 80 \
 --random-input-len 4096 \
 --random-output-len 1024 \
 --metric-percentiles 50,90,95,99 \
 --ignore-eos \
 --percentile-metrics ttft,tpot,itl,e2el

