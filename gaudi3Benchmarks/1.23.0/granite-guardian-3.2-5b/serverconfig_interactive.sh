# ibm-granite/granite-guardian-3.2-5b batch mode(interactive) with batch size 1 

VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_CONFIG_HIDDEN_LAYERS=40 \
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
python3 -m vllm.entrypoints.openai.api_server \
  --model=ibm-granite/granite-guardian-3.2-5b \
  --port 8080 \
  --max-num-seqs=1 \
  --dtype=bfloat16 \
  --gpu-memory-util 0.95 \
  --tensor-parallel-size=1 \
  --max-model-len=131072 \
  --block-size=128 \
  --disable-log-requests \
  --async_scheduling \
  --disable-log-stats \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 9216 \
  --trust-remote-code 2>&1 | tee granite-guardian-3.2-5b_server.txt

