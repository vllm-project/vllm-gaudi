VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
TRANSFORMERS_VERBOSITY=info \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
EXPERIMENTAL_WEIGHT_SHARING=0 \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_LAZY_MODE=0 \
python3 -m vllm.entrypoints.openai.api_server \
  --model=Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --port 8080 \
  --max-num-seqs=16 \
  --gpu-memory-util 0.9 \
  --tensor-parallel-size=1 \
  --max-model-len=131072 \
  --block-size=128 \
  --disable-log-requests \
  --async_scheduling \
  --disable-log-stats \
  --limit-mm-per-prompt '{""image"":20}' \
  --trust-remote-code 