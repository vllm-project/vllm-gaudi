# Qwen/Qwen3-VL-32B-Thinking-FP8 for interactive with batch size 16 with SLA TTFT <2s 

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
TRANSFORMERS_VERBOSITY=info \
VLLM_FP32_SOFTMAX=true \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
EXPERIMENTAL_WEIGHT_SHARING=0 \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_GRAPH_RESERVED_MEM=0.3 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --port 8090 \
  --max-num-seqs 16 \
  --gpu-memory-util 0.9 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --block-size 128 \
  --async-scheduling \
  --disable-log-requests \
  --disable-log-stats \
  --limit-mm-per-prompt '{"image": {"count": 20, "width": 864, "height": 480}}' \
  --no-enable-prefix-caching \
  --trust-remote-code 2>&1 | tee Qwen3-VL-32B-Thinking-FP8_server.txt

#benchamark command 
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen3-VL-32B-Instruct-Thinking-FP8 \
  --tokenizer Qwen/Qwen3-VL-32B-Thinking-FP8 \
  --dataset-name random-mm \
  --base-url http://localhost:8090 \
  --endpoint /v1/chat/completions \
  --num-prompts 160 \
  --random-input-len 6300 \
  --random-output-len 380 \
  --random-mm-base-items-per-request 20 \
  --random-mm-bucket-config "{(480, 864, 1): 1.0}" \
  --random-mm-num-mm-items-range-ratio 0.0 \
  --max-concurrency 8 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --trust-remote-code 2>&1 | tee Qwen3-VL-32B-Thinking-FP8benchmark.txt


