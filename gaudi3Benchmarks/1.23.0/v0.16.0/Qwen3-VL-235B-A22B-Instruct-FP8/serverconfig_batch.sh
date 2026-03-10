# Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 for batch mode with batch size 8 for max perf 

export VLLM_ENGINE_ITERATION_TIMEOUT_S=3600
export VLLM_RPC_TIMEOUT=100000
export VLLM_DELAYED_SAMPLING=true
export VLLM_EXPONENTIAL_BUCKETING=true
export VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true
export VLLM_SKIP_WARMUP=false
export VLLM_GRAPH_PROMPT_RATIO=0.3
export VLLM_CONTIGUOUS_PA=true
vllm serve \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --port 8090 \
  --max-num-seqs 8 \
  --max_num_batched_tokens 32768 \
  --gpu-memory-util 0.65 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --disable-log-requests \
  --enable-expert-parallel \
  --async-scheduling \
  --disable-log-stats \
  --limit-mm-per-prompt '{"image": {"count": 20, "width": 864, "height": 480}}' \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --trust-remote-code 2>&1 | tee Qwen3-VL-235B-A22B-Instruct-FP8_server.txt

#benchamark command 
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tokenizer Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --dataset-name random-mm \
  --base-url http://localhost:8090 \
  --endpoint /v1/chat/completions \
  --num-prompts 20 \
  --random-input-len 6300 \
  --random-output-len 380 \
  --random-mm-base-items-per-request 20 \
  --random-mm-bucket-config "{(480, 864, 1): 1.0}" \
  --random-mm-num-mm-items-range-ratio 0.0 \
  --max-concurrency 8 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --trust-remote-code 2>&1 | tee Qwen3-VL-235B-A22B-Instruct-FP8_benchmark.txt


