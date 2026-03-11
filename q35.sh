export MODEL=/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8/snapshots/4bf2c2f39c37c0fede78bede4056e1f18cdf8109/
VLLM_SKIP_WARMUP=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=true VLLM_DEFRAG=true \
VLLM_EXPONENTIAL_BUCKETING=true VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 VLLM_RPC_TIMEOUT=100000 \
EXPERIMENTAL_WEIGHT_SHARING=0 FUSER_ENABLE_LOW_UTILIZATION=true ENABLE_FUSION_BEFORE_NORM=true \
VLLM_GRAPH_RESERVED_MEM=0.5 numactl --cpunodebind=0 --membind=0 \
python3 -m vllm.entrypoints.openai.api_server --model $MODEL --port 12346 --tensor-parallel-size 2 \
--block-size 128 --max-num-seqs 1 --max-model-len 10240 --enforce-eager \
--async-scheduling --no-enable-prefix-caching --trust-remote-code \
--limit-mm-per-prompt '{"image": {"count": 1, "width": 864, "height": 480}}' 2>&1 | tee server.log
 
