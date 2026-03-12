export MODEL=/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b/
export VLLM_USE_HYBRID_CACHE=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=false

VLLM_SKIP_WARMUP=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=true VLLM_DEFRAG=true \
VLLM_EXPONENTIAL_BUCKETING=true VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 VLLM_RPC_TIMEOUT=100000 \
EXPERIMENTAL_WEIGHT_SHARING=0 FUSER_ENABLE_LOW_UTILIZATION=true ENABLE_FUSION_BEFORE_NORM=true \
VLLM_GRAPH_RESERVED_MEM=0.5 numactl --cpunodebind=0 --membind=0 \
python3 -m vllm.entrypoints.openai.api_server --model $MODEL --port 12346 \
--block-size 128 --max-num-seqs 1 --max-model-len 10240 \
--async-scheduling --no-enable-prefix-caching --trust-remote-code --tensor-parallel-size 8 --enable-expert-parallel \
--limit-mm-per-prompt '{"image": {"count": 1, "width": 864, "height": 480}}' 2>&1 | tee server.log
 
