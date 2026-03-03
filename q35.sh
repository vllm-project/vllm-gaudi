
export VLLM_EXPONENTIAL_BUCKETING=true
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DELAYED_SAMPLING=true
export TRANSFORMERS_VERBOSITY=info
export VLLM_EXPONENTIAL_BUCKETING=False
export VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_CONTIGUOUS_PA=true
export VLLM_DEFRAG=true
export VLLM_USE_HYBRID_CACHE=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=false
export HABANA_VISIBLE_DEVICES=0,1
#export ModelName="/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8/snapshots/4bf2c2f39c37c0fede78bede4056e1f18cdf8109/"
export ModelName="Qwen/Qwen3.5-35B-A3B-FP8"
export MaxModelLen=-1
vllm serve $ModelName --port 12346 --tensor-parallel-size 1 \
    --max-model-len $MaxModelLen \
    --trust-remote-code \
    --gpu-memory-util 0.9 --limit-mm-per-prompt '{"image": {"count": 20, "width": 864, "height": 480}}' \
    --mm-processor-kwargs '{"size": {"shortest_edge": 65536, "longest_edge": 1048576}}' 2>&1 | tee log_q3_libin_noprefix.txt
