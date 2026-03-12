# Set default MODEL if not defined
MODEL=${MODEL:-"/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/"}

VLLM_USE_HYBRID_CACHE=true \
VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=false \
VLLM_LOGGING_LEVEL=DEBUG \
TRANSFORMERS_VERBOSITY=info \
VLLM_EXPONENTIAL_BUCKETING=false \
VLLM_BUCKETING_STRATEGY=linear_bucketing \
VLLM_PROMPT_QUERY_BUCKET_MIN=128 \
VLLM_PROMPT_QUERY_BUCKET_STEP=128 \
VLLM_PROMPT_QUERY_BUCKET_MAX=256 \
VLLM_PROMPT_CTX_BUCKET_MIN=0 \
VLLM_PROMPT_CTX_BUCKET_STEP=1 \
VLLM_PROMPT_CTX_BUCKET_MAX=64 \
VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
EXPERIMENTAL_WEIGHT_SHARING=0 \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
VLLM_SKIP_WARMUP=true \
VLLM_GRAPH_RESERVED_MEM=0.4 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
python test.py \
  --model $MODEL \
  --mode video \
  --text-api generate \
  --max-model-len 16384 \
  --max-num-batched-tokens 16384 \
  --video-url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4" \
  2>&1 | tee log_video.txt
