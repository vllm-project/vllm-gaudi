export VLLM_LOGGING_LEVEL=DEBUG
export TRANSFORMERS_VERBOSITY=info
export VLLM_EXPONENTIAL_BUCKETING=false
export VLLM_BUCKETING_STRATEGY=linear_bucketing
# Hybrid GDN/Mamba aligns prompt query buckets to mamba_chunk_size (2048).
# Keep query bucket config consistent to avoid generating zero valid buckets.
export VLLM_PROMPT_QUERY_BUCKET_MIN=128
export VLLM_PROMPT_QUERY_BUCKET_STEP=128
export VLLM_PROMPT_QUERY_BUCKET_MAX=256
export VLLM_PROMPT_CTX_BUCKET_MIN=0
export VLLM_PROMPT_CTX_BUCKET_STEP=1
export VLLM_PROMPT_CTX_BUCKET_MAX=64
export VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true
export VLLM_SKIP_WARMUP=true
export VLLM_GRAPH_RESERVED_MEM=0.1
# export VLLM_CONTIGUOUS_PA=true
# export VLLM_DEFRAG=true
export VLLM_USE_HYBRID_CACHE=true
export VLLM_USE_NAIVE_MAMBA_CACHE_SHARING=false
export MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b/
python test.py \
  --model $MODEL \
  --mode text \
  --text-api generate \
  --tensor-parallel-size 8 \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 2>&1 | tee log_8x.txt
