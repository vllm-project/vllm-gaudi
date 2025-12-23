#export PT_HPU_SDPA_QKV_SLICE_MODE_FWD=true
#export PT_HPU_QKV_SLICE_SEQ_LEN_THLD=18432
export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_STEP=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_PROMPT_CTX_BUCKET_MIN=0
export VLLM_PROMPT_CTX_BUCKET_STEP=12
export VLLM_PROMPT_CTX_BUCKET_MAX=24
 
export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_STEP=4
export VLLM_DECODE_BS_BUCKET_MAX=64
#export VLLM_PROMPT_SEQ_BUCKET_MIN=1024
#export VLLM_PROMPT_SEQ_BUCKET_STEP=256
#export VLLM_PROMPT_SEQ_BUCKET_MAX=3072
#export VLLM_DECODE_BLOCK_BUCKET_MIN=128
#export VLLM_DECODE_BLOCK_BUCKET_STEP=256
#export VLLM_DECODE_BLOCK_BUCKET_MAX=1024
 
export VLLM_PROMPT_SEQ_BUCKET_MIN=5120
export VLLM_PROMPT_SEQ_BUCKET_STEP=1024
export VLLM_PROMPT_SEQ_BUCKET_MAX=20480
export VLLM_DECODE_BLOCK_BUCKET_MIN=256
export VLLM_DECODE_BLOCK_BUCKET_STEP=128
export VLLM_DECODE_BLOCK_BUCKET_MAX=1152
 
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_DELAYED_SAMPLING=true
export TRANSFORMERS_VERBOSITY=info
 
export VLLM_EXPONENTIAL_BUCKETING=False
export VLLM_FUSED_BLOCK_SOFTMAX_ADJUSTMENT=False
 
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING=0
export FUSER_ENABLE_LOW_UTILIZATION=true
export ENABLE_FUSION_BEFORE_NORM=true
export VLLM_SKIP_WARMUP=false
export VLLM_GRAPH_RESERVED_MEM=0.4
export VLLM_GRAPH_PROMPT_RATIO=0.4
export VLLM_CONTIGUOUS_PA=true
export VLLM_DEFRAG=true
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# export VLLM_PROFILER_ENABLED=1
# export VLLM_TORCH_PROFILER_DIR=/root/migration/vllm-gaudi/profile
# export VLLM_PROFILE_STEPS="1,2"
# export TORCH_LOGS="dynamo,graph_breaks"
# export TORCHDYNAMO_VERBOSE=1
ModelName=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8/snapshots/4bf2c2f39c37c0fede78bede4056e1f18cdf8109/
#ModelName=Qwen/Qwen3-VL-32B-Thinking-FP8
#ModelName=/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb/
#ModelName=/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e
MaxModelLen=32768
    # safest single-shot guess for your workload
  # VLLM_PROFILE_PROMPT=1,1024,0 
  python3 -m vllm.entrypoints.openai.api_server --host localhost --port 12346 --model $ModelName --trust-remote-code --tensor-parallel-size 1 \
    --max-model-len $MaxModelLen \
    --trust-remote-code \
    --gpu-memory-util 0.75 --max_num_batched_tokens 32768 --limit-mm-per-prompt '{"image": {"count": 20, "width": 864, "height": 480}}' \
    --mm-processor-kwargs '{"size": {"shortest_edge": 65536, "longest_edge": 1048576}}' 2>&1 | tee log_q3_remove.txt
 
 