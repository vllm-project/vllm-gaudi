#granite-4.0-h-small for interactive with batch size 32 with SLA TTFT-P99 <2s 

#granite4.0h vllm server command
VLLM_EXPONENTIAL_BUCKETING=false \
VLLM_PROMPT_BS_BUCKET_MAX=1 \
VLLM_PROMPT_SEQ_BUCKET_MIN=128 \
VLLM_PROMPT_SEQ_BUCKET_STEP=1024 \
VLLM_PROMPT_SEQ_BUCKET_MAX=4096 \
VLLM_DECODE_BS_BUCKET_MIN=32 \
VLLM_DECODE_BS_BUCKET_STEP=32 \
VLLM_DECODE_BS_BUCKET_MAX=32 \
VLLM_DECODE_BLOCK_BUCKET_MIN=1088 \
VLLM_DECODE_BLOCK_BUCKET_MAX=1280 \
VLLM_DECODE_BLOCK_BUCKET_STEP=64 \
VLLM_PROMPT_CTX_BUCKET_MIN=0 \
VLLM_PROMPT_CTX_BUCKET_STEP=31 \
VLLM_PROMPT_CTX_BUCKET_MAX=31 \
VLLM_CONTIGUOUS_PA=true \
VLLM_SKIP_WARMUP=true \
vllm serve ibm-granite/granite-4.0-h-small \
    --max-num-seqs 32 \
    --block-size 128 \
    --disable-log-stats \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --load-format dummy \
    --port 30360 \
    --gpu-memory-utilization 0.45 \
    --no-enable-prefix-caching \
    --async-scheduling 2>&1 | tee granite-4.0-h-small_server.txt

#benchmark script 
python -m vllm.entrypoints.cli.main bench serve \
    --backend vllm \
    --model ibm-granite/granite-4.0-h-small  \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --ignore-eos \
    --num-prompts 32 \
    --port 8000 \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 1024 \
    --max-concurrency 32 2>&1 | tee granite-4.0-h-small_benchmark.txt


