PT_HPU_LAZY_MODE=1 \
vllm serve --model Qwen/Qwen3-VL-8B-Instruct \
    --block-size 128 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --max-model-len 16384 \
    --trust-remote-code 2>&1 | tee log_q3_GAUDISW-245825.txt