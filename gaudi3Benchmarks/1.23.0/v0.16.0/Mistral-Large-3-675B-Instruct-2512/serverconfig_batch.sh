# mistralai/Mistral-Large-3-675B-Instruct-2512 example with batch size 8

VLLM_PROMPT_BS_BUCKET_MAX=2 \
VLLM_PROMPT_CTX_BUCKET_STEP=32 \
VLLM_PROMPT_QUERY_BUCKET_MIN=1024 \
VLLM_PROMPT_QUERY_BUCKET_STEP=1024 \
VLLM_PROMPT_QUERY_BUCKET_MAX=4096 \
VLLM_DECODE_BS_BUCKET_MAX=8 \
VLLM_DECODE_BLOCK_BUCKET_MIN=256 \
VLLM_DECODE_BLOCK_BUCKET_STEP=256 \
VLLM_DECODE_BLOCK_BUCKET_MAX=8192 \
VLLM_EXPONENTIAL_BUCKETING=false \
vllm serve mistralai/Mistral-Large-3-675B-Instruct-2512 \
--port 9990 \
--max-num-seqs 8 \
--max-model-len 131072 \
--tensor-parallel-size 8 \
--gpu-memory-utilization 0.95 \
--max_num_batched_tokens 4096 \
--enable-expert-parallel \
--disable-log-requests \
--disable-log-stats


#benchmark command 
vllm bench serve \
--dataset-name random \
--model mistralai/Mistral-Large-3-675B-Instruct-2512  \
 --request-rate inf \
 --max-concurrency 8 \
 --endpoint /v1/completions \
 --host localhost \
 --port 9990 \
 --num-prompts 240 \
 --random-input-len 4096 \
 --random-output-len 1024 \
 --metric-percentiles 50,90,95,99 \
 --ignore-eos \
 --percentile-metrics ttft,tpot,itl,e2el

