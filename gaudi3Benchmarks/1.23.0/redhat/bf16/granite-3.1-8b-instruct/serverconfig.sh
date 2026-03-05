#RedHatAI/granite-3.1-8b-instruct with guidellm

#vllm serve command 


VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_FUSED_BLOCK_SOFTMAX=true \
python3 -m vllm.entrypoints.openai.api_server \
  --model RedHatAI/granite-3.1-8b-instruct \
  --port 8110 \
  --max-num-seqs 128 \
  --dtype bfloat16 \
  --gpu-memory-util 0.9 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --block-size 128 \
  --disable-log-requests \
  --async-scheduling \
  --disable-log-stats \
  --no-enable-prefix-caching \
  --trust-remote-code \
  > /tmp/pt_vllm_RedHatAI_granite-3.1-8b-instruct_3_8110.log

#benchmark command 

python3 -m venv venv
source venv/bin/activate

pip install guidellm

# Remove proxy variables
unset http_proxy
unset https_proxy
export no_proxy=localhost,127.0.0.1

# Increase file descriptors for high concurrency
ulimit -n 1048576

guidellm benchmark \
  --target "http://localhost:8110" \
  --model RedHatAI/granite-3.1-8b-instruct \
  --rate-type concurrent \
  --rate 1,5,10,25,50,100 \
  --max-seconds 30 \
  --data "prompt_tokens=1024,output_tokens=1024" \
  | tee benchmark_core_RedHatAI_granite-3.1-8b-instruct-input_len-1024-output_len-1024-batch_size-128-device-1-port-8110-build_id-4.log

