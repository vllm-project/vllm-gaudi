#Llama-4-Maverick-17B-128E-Instruct for interactive with batch size 16 with SLA TTFT <2s 

QUANT_CONFIG=/root/llama-4-maverick-17b-128e-instruct/maxabs_quant_g3.json \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
RUNTIME_SCALE_PATCHING=1 \
PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1 \
VLLM_PROMPT_USE_FUSEDSDPA=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_LAZY_MODE=0 \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=false \
VLLM_DEFRAG=false \
VLLM_FUSED_BLOCK_SOFTMAX=true \
VLLM_GRAPH_RESERVED_MEM=0.05 \
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --port 8090 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --gpu-memory-util 0.9 \
  --max-model-len 131072 \
  --block-size 128 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --async_scheduling \
  --quantization inc \
  --kv-cache-dtype fp8_inc \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --disable-log-requests \
  --disable-log-stats \
  --trust-remote-code 2>&1 | tee Llama-3.3-70B-Instruct.txt

