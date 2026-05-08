#!/bin/bash
set -e

# =========================
# SERVER CONFIG
# =========================

export VLLM_HPU_MULTI_MODEL_CONFIG=multi_models.yaml && \
cat > "$VLLM_HPU_MULTI_MODEL_CONFIG" << 'EOF'
default_model: meta-llama/Llama-Guard-4-12B
models:
  meta-llama/Llama-Guard-4-12B:
    model: meta-llama/Llama-Guard-4-12B
    tensor_parallel_size: 1
    max_num_seqs: 4
    dtype: bfloat16
    block_size: 128
    max_model_len: 131072
    async_scheduling: True
    enable_prefix_caching: False
    gpu_memory_utilization: 0.95
    max_num_batched_tokens: 8192
    override_generation_config:
      temperature: 0.0
      top_p: 1.0
      max_new_tokens: 20
  
  ibm-granite/granite-guardian-3.3-8b:
    model: ibm-granite/granite-guardian-3.3-8b
    tensor_parallel_size: 1
    max_num_seqs: 4
    dtype: bfloat16
    block_size: 128
    max_model_len: 131072
    async_scheduling: True
    enable_prefix_caching: False
    gpu_memory_utilization: 0.90
    override_generation_config:
      temperature: 0.0
      top_p: 1.0
      max_new_tokens: 512
EOF

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
VLLM_RPC_TIMEOUT=100000 \
VLLM_EXPONENTIAL_BUCKETING=true \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
FUSER_ENABLE_LOW_UTILIZATION=true \
ENABLE_FUSION_BEFORE_NORM=true \
PT_HPU_LAZY_MODE=0 \
VLLM_USE_V1=1 \
VLLM_CONTIGUOUS_PA=true \
VLLM_DEFRAG=true \
VLLM_FUSED_BLOCK_SOFTMAX=true \
VLLM_WEIGHT_LOAD_FORCE_SYNC=1 \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_GRAPH_RESERVED_MEM=0.3 \
VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
VLLM_HPU_MULTI_MODEL_CONFIG=multi_models.yaml \
python -m vllm_gaudi.entrypoints.openai.multi_model_api_server \
    --port 8090 \
    --disable-log-stats \
    --trust-remote-code
	


# =========================
# BENCHMARK COMMAND - Model 1
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model meta-llama/Llama-Guard-4-12B \
    --dataset-name custom \
    --dataset-path /tmp/aegis-benchmark.jsonl \
    --base-url http://localhost:8090 \
    --num-prompts 40 \
    --max-concurrency 4 \
    --request-rate inf \
    --port 8090 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --temperature 0 \
    --skip-chat-template \
    --trust-remote-code

# =========================
# ONLINE SWAP COMMAND 
# =========================
curl -s http://localhost:8090/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ibm-granite/granite-guardian-3.3-8b",
    "drain_timeout": 60
  }' | jq

# =========================
# BENCHMARK COMMAND - Model 2
# =========================

PYTHONUNBUFFERED=1 vllm bench serve \
    --model ibm-granite/granite-guardian-3.3-8b \
    --dataset-name custom \
    --dataset-path /tmp/aegis-benchmark-granite-guardian-3.3-8b.jsonl \
    --base-url http://localhost:8090 \
    --num-prompts 40 \
    --max-concurrency 4 \
    --request-rate inf \
    --port 8090 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --save-result \
    --save-detailed \
    --temperature 0 \
    --trust-remote-code
