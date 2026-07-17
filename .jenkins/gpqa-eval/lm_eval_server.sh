#!/bin/bash
# Launch the vLLM OpenAI-compatible API server for the GPQA online eval.
# Config values come from run-tests.sh (CFG_* env, sourced from the eval yaml);
# each has a fallback so the script still runs standalone. Server settings
# mirror the repo's known-good Kimi-K2.6 launch scripts (server_run.sh).

VMODEL=${CFG_MODEL:-${VMODEL:-/software/data/pytorch/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/}}
TP_SIZE=${CFG_TP_SIZE:-${LM_EVAL_TP_SIZE:-8}}
PORT=${LM_EVAL_PORT:-12345}
MAX_MODEL_LEN=${CFG_MAX_MODEL_LEN:-${LM_EVAL_MAX_MODEL_LEN:-16384}}
DTYPE=${CFG_DTYPE:-bfloat16}

export VMODEL
export HF_HOME=${HF_HOME:-/software/data/pytorch/huggingface}
export PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-0}
export VLLM_SKIP_WARMUP=${VLLM_SKIP_WARMUP:-false}
export HPU_FUSED_MOE=${HPU_FUSED_MOE:-1}

# Optional args gated by config.
EP_ARG=""
if [[ "${CFG_ENABLE_EP:-1}" == "1" ]]; then
    EP_ARG="--enable-expert-parallel"
fi
TRC_ARG=""
if [[ "${CFG_TRUST_REMOTE_CODE:-1}" == "1" ]]; then
    TRC_ARG="--trust-remote-code"
fi
# thinking:false is applied via the chat template kwargs (server-only knob).
THINKING="false"
if [[ "${CFG_THINKING:-0}" == "1" ]]; then
    THINKING="true"
fi

python3 -m vllm.entrypoints.openai.api_server \
  --model "$VMODEL" \
  --served-model-name "$VMODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  $EP_ARG \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.75 \
  $TRC_ARG \
  --default-chat-template-kwargs "{\"thinking\": ${THINKING}}" \
  --port "$PORT"
