#!/bin/bash
# Launch the vLLM OpenAI-compatible API server for the GPQA online eval.
# Config values come from run-tests.sh (CFG_* env, sourced from the eval yaml);
# each has a fallback so the script still runs standalone. Server settings
# mirror the repo's known-good Kimi-K2.6 launch scripts (server_run.sh).

VMODEL=${CFG_MODEL:-${VMODEL:-/software/data/pytorch/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/}}
TP_SIZE=${CFG_TP_SIZE:-${LM_EVAL_TP_SIZE:-8}}
PORT=${LM_EVAL_PORT:-12345}
MAX_MODEL_LEN=${CFG_MAX_MODEL_LEN:-${LM_EVAL_MAX_MODEL_LEN:-131072}}

export VMODEL
# Set inline so CI (which has no surrounding shell) matches the environment the
# repo-root lm_eval_server.sh is launched with.
export HF_HOME=${HF_HOME:-/software/data/pytorch/huggingface}
export PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-0}
export VLLM_SKIP_WARMUP=${VLLM_SKIP_WARMUP:-true}
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

# Server args mirror the repo-root lm_eval_server.sh.
python3 -m vllm.entrypoints.openai.api_server \
  --model "$VMODEL" \
  --served-model-name "$VMODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  $EP_ARG \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90 \
  $TRC_ARG \
  --limit-mm-per-prompt '{"image": 8}' \
  --default-chat-template-kwargs "{\"thinking\": ${THINKING}}" \
  --port "$PORT"
