#!/bin/bash
# Run lm-eval-harness as a client against a running vLLM OpenAI server,
# evaluating the configured task (GPQA diamond CoT zero-shot by default).
# Config values come from run-tests.sh (CFG_* env, sourced from the eval yaml);
# each has a fallback so the script still runs standalone. Results are written
# to $RESULTS_DIR for check_accuracy.py.

VMODEL=${CFG_MODEL:-${VMODEL:-/software/data/pytorch/huggingface/hub/models--moonshotai--Kimi-K2.6/snapshots/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/}}
PORT=${LM_EVAL_PORT:-12345}
TASK=${CFG_TASK:-gpqa_diamond_cot_zeroshot}
LIMIT=${CFG_LIMIT:-${LM_EVAL_LIMIT:-16}}
NUM_CONCURRENT=${CFG_NUM_CONCURRENT:-${LM_EVAL_NUM_CONCURRENT:-8}}
GEN_KWARGS=${CFG_GEN_KWARGS:-temperature=0.0,max_gen_toks=8192}
RESULTS_DIR=${RESULTS_DIR:-./results/gpqa_greedy_nothink}

export OPENAI_API_BASE=http://localhost:${PORT}/v1
export OPENAI_API_KEY=EMPTY
export MODEL="$VMODEL"

rm -rf "$RESULTS_DIR"
lm_eval \
  --model local-chat-completions \
  --model_args model=$MODEL,\
base_url=$OPENAI_API_BASE/chat/completions,\
num_concurrent=$NUM_CONCURRENT,\
max_retries=3,\
tokenized_requests=False,\
timeout=3600 \
  --tasks "$TASK" \
  --gen_kwargs "$GEN_KWARGS" \
  --apply_chat_template \
  --batch_size auto \
  --output_path "$RESULTS_DIR" \
  --log_samples --trust_remote_code \
  --limit "$LIMIT"
