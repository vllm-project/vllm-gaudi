# !/bin/bash

# Default tests, disable_prefix_cache, hpu graph 

# llama
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# APC tests
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 ENABLE_APC=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# Eager tests
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 ENFORCE_EAGER=1 \
pytest -v -s../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# hpu graph warmup 
VLLM_CONTIGUOUS_PA=False PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# TP=2
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=2 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# ======================== Model validate ========================== #
# qwen3 moe
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Qwen3-30B-A3B.yaml

# granite
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/granite-8b.yaml

# mistral
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Mistral-7B-Instruct-v0.3.yaml

# mixtral TP=2
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=2 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Mixtral-8x7B-Instruct-v0.1.yaml

# Qwen3-30B-A3B-FP8
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=0 VLLM_USE_V1=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Qwen3-30B-A3B-FP8.yaml

# llama4-moe TP=4 (text generation)
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=4 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Llama-4-Scout-17B-16E-Instruct.yaml

# llama4-moe fp8-compressed-tensor TP=4 (text generation)
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=4 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic.yaml

# deepseek-v2-lite-chat
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/DeepSeek-V2-Lite-chat.yaml

# # llama4-moe TP=4 (vision inference)
# ENFORCE_EAGER=True VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false TP_SIZE=4 \
# pytest -v -s ../tests/models/multimodal/generation/test_common.py --model_card_path ../tests/models/multimodal/generation/model_cards/Llama-4-Scout-17B-16E-Instruct-vision.yaml

# deepseek-R1
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 TP_SIZE=8 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/DeepSeek-R1.yaml

# llama fp8
KV_CACHE_DTYPE=fp8_inc \
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
QUANT_CONFIG=../tests/models/language/generation/inc_unit_scale_quant.json \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# llama fp8 without fp8 kv
VLLM_CONTIGUOUS_PA=False VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 \
QUANT_CONFIG=../tests/models/language/generation/inc_unit_scale_quant_without_fp8kv.json \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml