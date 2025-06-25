# !/bin/bash

VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Qwen3-30B-A3B.yaml