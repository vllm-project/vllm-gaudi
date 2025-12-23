vllm bench serve --backend openai-chat --model /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct-FP8/snapshots/4bf2c2f39c37c0fede78bede4056e1f18cdf8109/ \
--dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat \
--num-prompts 100 --base-url http://localhost:12346 --endpoint /v1/chat/completions \
--max-concurrency 100 --ready-check-timeout-sec 0 2>&1 |tee log_c.txt