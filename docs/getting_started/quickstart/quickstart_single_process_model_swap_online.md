# Single-Process Model Swap (Online Quickstart)

This quickstart shows an end-to-end online flow for serving multiple small models sequentially in one Gaudi server process, without restarting the API server between model changes.

## When to Use

Use this mode when:

- You need to switch model A → model B without server restart.
- Your workload is sequential and each model fits the available device budget when loaded.

Do not use this mode as a replacement for multi-process or multi-node orchestration.

## Prerequisites

- vLLM and vLLM Gaudi plugin installed.
- Multi-model config file, for example:

```yaml
default_model: llama
models:
  llama:
    model: meta-llama/Llama-3.1-8B-Instruct
    tensor_parallel_size: 1
    max_model_len: 4096
    enable_auto_tool_choice: false
  qwen:
    model: Qwen/Qwen3-0.6B
    tensor_parallel_size: 1
    max_model_len: 4096
    enable_auto_tool_choice: true
    tool_call_parser: hermes
```

## Start Server

```bash
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_HPU_MULTI_MODEL_CONFIG=/path/to/multi_models.yaml

python -m vllm_gaudi.entrypoints.openai.multi_model_api_server \
  --host 0.0.0.0 \
  --port 8080
```

Notes:

- This entrypoint reads configured model aliases from `VLLM_HPU_MULTI_MODEL_CONFIG`.
- `/v1/models` lists every configured alias, but generation requests are handled by the currently active model only.
- `/v1/models/switch` is available only when `VLLM_SERVER_DEV_MODE=1`.
- `VLLM_ALLOW_INSECURE_SERIALIZATION=1` is currently required because the in-process reconfigure hook uses `cloudpickle` internally. Use this mode only in trusted/internal deployments.
- Frontend settings can now be set per model in the YAML config for `enable_auto_tool_choice`, `tool_call_parser`, and `chat_template`.
- Per-model `chat_template` values can be absolute paths or paths relative to the multi-model config file.
- Per-model `quant_config` path can be specified to modify `QUANT_CONFIG` env variable. 
- If a per-model setting is absent, the server falls back to the corresponding CLI value.

## Online Flow (Smoke Test)

- List available models:

```bash
curl -s http://localhost:8080/v1/models | jq
```

- Generate with default model:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "Explain Intel Gaudi in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq
```

The `model` field should match the active model. After a successful switch, use the new model alias in subsequent requests.

- Switch model in-process:

```bash
curl -s http://localhost:8080/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "drain_timeout": 60
  }' | jq
```

- Generate with switched model:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Explain Intel Gaudi in one sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }' | jq
```

## Rollback

To disable this mode, unset multi-model env flag and use standard serving:

```bash
unset VLLM_HPU_MULTI_MODEL_CONFIG
vllm serve <your-model>
```
