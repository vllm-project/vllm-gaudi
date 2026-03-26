# Architecture Overview

This document summarizes the vLLM Gaudi additions used to support single-process model swap. The baseline V1 architecture is described in the upstream vLLM architecture documentation; only the Gaudi-specific delta is covered here.

## Entrypoints

vLLM provides multiple entrypoints for interacting with the system. For online inference, the model-swap feature uses a dedicated OpenAI-compatible Gaudi server entrypoint.

### OpenAI-Compatible Gaudi API Server

The server can be launched directly via:

```bash
export VLLM_SERVER_DEV_MODE=1 
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_HPU_MULTI_MODEL_CONFIG=/path/to/multi_models.yaml
python -m vllm_gaudi.entrypoints.openai.multi_model_api_server
```

Implementation lives in [vllm_gaudi/entrypoints/openai/multi_model_api_server.py](../../vllm_gaudi/entrypoints/openai/multi_model_api_server.py).

## New / Modified Components

| Component | Type | Role in Model Swap |
|---|---|---|
| `vllm_gaudi.v1.engine.MultiModelAsyncLLM` | New manager wrapper | Owns multi-model configs, serializes swap requests, drains in-flight requests, and triggers in-process reconfigure |
| `MultiModelEngineClient` | Engine client adapter | Exposes the wrapped `AsyncLLM` through the standard server-facing `EngineClient` interface |
| `MultiModelServingModels` | OpenAI model registry adapter | Lists all configured models in `/v1/models`, while keeping request validation aligned with the currently active model |
| `install_engine_core_patch()` | Runtime patch installer | Injects `gaudi_reconfigure_engine()` into V1 `EngineCore` when `MultiModelAsyncLLM` is constructed |
| `EngineCore.gaudi_reconfigure_engine()` | Added utility method (patched) | Performs in-place runtime rebuild: pause/sleep, worker reload, KV cache re-init, scheduler/state reconstruction, resume |
| `HPUWorker.load_model()` | Extended worker load path | Reloads model runner/model with new config |
| `HPUWorker._rebuild_kv_cache_config_for_current_model(...)` | New helper | Rebuilds KV cache layer mappings from current model spec to prevent stale block-table/layer mapping state |
| `MultiModelAsyncLLM._refresh_engine_frontend_config()` | Frontend refresh step | Rebuilds frontend-side renderer and processors so request parsing/tokenization matches the newly active model |

## Control Plane Delta: Switch Flow

### Caller side (`MultiModelAsyncLLM.switch_model`)

1. Acquire `_switching_lock` (single swap at a time).
2. Validate target model and skip no-op switches.
3. Drain pending requests (`wait_for_requests_to_drain`).
4. Serialize target `VllmConfig` with `cloudpickle`.
5. Invoke EngineCore utility: `call_utility_async("gaudi_reconfigure_engine", serialized_config)`.
6. Refresh frontend-side `AsyncLLM` state (`renderer`, I/O processor, input processor, output processor).
7. Update local model sleep-state bookkeeping and active model pointer.

### EngineCore side (`gaudi_reconfigure_engine`)

1. Deserialize new config.
2. Pause scheduler with cache reset (`pause_scheduler(mode="abort", clear_cache=True)`).
3. Sleep executor at level 1 to release device memory pressure.
4. Unload current worker model via collective RPC (`unload_model`).
5. Broadcast worker reload via collective RPC (`load_model`).
6. Recompute and initialize KV cache (`_initialize_kv_caches`, `initialize_cache`).
7. Rebuild scheduler-dependent runtime objects:
   - `StructuredOutputManager`
   - scheduler instance
   - KV connector handshake metadata
   - multimodal receiver cache
   - request block hasher and batch queue helpers
8. Reset executor sleep bookkeeping and resume scheduler.

## State Rebuild Delta

The model swap path rebuilds runtime state that is model-shape or scheduler-policy dependent. This avoids carrying stale state across model boundaries (for example incompatible layer bindings or stale block-table assumptions).

Rebuilt state includes:

- KV cache configuration and block counts
- scheduler instance and block sizing
- structured output manager
- multimodal receiver cache
- request block hashing setup
- queueing/execution helper state (`batch_queue`, `step_fn`, abort queue)
- frontend-side renderer and request processors used by the OpenAI server

## API Behavior Notes

- `/v1/models` returns all configured model aliases from the multi-model YAML file.
- Inference requests are still served by the currently active model only.
- `/v1/models/switch` is exposed only when `VLLM_SERVER_DEV_MODE=1`.
