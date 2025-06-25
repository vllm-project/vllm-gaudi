# deploy

```

# install vllm, before upstream PR merged, need to use fork firstly.
VLLM_TARGET_DEVICE=hpu pip install git+https://github.com/vllm-project/vllm.git

# install plugin
git clone -b plugin_poc https://github.com/HabanaAI/vllm-hpu-extension.git
cd vllm-hpu-extension; pip install -e .

# install pytest dependencies
pip install lm_eval pytest pytest_asyncio
```

# supported model and feature

| model_name | worker | task | acc metrics | acc score |
|----------- | ------ | ---- | ----------- | --------- |
| Meta-Llama-3.1-8B-Instruct | v0 | gsm8k_cot_llama | exact_match,strict-match | 0.8066 |
| Meta-Llama-3.1-8B-Instruct | v1 | gsm8k_cot_llama | exact_match,strict-match | 0.8105 |
| Qwen3-30B-A3B | v0 | gsm8k | exact_match,strict-match | 0.9023 |
| Qwen3-30B-A3B | v1 | gsm8k | exact_match,strict-match | 0.9062 |

```
cd vllm-hpu/examples; bash run_tests.sh
```

# test

```
cd vllm-hpu/examples;

# v0
PT_HPU_LAZY_MODE=1 python test_plugin.py

# v1
PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false python test_plugin.py
```

expected see vllm hpu plugin registered
* For v0

```
INFO 06-05 01:12:17 [__init__.py:31] Available plugins for group vllm.platform_plugins:
INFO 06-05 01:12:17 [__init__.py:33] - hpu -> vllm_hpu:register
INFO 06-05 01:12:17 [__init__.py:36] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 06-05 01:12:17 [__init__.py:234] Platform plugin hpu is activated
WARNING 06-05 01:12:18 [_custom_ops.py:21] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
INFO 06-05 01:12:19 [__init__.py:31] Available plugins for group vllm.general_plugins:
INFO 06-05 01:12:19 [__init__.py:33] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
INFO 06-05 01:12:19 [__init__.py:33] - hpu_custom_ops -> vllm_hpu:register_ops
INFO 06-05 01:12:19 [__init__.py:36] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
......
```

* For v1

```
WARNING 06-07 01:38:28 [importing.py:29] Triton is not installed. Using dummy decorators. Install it via `pip install triton` to enable kernel compilation.
INFO 06-07 01:38:28 [__init__.py:39] Available plugins for group vllm.platform_plugins:
INFO 06-07 01:38:28 [__init__.py:41] - hpu -> vllm_hpu:register
INFO 06-07 01:38:28 [__init__.py:44] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 06-07 01:38:28 [__init__.py:235] Platform plugin hpu is activated
WARNING 06-07 01:38:29 [_custom_ops.py:22] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")

INFO 06-07 01:38:36 [config.py:822] This model supports multiple tasks: {'classify', 'reward', 'generate', 'embed', 'score'}. Defaulting to 'generate'.
WARNING 06-07 01:38:36 [arg_utils.py:1638] Detected VLLM_USE_V1=1 with hpu. Usage should be considered experimental. Please report any issues on Github.
INFO 06-07 01:38:36 [config.py:1967] Disabled the custom all-reduce kernel because it is not supported on current platform.
INFO 06-07 01:38:36 [config.py:2176] Chunked prefill is enabled with max_num_batched_tokens=8192.
=========compilation_config.custom_ops=['all']===========
INFO 06-07 01:38:36 [core.py:455] Waiting for init message from front-end.
=========compilation_config.custom_ops=['all']===========
```
