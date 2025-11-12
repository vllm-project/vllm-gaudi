# Plugin System

The vLLM Hardware Plugin for Intel® Gaudi® integrates Intel® Gaudi® AI accelerators with vLLM using the standard plugin architecture. This document explains how the plugin system is implemented.

## Plugin Implementation

The vLLM Hardware Plugin for Intel® Gaudi® system extends the vLLM functionality without modifying the core codebase. It uses Python's standard entry points mechanism for plugin discovery and registration. For general information about the plugin system, refer to the [vLLM Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html) documentation.

The plugin consists of two complementary components that are registered through the Python’s entry points mechanism in the `setup.py` file: the platform plugin and the general plugin. This registration structure enables platform integration and custom operation support for Intel® Gaudi® hardware.

```python
entry_points={
    "vllm.platform_plugins": ["hpu = vllm_gaudi:register"],
    "vllm.general_plugins": ["hpu_custom_ops = vllm_gaudi:register_ops"],
}
```

### Platform Plugin

The platform plugin provides the core hardware integration for Intel® Gaudi® accelerators. It is responsible for initializing and managing devices, allocating and managing memory resources, applying platform-specific configurations, and selecting the appropriate attention backend. The plugin is defined in the Python’s entry points definition with the following elements:

- **Group** `vllm.platform_plugins`: Identifies the entry point as a platform plugin for hardware integration, allowing vLLM to automatically detect it.

- **Name** `hpu`: Identifies Intel® Gaudi® hardware for vLLM.

- **Value** `vllm_gaudi:register`: Points to the `register` function from the `vllm_gaudi` package. The function configures and returns the platform class.

The `register()` function defined in `vllm_gaudi/__init__.py` performs platform initialization to configure torch compilation settings for Intel® Gaudi® and returns the fully qualified class name `"vllm_gaudi.platform.HpuPlatform"`:

```python
from vllm_gaudi.platform import HpuPlatform

def register():
    """Register the HPU platform"""
    HpuPlatform.set_torch_compile()
    return "vllm_gaudi.platform.HpuPlatform"
```

The `set_torch_compile()` method, defined in `vllm_gaudi/platform.py`, configures PyTorch compilation behavior for Intel® Gaudi®:

```python
@classmethod
def set_torch_compile(cls) -> None:
    # Disable weight sharing for HPU
    os.environ['PT_HPU_WEIGHT_SHARING'] = '0'
    
    is_lazy = htorch.utils.internal.is_lazy()
    if is_lazy:
        # Lazy backend does not support torch.compile
        torch._dynamo.config.disable = True
        # Enable lazy collectives for multi-HPU inference
        os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES'] = 'true'
    elif os.environ.get('RUNTIME_SCALE_PATCHING') is None:
        # Enable runtime scale patching for eager mode
        os.environ['RUNTIME_SCALE_PATCHING'] = '1'
```

This setup ensures:

- Proper configuration based on whether lazy or eager execution mode is active
- Multi-HPU inference support through lazy collectives
- Correct `torch.compile` behavior for the Intel® Gaudi® backend

### General Plugin

The general plugin registers custom operations for Intel® Gaudi® accelerators. It is responsible for implementing custom kernels, supporting quantization, and supplying optimized operators. The plugin is composed of the following elements:

- **Group** `vllm.general_plugins`: Identifies the entry point as a general plugin that provides additional functionalities beyond the platform integration.

- **Name** `hpu_custom_ops`: Identifies Intel® Gaudi® HPU hardware customization options.

- **Value** `vllm_gaudi:register_ops`: Points to the `register_ops` function that registers custom HPU operations.

The `register_ops()` function, defined in `vllm_gaudi/__init__.py`, registers all Intel® Gaudi®-specific custom operations.

```python
def register_ops():
    """Register custom operations for the HPU platform"""
    import vllm_gaudi.v1.sample.hpu_rejection_sampler  # noqa: F401
    import vllm_gaudi.distributed.kv_transfer.kv_connector.v1.hpu_nixl_connector  # noqa: F401
    import vllm_gaudi.ops.hpu_fused_moe  # noqa: F401
    import vllm_gaudi.ops.hpu_layernorm  # noqa: F401
    import vllm_gaudi.ops.hpu_lora  # noqa: F401
    import vllm_gaudi.ops.hpu_rotary_embedding  # noqa: F401
    import vllm_gaudi.ops.hpu_compressed_tensors  # noqa: F401
    import vllm_gaudi.ops.hpu_fp8  # noqa: F401
    import vllm_gaudi.ops.hpu_gptq  # noqa: F401
    import vllm_gaudi.ops.hpu_awq  # noqa: F401
    import vllm_gaudi.ops.hpu_multihead_attn  # noqa: F401
```

These custom operations are imported (not called) to register them with vLLM’s operation registry so they can be used across the inference pipeline.

## Plugin Discovery and Loading

When vLLM starts, it performs the following steps:

1. Scans for platform plugins to discover all `vllm.platform_plugins` entry points.
2. Executes each plugin's `register()` function.
3. Chooses the appropriate platform based on hardware and registration results.
4. Discovers and loads all `vllm.general_plugins` entry points.
5. Initializes the selected platform with all registered custom operations.

vLLM may spawn multiple processes, for example, when performing distributed inference with tensor parallelism. The plugin registration mechanism ensures that:

- Each process independently discovers and loads both platform and general plugins
- Registration functions are executed in each worker process
- Platform-specific configurations, such as `set_torch_compile()`, are applied in each process
- Custom operations are registered in each processes that requires them

## Verifying Plugin Installation

After installing vLLM Hardware Plugin for Intel® Gaudi®, verify that both plugins are correctly registered:

```python
from importlib.metadata import entry_points

# List all vLLM platform plugins
platform_plugins = entry_points(group='vllm.platform_plugins')
for plugin in platform_plugins:
    print(f"Platform Plugin - name: {plugin.name}, value: {plugin.value}")

# List all vLLM general plugins
general_plugins = entry_points(group='vllm.general_plugins')
for plugin in general_plugins:
    print(f"General Plugin - name: {plugin.name}, value: {plugin.value}")
```

Expected output:

```
Platform Plugin - name: hpu, value: vllm_gaudi:register
General Plugin - name: hpu_custom_ops, value: vllm_gaudi:register_ops
```

You can verify which platform vLLM has selected by running the following command:

```python
from vllm.platforms import current_platform
print(f"Current platform: {current_platform}")
```

If Intel® Gaudi® hardware is detected and the plugin is functioning correctly, the output should indicate the HPU platform.

## Reference

For more information, see:

- [vLLM Plugin System Overview](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [RFC: Hardware Pluggable](https://github.com/vllm-project/vllm/issues/18641)
- [RFC: Enhancing vLLM Plugin Architecture](https://github.com/vllm-project/vllm/issues/19161)
- [Python Entry Points Documentation](https://packaging.python.org/en/latest/specifications/entry-points/)
