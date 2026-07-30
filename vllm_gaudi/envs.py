# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH: bool = True
    VLLM_HPU_FORCE_CHANNEL_FP8: bool = True
    VLLM_HPU_HETERO_KV_LAYOUT: bool = False
    VLLM_HPU_MULTI_MODEL_CONFIG: Optional[str] = None
    VLLM_NIXL_ABORT_REQUEST_TIMEOUT: float = 300.0
    VLLM_NIXL_SIDE_CHANNEL_HOST: str = "localhost"
    VLLM_NIXL_SIDE_CHANNEL_PORT: int = 5600
    VLLM_HPU_NIXL_JOINT_KV: bool = False
    VLLM_HPU_NIXL_STAGING_SLOTS: int = 0

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition
environment_variables: dict[str, Callable[[], Any]] = {
    # Contiguous cache fetching to avoid using costly gather operation on
    # Gaudi3. This is only applicable to HPU contiguous cache. If set to true,
    # contiguous cache fetch will be used.
    "VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH":
    lambda: os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() in ("1", "true"),

    # Convert block fp8 to channel fp8 for HPU
    # If `QUANT_CONFIG` is set, this will be forced to false.
    "VLLM_HPU_FORCE_CHANNEL_FP8":
    lambda: os.environ.get("VLLM_HPU_FORCE_CHANNEL_FP8", "true").lower() in
    ("1", "true") and os.environ.get("QUANT_CONFIG", None) is None,

    # Enable prefill side kv_layout and block_size for heterogeneous run.
    "VLLM_HPU_HETERO_KV_LAYOUT":
    lambda: os.environ.get("VLLM_HPU_HETERO_KV_LAYOUT", "false").lower() in ("0", "false"),

    # Path to a YAML config file describing the multi-model setup
    # (model names, weights, tensor-parallel config, etc.).
    # When unset, multi-model mode is disabled.
    "VLLM_HPU_MULTI_MODEL_CONFIG":
    lambda: os.environ.get("VLLM_HPU_MULTI_MODEL_CONFIG", None),

    # Timeout in seconds for NIXL abort request handling
    "VLLM_NIXL_ABORT_REQUEST_TIMEOUT":
    lambda: float(os.environ.get("VLLM_NIXL_ABORT_REQUEST_TIMEOUT", "300")),

    # NIXL side channel host and port for KV transfer
    "VLLM_NIXL_SIDE_CHANNEL_HOST":
    lambda: os.environ.get("VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"),
    "VLLM_NIXL_SIDE_CHANNEL_PORT":
    lambda: int(os.environ.get("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600")),

    # Enable joint-KV staging so an HPU prefill instance can serve a GPU
    # (FLASH_ATTN, blocks-first) decode instance over NIXL. When the remote
    # decode registers ONE joint [K|V] region per layer (block_len covering
    # both K and V, V at block_len // 2), the HPU must advertise the same
    # region model. Leave false for HPU<->HPU disagg (separate K/V regions).
    "VLLM_HPU_NIXL_JOINT_KV":
    lambda: os.environ.get("VLLM_HPU_NIXL_JOINT_KV", "false").lower() in ("1", "true"),

    # Number of joint-KV staging slots for heterogeneous (HPU prefill ->
    # GPU decode) NIXL transfer. Each slot holds one on-save block for all
    # layers laid out blocks-first [2, n_kv_heads, block_size_on_save,
    # head_size] so the GPU can read K and V from one contiguous region
    # (V at block_len // 2). 0 (default) auto-sizes to the full concurrent
    # workload: max_num_seqs * ceil(max_model_len / block_size_on_save), which
    # guarantees every schedulable request gets a reservation. Registration
    # fails fast if that pool does not fit device memory. Only set a smaller
    # value if you are certain peak concurrent transfer demand stays under it.
    "VLLM_HPU_NIXL_STAGING_SLOTS":
    lambda: int(os.environ.get("VLLM_HPU_NIXL_STAGING_SLOTS", "0")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
