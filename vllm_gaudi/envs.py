# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH: bool = True
    VLLM_HPU_USE_DELAYED_SAMPLING: bool = False
    VLLM_HPU_FORCE_CHANNEL_FP8: bool = True

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition
environment_variables: dict[str, Callable[[], Any]] = {
    # Contiguous cache fetching to avoid using costly gather operation on
    # Gaudi3. This is only applicable to HPU contiguous cache. If set to true,
    # contiguous cache fetch will be used.
    "VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH":
    lambda: os.environ.get("VLLM_CONTIGUOUS_PA", "true").lower() in ("1", "true"),

    # Use delayed sampling for HPU to reduce host cpu overhead
    # between each step.
    "VLLM_HPU_USE_DELAYED_SAMPLING":
    lambda: os.environ.get("VLLM_DELAYED_SAMPLING", "false").lower() in ("1", "true"),

    # Convert block fp8 to channel fp8 for HPU
    # If `QUANT_CONFIG` is set, this will be forced to false.
    "VLLM_HPU_FORCE_CHANNEL_FP8":
    lambda: os.environ.get("VLLM_HPU_FORCE_CHANNEL_FP8", "true").lower() in
    ("1", "true") and os.environ.get("QUANT_CONFIG", None) is None,
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
