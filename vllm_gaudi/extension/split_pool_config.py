# SPDX-License-Identifier: Apache-2.0
"""Configuration for split KV cache pools.

Enables co-execution of short-context and long-context requests in a single
vLLM-Gaudi instance by defining two logical pools with different capacity
parameters.  The scheduler and model runner use pool membership to enforce
per-pool batch-size limits and to generate pool-specific HPU graph buckets.

Configuration is driven by the ``VLLM_SPLIT_POOLS_CONFIG`` environment
variable which accepts a JSON string, e.g.::

    VLLM_SPLIT_POOLS_CONFIG='{
        "short": {"max_model_len": 8192, "max_num_seqs": 32, "memory_fraction": 0.40},
        "long":  {"max_model_len": 131072, "max_num_seqs": 8, "memory_fraction": 0.55}
    }'

When the variable is unset or empty the feature is disabled and a single
unified pool is used (the default behaviour).
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


@dataclass
class PoolConfig:
    """Parameters for one logical KV cache pool."""
    name: str
    max_model_len: int
    max_num_seqs: int
    memory_fraction: float = 0.5
    # Bucketing overrides — populated at runtime
    prompt_bs_max: int = 0
    decode_bs_max: int = 0

    def __post_init__(self):
        if self.max_model_len <= 0:
            raise ValueError(f"Pool '{self.name}': max_model_len must be > 0")
        if self.max_num_seqs <= 0:
            raise ValueError(f"Pool '{self.name}': max_num_seqs must be > 0")
        if not (0.0 < self.memory_fraction < 1.0):
            raise ValueError(f"Pool '{self.name}': memory_fraction must be in (0, 1)")
        if self.prompt_bs_max == 0:
            self.prompt_bs_max = self.max_num_seqs
        if self.decode_bs_max == 0:
            self.decode_bs_max = self.max_num_seqs


@dataclass
class SplitPoolConfig:
    """Top-level configuration for split KV cache pools.

    Attributes:
        enabled: Whether split pools are active.
        short_pool: Configuration for the short-context pool.
        long_pool: Configuration for the long-context pool.
        threshold: Token count boundary between short and long.
                   Requests with ``num_prompt_tokens > threshold`` are routed
                   to the long pool.  Defaults to ``short_pool.max_model_len``.
    """
    enabled: bool = False
    short_pool: Optional[PoolConfig] = None
    long_pool: Optional[PoolConfig] = None
    threshold: int = 0

    def classify_request(self, num_prompt_tokens: int) -> str:
        """Return ``"short"`` or ``"long"`` based on prompt token count."""
        if not self.enabled:
            return "short"
        return "long" if num_prompt_tokens > self.threshold else "short"

    def get_pool(self, pool_name: str) -> Optional[PoolConfig]:
        if pool_name == "short":
            return self.short_pool
        elif pool_name == "long":
            return self.long_pool
        return None


def _parse_pool_dict(name: str, d: dict) -> PoolConfig:
    return PoolConfig(
        name=name,
        max_model_len=int(d["max_model_len"]),
        max_num_seqs=int(d["max_num_seqs"]),
        memory_fraction=float(d.get("memory_fraction", 0.5)),
        prompt_bs_max=int(d.get("prompt_bs_max", 0)),
        decode_bs_max=int(d.get("decode_bs_max", 0)),
    )


_CACHED_CONFIG: Optional[SplitPoolConfig] = None


def get_split_pool_config() -> SplitPoolConfig:
    """Parse ``VLLM_SPLIT_POOLS_CONFIG`` and return a ``SplitPoolConfig``.

    The result is cached after the first call.
    """
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG

    raw = os.environ.get("VLLM_SPLIT_POOLS_CONFIG", "").strip()
    if not raw:
        _CACHED_CONFIG = SplitPoolConfig(enabled=False)
        return _CACHED_CONFIG

    try:
        cfg = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"VLLM_SPLIT_POOLS_CONFIG is not valid JSON: {e}") from e

    if "short" not in cfg or "long" not in cfg:
        raise ValueError("VLLM_SPLIT_POOLS_CONFIG must contain 'short' and 'long' keys")

    short_pool = _parse_pool_dict("short", cfg["short"])
    long_pool = _parse_pool_dict("long", cfg["long"])

    total_frac = short_pool.memory_fraction + long_pool.memory_fraction
    if total_frac > 1.0:
        raise ValueError(
            f"Sum of memory_fraction ({short_pool.memory_fraction} + "
            f"{long_pool.memory_fraction} = {total_frac}) exceeds 1.0")

    threshold = int(cfg.get("threshold", short_pool.max_model_len))

    split_cfg = SplitPoolConfig(
        enabled=True,
        short_pool=short_pool,
        long_pool=long_pool,
        threshold=threshold,
    )

    logger().info(
        "Split KV cache pools enabled: short(max_len=%d, max_seqs=%d, mem=%.0f%%), "
        "long(max_len=%d, max_seqs=%d, mem=%.0f%%), threshold=%d",
        short_pool.max_model_len, short_pool.max_num_seqs,
        short_pool.memory_fraction * 100,
        long_pool.max_model_len, long_pool.max_num_seqs,
        long_pool.memory_fraction * 100,
        threshold,
    )

    _CACHED_CONFIG = split_cfg
    return _CACHED_CONFIG


def reset_split_pool_config():
    """Reset cached config (for testing)."""
    global _CACHED_CONFIG
    _CACHED_CONFIG = None
