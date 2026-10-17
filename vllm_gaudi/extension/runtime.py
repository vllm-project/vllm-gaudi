###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm_gaudi.extension.environment import get_environment
from vllm_gaudi.extension.features import get_features, get_user_flags, get_experimental_flags
from vllm_gaudi.extension.config import Config
from vllm_gaudi.extension.logger import logger

RUNTIME_CONFIG = None
USER_FLAGS = None
EXPERIMENTAL_FLAGS = None
ENVIRONMENT_VALUES = None
FEATURE_VALUES = None
HIDDEN_PARAMS = ['flex_impl', 'fsdpa_impl', 'naive_impl']


def eager_deferred_sampling_warning(max_model_len: int, free_device_bytes: float, bytes_per_token: int = 16384):
    """Build the boot-time advisory for issue #1846's deferred-sampling ask,
    or return None when this configuration cannot hit the class.

    Under the eager PT bridge with the default 'exp' bucketing tree, the
    first sampling pass after a near-cap prefill triggers one deferred
    device allocation whose size tracks the request's block-row length:
    ``16,384 B x blockpad128(prompt + generation - 64)`` (issue #1846;
    measured byte-exactly six times on the reporting deployment, a
    Qwen3.8-class GDN hybrid -- the per-token width is model-geometry
    dependent). The worst case runs at the cap, where prompt + generation
    cannot exceed ``max_model_len``.
    """
    blockpad = max_model_len - 64
    if blockpad <= 0:
        return None
    blockpad = (blockpad + 127) // 128 * 128
    ask_bytes = bytes_per_token * blockpad
    if ask_bytes <= free_device_bytes:
        return None
    return (
        f'Predicted deterministic OOM on this configuration '
        f'(https://github.com/vllm-project/vllm-gaudi/issues/1846): under the eager PT bridge '
        f'with the default \'exp\' bucketing tree, a near-cap request (prompt + generation close '
        f'to max_model_len={max_model_len}) triggers one deferred device allocation of about '
        f'{ask_bytes / 2**30:.1f} GiB ({ask_bytes} B on the measured model class) at its first '
        f'sampling pass, and free device memory is {free_device_bytes / 2**30:.1f} GiB -- such a '
        f'request aborts all workers. The allocation is authored below the Python layer (deferred '
        f'pipeline materialization) and its per-token width is model-geometry dependent, so verify '
        f'against your own model. The measured remedy for long-context serving is the padding-aware '
        f'bucketing tree (VLLM_BUCKETING_STRATEGY=pad with VLLM_HPU_FSDPA_SLICE_ENABLED=0, '
        f'https://github.com/vllm-project/vllm-gaudi/issues/1854); otherwise cap --max-model-len.')


def filter_defined(config, keys):
    return {k: v for k, v in config.get_all(keys).items() if v is not None}


def dump(prefix, values):
    if values:
        padding = ' ' * 4
        logger().info(f'{prefix}:')
        for key, value in values.items():
            if key in HIDDEN_PARAMS:
                continue
            logger().info(f'{padding}{key}: {value}')


def get_config(**overrides):

    global RUNTIME_CONFIG, USER_FLAGS, EXPERIMENTAL_FLAGS, ENVIRONMENT_VALUES, FEATURE_VALUES, HIDDEN_PARAMS

    if RUNTIME_CONFIG:
        assert len(overrides) == 0, 'Overrides cannot be applied when config has been already created!'
        return RUNTIME_CONFIG

    user_flags = get_user_flags()
    experimental_flags = get_experimental_flags()
    environment_values, environment_flags = get_environment()
    feature_values, feature_flags = get_features()

    experimental_flags = experimental_flags | environment_flags | feature_flags

    detected = Config(user_flags | experimental_flags | environment_values | feature_values | overrides)

    RUNTIME_CONFIG = detected
    USER_FLAGS = list(user_flags.keys())
    EXPERIMENTAL_FLAGS = list(experimental_flags.keys())
    ENVIRONMENT_VALUES = list(environment_values.keys())
    FEATURE_VALUES = list(feature_values.keys())
    return RUNTIME_CONFIG


def finalize_config():
    detected = get_config()
    detected.finalize()

    user_flags = filter_defined(detected, USER_FLAGS)
    experimental_flags = filter_defined(detected, EXPERIMENTAL_FLAGS)
    experimental_flags = {k: v for k, v in user_flags.items() if k not in user_flags}
    environment_values = filter_defined(detected, ENVIRONMENT_VALUES)
    feature_values = filter_defined(detected, FEATURE_VALUES)

    if len(experimental_flags) > 0 and not detected.VLLM_DEVELOPER_MODE:
        asterisks = 48 * '*'
        header = f"{asterisks} Warning! {asterisks}"
        footer = '*' * len(header)
        logger().warning(header)
        logger().warning(
            f"Following environment variables are considered experimental: {', '.join(experimental_flags)}")
        logger().warning(
            "From v0.12.0 release using those flags without VLLM_DEVELOPER_MODE will trigger a fatal error.")
        logger().warning(footer)

    dump('Environment', environment_values)
    dump('Features', feature_values)
    dump('User flags', user_flags)
    dump('Experimental flags', experimental_flags)

    return detected


def clear_config():
    global RUNTIME_CONFIG
    RUNTIME_CONFIG = None
