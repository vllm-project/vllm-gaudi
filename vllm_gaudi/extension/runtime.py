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
HIDDEN_PARAMS = ['exponential_bucketing', 'linear_bucketing', 'flex_impl', 'fsdpa_impl', 'naive_impl']


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
    environment_values = filter_defined(detected, ENVIRONMENT_VALUES)
    feature_values = filter_defined(detected, FEATURE_VALUES)

    if len(experimental_flags) > 0 and not detected.VLLM_ENABLE_EXPERIMENTAL_FLAGS:
        asterisks = 48 * '*'
        header = f"{asterisks} Warning! {asterisks}"
        footer = '*' * len(header)
        logger().warning(header)
        logger().warning(
            f"Following environment variables are considered experimental: {', '.join(experimental_flags)}")
        logger().warning(
            "From v0.12.0 release using those flags without VLLM_ENABLE_EXPERIMENTAL_FLAGS will trigger a fatal error.")
        logger().warning(footer)

    dump('Environment', environment_values)
    dump('Features', feature_values)
    dump('User flags', user_flags)
    dump('Experimental flags', experimental_flags)

    return detected


def clear_config():
    global RUNTIME_CONFIG
    RUNTIME_CONFIG = None
