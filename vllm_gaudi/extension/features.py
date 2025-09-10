###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm_gaudi.extension.config import Not, Hardware, VersionRange, ModelType, Kernel, Any, All, Value, ValueFromList, Env, Enabled, Disabled, Engine, boolean, to_dict, split_values_and_flags, list_of
from vllm_gaudi.extension.kernels import fsdpa, block_softmax_adjustment
from vllm_gaudi.extension.validation import for_all, choice


def get_user_flags():
    flags = [
        Env('VLLM_USE_V1', boolean),
        Env('VLLM_ENABLE_EXPERIMENTAL_FLAGS', boolean),
        Env('VLLM_EXPONENTIAL_BUCKETING', boolean),
        Env('VLLM_PROMPT_BS_BUCKET_MIN', int),
        Env('VLLM_PROMPT_BS_BUCKET_STEP', int),
        Env('VLLM_PROMPT_BS_BUCKET_MAX', int),
        Env('VLLM_PROMPT_BS_BUCKET_LIMIT', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_MIN', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_STEP', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_MAX', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_LIMIT', int),
        Env('VLLM_DECODE_BS_BUCKET_MIN', int),
        Env('VLLM_DECODE_BS_BUCKET_STEP', int),
        Env('VLLM_DECODE_BS_BUCKET_MAX', int),
        Env('VLLM_DECODE_BS_BUCKET_LIMIT', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_MIN', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_STEP', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_MAX', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_LIMIT', int),

        # Non-vllm flags that are also important to print
        Env('EXPERIMENTAL_WEIGHT_SHARING', str),
        Env('PT_HPU_WEIGHT_SHARING', str),
    ]
    return to_dict(flags)


def get_experimental_flags():
    flags = [
        Env('VLLM_PT_PROFILE', str),
        Env('VLLM_PROFILE_PROMPT', str),
        Env('VLLM_PROFILE_DECODE', str),
        Env('VLLM_PROFILE_STEPS', list_of(int)),
        Env('VLLM_DEFRAG_THRESHOLD', int),
        Env('VLLM_DEFRAG_WITH_GRAPHS', boolean),
        Env('VLLM_DEBUG', list_of(str), check=for_all(choice('steps', 'defrag', 'fwd'))),
    ]
    return to_dict(flags)


def get_features():
    supported_attn_impls = ['flex_impl', 'fsdpa_impl', 'naive_impl']
    bucketing_strategies = ['exponential_bucketing', 'linear_bucketing']
    features = [
        Value('fp32_alibi_biases', True, env_var='VLLM_ALIBI_USE_FLOAT32_BIASES'),
        Value('fp32_softmax', ModelType('qwen2')),
        Value(
            'fused_block_softmax_adjustment',
            All(VersionRange(">=1.22.0.494"), Hardware('gaudi3'), Kernel(block_softmax_adjustment),
                Not(ModelType('qwen2')))),
        Value('fused_block_softmax', False),
        Value('flex_impl', False, env_var='VLLM_PROMPT_USE_FLEX_ATTENTION'),
        Value('fsdpa_impl', All(Kernel(fsdpa), Not(ModelType('mllama'))), env_var='VLLM_PROMPT_USE_FUSEDSDPA'),
        Value('naive_impl', True),
        ValueFromList('prompt_attn_impl', supported_attn_impls),
        Value('skip_warmup', False),
        Value('merged_prefill', Enabled('unified_attn')),
        Value('use_contiguous_pa',
              Any(Disabled('prefix_caching'), Enabled('unified_attn')),
              env_var='VLLM_CONTIGUOUS_PA'),
        Value('use_delayed_sampling', Engine('v0'), env_var='VLLM_DELAYED_SAMPLING'),
        Value('use_bucketing', True, env_var='VLLM_ENABLE_BUCKETING'),
        Value('exponential_bucketing', True),
        Value('linear_bucketing', True),
        ValueFromList('bucketing_strategy', bucketing_strategies),
        Value('defrag', Enabled('unified_attn')),
        Value('regional_compilation', True, env_var='VLLM_T_COMPILE_REGIONAL_COMPILATION', env_var_type=boolean),
        Value('dynamic_shapes_compilation', True, env_var='VLLM_T_COMPILE_DYNAMIC_SHAPES', env_var_type=boolean),
        Value('fullgraph_compilation', False, env_var='VLLM_T_COMPILE_FULLGRAPH', env_var_type=boolean),
        Value('unified_attn', False),
    ]
    return split_values_and_flags(features)
