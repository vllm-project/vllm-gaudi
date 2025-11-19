import itertools
import operator
import os
import math
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_gaudi.extension.logger import logger as logger
from vllm_gaudi.extension.runtime import get_config


class LinearBucketingStrategy:

    def get_prompt_cfgs(self, max_num_prefill_seqs, block_size, max_num_batched_tokens, max_model_len):
        use_merged_prefill = get_config().merged_prefill
        prefix_caching = get_config().prefix_caching

        prompt_bs_bucket_cfg = read_bucket_settings('prompt', 'bs', min=1, step=1, max=max_num_prefill_seqs)
        prompt_query_bucket_cfg = read_bucket_settings('prompt',
                                                       'query',
                                                       min=block_size,
                                                       step=block_size,
                                                       max=max_num_batched_tokens)
        max_ctx = math.ceil((max_model_len - prompt_query_bucket_cfg[0]) // block_size)
        prompt_ctx_bucket_cfg = read_bucket_settings('prompt', 'ctx', min=0, step=1, max=max_ctx)

        if use_merged_prefill:
            prev_prompt_bs_bucket_cfg = tuple(prompt_bs_bucket_cfg)
            prev_prompt_query_bucket_cfg = tuple(prompt_query_bucket_cfg)
            prev_prompt_ctx_bucket_cfg = tuple(prompt_ctx_bucket_cfg)

            prompt_bs_bucket_cfg = (1, 1, 1)
            query_min, query_step, _ = prev_prompt_query_bucket_cfg
            prompt_query_bucket_cfg = (query_min, query_step * 4, max_num_batched_tokens)
            prompt_ctx_bucket_cfg = read_bucket_settings('prompt',
                                                         'ctx',
                                                         min=0,
                                                         step=4,
                                                         max=max_ctx * max_num_prefill_seqs)

            msg = ('Merged prefill is enabled!\n'
                   'Overriding prompt bucketing settings!\n'
                   f'prompt bs cfg: {prev_prompt_bs_bucket_cfg} -> {prompt_bs_bucket_cfg}\n'
                   f'prompt query cfg: {prev_prompt_query_bucket_cfg} -> {prompt_query_bucket_cfg}\n'
                   f'prompt ctx cfg: {prev_prompt_ctx_bucket_cfg} -> {prompt_ctx_bucket_cfg}\n')
            logger().info(msg)

        msg = ("Prompt bucket config (min, step, max_warmup) "
               f"bs:{prompt_bs_bucket_cfg}, "
               f"query:{prompt_query_bucket_cfg}, "
               f"blocks:{prompt_ctx_bucket_cfg}")
        logger().info(msg)

        return prompt_bs_bucket_cfg, prompt_query_bucket_cfg, prompt_ctx_bucket_cfg

    def get_decode_cfgs(self, max_num_seqs, block_size, max_num_batched_tokens, max_model_len, max_blocks):
        prefix_caching = get_config().prefix_caching
        contiguous_pa = get_config().use_contiguous_pa

        decode_bs_bucket_cfg = read_bucket_settings('decode', 'bs', min=1, step=32, max=max_num_seqs)
        decode_query_bucket_cfg = [1, 1, 1]
        max_decode_blocks = max(math.ceil(max_model_len * max_num_seqs // block_size), block_size)
        if contiguous_pa:
            max_decode_blocks = max_blocks
        decode_block_bucket_cfg = read_bucket_settings('decode', 'block', min=1, step=block_size, max=max_decode_blocks)
        if decode_block_bucket_cfg[2] > max_blocks:
            logger().info(
                f'VLLM_DECODE_BLOCK_BUCKET_MAX={decode_block_bucket_cfg[2]} is higher than max_blocks={max_blocks}. Your configuration VLLM_DECODE_BLOCK_BUCKET_MAX={decode_block_bucket_cfg[2]} will be overwritten to VLLM_DECODE_BLOCK_BUCKET_MAX={max_blocks}'
            )
            decode_block_bucket_cfg[2] = max_blocks
            if decode_block_bucket_cfg[0] > max_blocks:
                decode_block_bucket_min = max(1, max_blocks - decode_block_bucket_cfg[1])
                logger().info(
                    f'VLLM_DECODE_BLOCK_BUCKET_MIN={decode_block_bucket_cfg[0]} is higher than max_blocks={max_blocks}. Your configuration VLLM_DECODE_BLOCK_BUCKET_MIN={decode_block_bucket_cfg[0]} will be overwritten to VLLM_DECODE_BLOCK_BUCKET_MIN={decode_block_bucket_min}'
                )
                decode_block_bucket_cfg[0] = decode_block_bucket_min

        msg = ("Decode bucket config (min, step, max_warmup) "
               f"bs:{decode_bs_bucket_cfg}, "
               f"blocks:{decode_block_bucket_cfg}")
        logger().info(msg)

        return decode_bs_bucket_cfg, decode_query_bucket_cfg, decode_block_bucket_cfg

    def get_range(self, cfg):
        range_for_cfg = warmup_range(cfg)
        return sorted(range_for_cfg)


def read_bucket_settings(phase: str, dim: str, **defaults):
    """Read bucketing configuration from env variables.

    phase is either 'prompt' or 'decode'
    dim is either 'bs', 'query' or 'block'
    param is either 'min', 'step' or 'max'
    example env variable: VLLM_DECODE_BS_BUCKET_STEP=128
    """
    params = ['min', 'step', 'max']
    env_vars = [f'VLLM_{phase}_{dim}_BUCKET_{p}'.upper() for p in params]
    default_values = [defaults[p] for p in params]
    values = []

    for p, e, d in zip(params, env_vars, default_values):
        val = os.environ.get(e)

        if val is None and dim == 'query':
            # Check if fallback 'seq' flag is set
            fallback_env = f'VLLM_{phase}_SEQ_BUCKET_{p}'.upper()
            fallback_val = os.environ.get(fallback_env)

            if fallback_val is not None:
                val = fallback_val
                logger().warning(f"{e} not set, using {fallback_env} value ({fallback_val}) instead. "
                                 "This fallback behavior is deprecated and will be removed in v0.12.0.")
        resolved_val = int(val) if val is not None else d
        logger().info(f'{e}={resolved_val} (default:{d})')
        values.append(resolved_val)

    return values


def warmup_range(config: Tuple[int, int, int]):
    """Generate a warmup range.

    Start from bmin and multiply by 2 until you reach bstep.
    Then, increase the values in the range by the value of bstep until you 
    reach bmax.

    Example:
    bmin = 2, bstep = 32, bmax = 64
    => ramp_up = (2, 4, 8, 16)
    => stable = (32, 64)
    => return ramp_up + stable => (2, 4, 8, 16, 32, 64)
    """
    bmin, bstep, bmax = config
    add_zero_bucket = bmin == 0
    assert bmin <= bmax, ("Min. batch size cannot be greater than max. "
                          "batch size. If you want to skip warmup, "
                          "set VLLM_SKIP_WARMUP=true")
    if add_zero_bucket:
        if bmin == 0 and bmax == 0:
            return [0]
        bmin = bstep
    base = itertools.repeat(2)
    ramp_up_acc = itertools.accumulate(base, func=operator.mul, initial=bmin)
    ramp_up_tw = itertools.takewhile(lambda x: x < bstep and x <= bmax, \
        ramp_up_acc)
    stable = range(bstep, bmax + 1, bstep)
    buckets = list(ramp_up_tw) + list(stable)
    buckets = [b for b in buckets if b >= bmin]
    if add_zero_bucket:
        buckets.append(0)
    return list(buckets)
