import itertools
import logging
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Tuple

from vllm_gaudi.extension.logger import logger as logger
from vllm_gaudi.extension.runtime import get_config


class ExponentialBucketingStrategy():

    def check_for_user_flags(self, phase):
        dim = ['bs', 'seq'] if phase == 'prompt' else ['bs', 'block']
        params = ['min', 'step', 'max']
        env_vars = [f'VLLM_{phase}_{dim}_BUCKET_{p}'.upper() for dim in dim for p in params]
        user_flags = []
        for e in env_vars:
            if getattr(get_config(), e) is not None:
                user_flags.append(e)
        if len(user_flags) > 0:
            logger().warning("*******************************************************")
            for flag in user_flags:
                logger().warning(
                    f"Using Exponential Strategy - Your configuration {flag}={getattr(get_config(), flag)} will be overwritten!"
                )
            logger().warning("*******************************************************")

    def get_prompt_cfgs(self, max_num_prefill_seqs, block_size, max_num_batched_tokens, max_model_len):
        self.check_for_user_flags('prompt')
        use_merged_prefill = get_config().merged_prefill

        # cfgs shape: [min, step, max, limit]
        prompt_bs_limit = math.ceil(math.log2(max_num_prefill_seqs)) + 1
        prompt_bs_bucket_cfg = [1, 2, max_num_prefill_seqs, prompt_bs_limit]
        max_prompt_seq_limit = math.ceil(math.log2(max_num_batched_tokens))
        prompt_query_bucket_cfg = [block_size, block_size, max_num_batched_tokens, max_prompt_seq_limit]
        max_ctx = max(1, math.ceil((max_model_len - prompt_query_bucket_cfg[0]) // block_size))
        max_prompt_ctx_limit = 2 if max_ctx == 1 else math.ceil(math.log2(max_ctx)) + 1
        prompt_ctx_bucket_cfg = [0, 1, max_ctx, max_prompt_ctx_limit]

        if use_merged_prefill:
            prev_prompt_bs_bucket_cfg = tuple(prompt_bs_bucket_cfg)
            prev_prompt_query_bucket_cfg = tuple(prompt_query_bucket_cfg)
            prev_prompt_ctx_bucket_cfg = tuple(prompt_ctx_bucket_cfg)

            prompt_bs_bucket_cfg = (1, 1, 1, 1)
            query_min, query_step, _, query_limit = prev_prompt_query_bucket_cfg
            prompt_query_bucket_cfg = (query_min, query_step * 4, max_num_batched_tokens, query_limit)
            prompt_ctx_bucket_cfg = (0, 4, max_ctx * max_num_prefill_seqs, max_prompt_ctx_limit)

            msg = ('Merged prefill is enabled!\n'
                   'Overriding prompt bucketing settings!\n'
                   f'prompt bs cfg: {prev_prompt_bs_bucket_cfg} -> {prompt_bs_bucket_cfg}\n'
                   f'prompt query cfg: {prev_prompt_query_bucket_cfg} -> {prompt_query_bucket_cfg}\n'
                   f'prompt ctx cfg: {prev_prompt_ctx_bucket_cfg} -> {prompt_ctx_bucket_cfg}\n')
            logger().info(msg)

        msg = ("Prompt bucket config (min, step, max_warmup, limit) "
               f"bs:{prompt_bs_bucket_cfg}, "
               f"query:{prompt_query_bucket_cfg}, "
               f"blocks:{prompt_ctx_bucket_cfg}")
        logger().info(msg)

        return prompt_bs_bucket_cfg, prompt_query_bucket_cfg, prompt_ctx_bucket_cfg

    def get_decode_cfgs(self, max_num_seqs, block_size, max_num_batched_tokens, max_model_len, max_blocks):
        self.check_for_user_flags('decode')
        prefix_caching = get_config().prefix_caching
        use_contiguous_pa = get_config().use_contiguous_pa

        # cfgs shape: [min, step, max, limit]
        decode_bs_limit = math.ceil(math.log2(max_num_seqs)) + 1
        decode_bs_bucket_cfg = [1, 2, max_num_seqs, decode_bs_limit]
        decode_query_bucket_cfg = [1, 1, 1, 1]
        max_decode_block_limit = math.ceil(math.log2(max_blocks)) + 1
        max_decode_blocks = max_blocks if use_contiguous_pa else \
                            min((max_model_len // block_size * max_num_seqs), max_blocks)
        decode_block_bucket_cfg = [1, max_num_seqs, max_decode_blocks, max_decode_block_limit]

        msg = ("Decode bucket config (min, step, max_warmup, limit) "
               f"bs:{decode_bs_bucket_cfg}, "
               f"block:{decode_block_bucket_cfg}")
        logger().info(msg)

        return decode_bs_bucket_cfg, decode_query_bucket_cfg, decode_block_bucket_cfg

    def get_range(self, cfg):
        range_for_cfg = warmup_range_with_limit(cfg)
        return sorted(range_for_cfg)


def warmup_range_with_limit(config: Tuple[int, int, int, int], long_context=False):
    """ 
    NOTE(kzawora): we'll use exponential spacing for buckets in which scaled 
    power will return bmin for first bucket iteration, and bmax for last 
    iteration, with elements between determined by the exponent, and base being 
    unchanged. Note that after padding to bstep, duplicates may occur, and
    then shall be removed.
    Example (bmin=128, bstep=128, bmax=2048, num_buckets=10):
    There are 16 possible buckets (2048/128), and we'll attempt to select 10 of 
    them with exponential spacing.
    base = (bmax/bmin) ** (1/(num_buckets-1)); (2048/128) ** (1/9) = 1.36079
    exponent = i
    power = base ** exponent
    scaled_power = b_min * power
    For i == 0 (first bucket), power is 1.36079 ** 0 = 1; 
        scaled_power is 1 * 128 = 128 (==bmin)
    For i == 9 (last bucket), power is 1.36079 ** 9 = 16; 
        scaled_power is 16 * 128 = 2048 (==bmax)
    So, computing for all buckets:
    scaled_powers_unpadded     = [bmin*base^0(==bmin), bmin*base^1, bmin*base^2,       ...,     bmin*base^9(==bmax)]
    scaled_powers_unpadded     = [128.00, 174.18, 237.02, 322.54, 438.91, 597.26, 812.75, 1105.98, 1505.01, 2048.00]
 
    We then remove duplicate buckets:
        scaled_powers_padded   = [   128,    256,    256,    384,    512,    640,    896,    1152,    1536,    2048]
                                               ^_______^ 
                                               duplicates
        buckets                = [   128,    256,            384,    512,    640,    896,    1152,    1536,    2048]
                                                      ^ 
                                         duplicate bucket removed
    """ # noqa: E501

    bmin, bstep, bmax, num_buckets = config
    add_zero_bucket = bmin == 0
    if add_zero_bucket:
        bmin = bstep
    linear_buckets = set(np.arange(bmin, bmax + 1, step=bstep))
    assert num_buckets > 0, "num_buckets must be a positive integer"
    if num_buckets == 1:
        return [bmax]
    buckets: Set[Tuple[int, int]] = set()

    if long_context:
        num_buckets_exp = math.floor(num_buckets / 2)
        num_buckets_linear = num_buckets - num_buckets_exp
        first_step = bmax / num_buckets  #or i.e. * 0.25
    else:
        num_buckets_exp = num_buckets
        first_step = bmax

    for i in range(num_buckets_exp):
        power_unpadded = bmin * np.float_power(first_step / bmin, (1. / float(num_buckets_exp - 1)) * i)
        if i == num_buckets - 1 and get_config().use_contiguous_pa:
            bucket = bmax
        else:
            bucket = math.ceil(power_unpadded / bstep) * bstep
        buckets.add(bucket)

    if long_context:
        #tmp_step = bmax / num_buckets
        tmp_step = (bmax - first_step) / num_buckets_linear
        for i in range(1, num_buckets_linear + 1):
            #for i in range(1, num_buckets+1):
            power_unpadded = first_step + i * tmp_step

            if i == num_buckets and get_config().use_contiguous_pa:
                bucket = bmax
            else:
                bucket = math.ceil(power_unpadded / bstep) * bstep
            '''if fill and bucket in buckets:
                available_buckets = linear_buckets.difference(buckets)
                if len(available_buckets) == 0:
                    break  # there are no more unique buckets, let's exit now
                new_bucket = min(available_buckets,
                             key=lambda x: abs(x - power_unpadded))
                if new_bucket not in buckets:
                    buckets.add(new_bucket)
            else:
                if bucket not in buckets:
                    buckets.add(bucket)
            '''
            if bucket not in buckets:
                buckets.add(bucket)
    if add_zero_bucket:
        buckets.add(0)
    return list(sorted(buckets))
