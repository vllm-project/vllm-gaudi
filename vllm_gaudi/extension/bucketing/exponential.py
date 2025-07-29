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
        params = ['min', 'step', 'max', 'limit']
        env_vars = [f'VLLM_{phase}_{dim}_BUCKET_{p}'.upper() for dim in dim for p in params]
        user_flags = []
        for e in env_vars:
            if getattr(get_config(), e) is not None:
                user_flags.append(e)
        if len(user_flags) > 0:
            logger().warning("*******************************************************")
            for flag in user_flags:
                logger().warning(f"Using Exponential Strategy - Your configuration {flag}={getattr(get_config(), flag)} will be overwritten!")
            logger().warning("*******************************************************")
        

    def get_prompt_buckets(self, max_num_prefill_seqs, block_size, 
                           max_num_batched_tokens, max_model_len):
        self.check_for_user_flags('prompt')
        use_merged_prefill = get_config().merged_prefill
        prefix_caching = get_config().prefix_caching
        max_prompt_seq = max_model_len

        # cfgs shape: [min, step, max, limit]
        prompt_bs_limit = math.ceil(math.log2(max_num_prefill_seqs)) + 1
        prompt_bs_bucket_cfg = [1, 2, max_num_prefill_seqs, prompt_bs_limit]
        max_prompt_seq_limit = math.ceil(math.log2(max_prompt_seq)) + 1
        prompt_seq_bucket_cfg = [block_size, block_size, max_prompt_seq, max_prompt_seq_limit]

        if use_merged_prefill:
            logger().info("Merged prefill warmup is not implemented for exponential bucketing yet")

        msg = ("Prompt bucket config (min, step, max_warmup, limit) "
               f"bs:{prompt_bs_bucket_cfg}, "
               f"seq:{prompt_seq_bucket_cfg}")
        logger().info(msg)

        prompt_buckets, prompt_omitted_buckets = \
            generate_prompt_buckets(
            prompt_bs_bucket_cfg,
            prompt_seq_bucket_cfg,
            block_size,
            prefix_caching,
            max_num_batched_tokens,
            max_model_len)

        return sorted(prompt_buckets)


    def get_decode_buckets(self, max_num_seqs, block_size, 
                           max_num_batched_tokens, max_model_len,
                           num_max_blocks):
        self.check_for_user_flags('decode')
        prefix_caching = get_config().prefix_caching
        max_blocks = num_max_blocks

        # cfgs shape: [min, step, max, limit]
        decode_bs_limit = math.ceil(math.log2(max_num_seqs)) + 1
        decode_bs_bucket_cfg = [1, 2, max_num_seqs, decode_bs_limit]
        max_decode_block_limit = math.ceil(math.log2(max_blocks)) + 1
        decode_block_bucket_cfg = [block_size, block_size, max_blocks, max_decode_block_limit]

        msg = ("Decode bucket config (min, step, max_warmup, limit) "
               f"bs:{decode_bs_bucket_cfg}, "
               f"block:{decode_block_bucket_cfg}")
        logger().info(msg)

        decode_buckets = generate_decode_buckets(
            decode_bs_bucket_cfg, decode_block_bucket_cfg,
            num_max_blocks, max_model_len, block_size)

        return sorted(decode_buckets)


def generate_prompt_buckets(bs_bucket_config,
                            seq_bucket_config,
                            block_size,
                            prefix_caching,
                            max_num_batched_tokens=None,
                            max_model_len=None):
    _, _, bmax, _ = seq_bucket_config
    batch_size_buckets = warmup_range_with_limit(bs_bucket_config)
    long_context = False
    if bmax >= 8192:
        long_context = True
    seq_bucket_config = warmup_range_with_limit(seq_bucket_config, long_context=True)

    if prefix_caching:
        buckets_3d = []
        for bs in batch_size_buckets:
            for b in seq_bucket_config:
                buckets_3d.append((bs, b, 0))
                max_blocks_range = (bmax - b) // block_size
                if max_blocks_range == 0:
                    continue
                else:
                    num_buckets_3d = math.ceil(math.log2(max_blocks_range)) + 1

                for i in range(1, num_buckets_3d + 1):
                    power_unpadded = 1 * np.float_power(
                        max_blocks_range, (1 / float(num_buckets_3d)) * i)
                    new_bucket = math.ceil(power_unpadded)
                    buckets_3d.append((bs, b, new_bucket))

        buckets = buckets_3d
    else:
        buckets = list(
                itertools.product(batch_size_buckets,
                                seq_bucket_config, [0]))

    if len(buckets) == 0:
        msg = ("No buckets could be captured with following config "
               f"(min, step, max_warmup): "
               f"bs:{bs_bucket_config}, "
               f"seq:{seq_bucket_config}")
        raise ValueError(msg)

    filtered_buckets = buckets
    if max_num_batched_tokens is not None and max_model_len is not None:
        # Remove buckets exceeding batch token budget
        filtered_buckets = list(
            filter(
                lambda bucket: bucket[0] * (bucket[1] +  bucket[2] * block_size) <= max_num_batched_tokens \
                and bucket[1] <= max_model_len, buckets))

        if len(filtered_buckets) == 0:
            # we can handle this if we ignore max_num_batched_tokens
            min_bucket_bs, min_bucket_seq, min_bucket_ctx = min(buckets,
                                                key=lambda b: (b[0] * b[1]))
            min_reqd_budget = min_bucket_bs * (min_bucket_seq + min_bucket_ctx * block_size)
            msg = (
                "The current bucketing configuration "
                f"(min, step, max_warmup): "
                f"bs:{bs_bucket_config}, "
                f"seq:{seq_bucket_config} cannot be used with specified "
                f"max_num_batched_tokens ({max_num_batched_tokens}), as the "
                f"smallest bucket ({min_reqd_budget}) would exceed token "
                "budget. Please increase max_num_batched_tokens or decrease "
                "bucket minimum. Ignoring max_num_batched_tokens at risk of "
                "out-of-memory errors.")
            logger().warning(msg)
            return list(
                sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0]))), []

    captured_buckets = list(
        sorted(filtered_buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))
    omitted_buckets = list(
        sorted([x for x in buckets if x not in filtered_buckets]))
    return captured_buckets, omitted_buckets


def generate_decode_buckets(bs_bucket_config, blocks_bucket_config,
                            max_blocks, max_model_len, block_size, 
                            skip_invalid=False):
    buckets = []
    bs_buckets = warmup_range_with_limit(bs_bucket_config)
    tmp_blocks_bucket_config = blocks_bucket_config
    tmp_blocks_bucket_config = (*tmp_blocks_bucket_config[:2], max_blocks, tmp_blocks_bucket_config[-1])
    block_buckets = warmup_range_with_limit(tmp_blocks_bucket_config)
    last_bucket = max_blocks
    valid_blocks = set()
    if not skip_invalid:
        #NOTE(kzawora): this case will generate all possible combinations of
        # exponentially-spaced bs and blocks, even if combination is
        # invalid (exceeds max_model_len). Unfortunately, this is necessary 
        # to handle scenario where bucket dimensions are determined by 
        # get_padded_decode_num_blocks or get_padded_decode_batch_size, 
        # since they don't include information about the other dimension. 
        # This will need to be refactored at some point in the model runner,
        # but for now, we are dealing with this.
        valid_blocks = set((bs, 1, x) for x in sorted(block_buckets) for bs in bs_buckets)
    else:
        #NOTE(kzawora): this case will generate only valid combinations of
        # exponentially-spaced bs and blocks, where the product of bs and blocks
        # is less than or equal to max_model_len. To handle corner cases 
        # (e.g. longer context due to fragmentation), we're adding an additional
        # bucket with max_blocks for each batch size.
        # For this to work properly, bucket dimensions need be requested as 
        # a combination of (batch_size, num_blocks), not separately.
        for bs in bs_buckets:
            max_blocks_per_bs = min(bs * math.ceil(max_model_len / block_size), last_bucket)
            upper_bucket_bound = next(x for x in sorted(block_buckets) if x >= max_blocks_per_bs)
            valid_blocks = set((bs, 1, x) for x in sorted(block_buckets) if x <= upper_bucket_bound)
            
    buckets.extend(list(valid_blocks))
    return list(sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))


def warmup_range_with_limit(config: Tuple[int, int, int, int], long_context=False, fill=True):
    """ 
    NOTE(kzawora): we'll use exponential spacing for buckets in which scaled 
    power will return bmin for first bucket iteration, and bmax for last 
    iteration, with elements between determined by the exponent, and base being 
    unchanged. Note that after padding to bstep, duplicates may occur. 
    Handling of duplicates is configured by fill parameter. 
    If fill is False, duplicates are removed and less buckets are returned. 
    
    If fill is True, duplicates are resolved by selecting the closest (larger 
    or smaller) bucket. If duplicate resolution is not possible, less buckets 
    are returned. In that case, buckets are guaranteed to be linearly spaced.
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
 
    if fill is False:
        scaled_powers_padded   = [   128,    256,    256,    384,    512,    640,    896,    1152,    1536,    2048]
                                               ^_______^ 
                                               duplicates
        buckets                = [   128,    256,            384,    512,    640,    896,    1152,    1536,    2048]
                                                      ^ 
                                         duplicate bucket removed
        len(buckets) = 9, num_buckets = 10
    if fill is True:
        buckets                = [   128,    256,    384,    512,    640,    768,    896,    1152,    1536,    2048]
                                                      ^_______^_______^_______^ 
                                                   closest unused buckets selected
                                                              ^_______^_______^ 
                                      these become duplicates once previous duplicates are resolved
        
        In this case we'll have four duplicated buckets:
        174.18 -> 256, optimal bucket,
        237.02 -> (256) -> 384, taking closest available bucket, 
            as optimal bucket 256 was already captured by 174.18, 
        322.54 -> (384) -> 512, taking closest available bucket, 
            as optimal bucket 384 was already captured by 237.02,
        438.91 -> (512) -> 640, taking closest available bucket, 
            as optimal bucket 512 was already captured by 322.54,
        597.26 -> (640) -> 768, taking closest available bucket, 
            as optimal bucket 640 was already captured by 438.91,
        812.75 -> 896, optimal bucket
        len(buckets) = 10, num_buckets = 10
        In this case, the end result has the same buckets as fill=False, 
        but with additional bucket 768 added. 
        The difference is more pronounced for larger ranges and larger number 
        of buckets.
    """ # noqa: E501

    bmin, bstep, bmax, num_buckets = config
    linear_buckets = set(np.arange(bmin, bmax + 1, step=bstep))
    assert num_buckets > 0, "num_buckets must be a positive integer"
    if num_buckets == 1:
        return [bmax]
    buckets: Set[Tuple[int, int]] = set()

    if long_context:
        num_buckets_exp = math.floor(num_buckets / 2)
        num_buckets_linear = num_buckets - num_buckets_exp
        first_step = bmax / num_buckets #or i.e. * 0.25
    else:
        num_buckets_exp = num_buckets
        first_step = bmax

    for i in range(num_buckets_exp):
        power_unpadded = bmin * np.float_power(
            first_step / bmin, (1. / float(num_buckets_exp - 1)) * i)
        if i == num_buckets - 1 and get_config().use_contiguous_pa:
            bucket = bmax
        else:
            bucket = math.ceil(power_unpadded / bstep) * bstep
        if fill and bucket in buckets:
            available_buckets = linear_buckets.difference(buckets)
            if len(available_buckets) == 0:
                break  # there are no more unique buckets, let's exit now
            new_bucket = min(available_buckets,
                             key=lambda x: abs(x - power_unpadded))
            buckets.add(new_bucket)
        else:
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
    return list(sorted(buckets))
