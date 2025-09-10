import os
import bisect
import math
from typing import Dict
import inspect
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_gaudi.extension.logger import logger as logger
from vllm_gaudi.extension.runtime import get_config


def calc_fallback_value(n: int, base_step: int):
    """ Calculate next bucket for yet unbucketized value"""
    if n <= 1:
        return n
    power = 1 / 3
    # The basic idea is that we first estimate bucket size based
    # on exponent of the number, so higher numbers will generate
    # bigger gaps between individual buckets, but it's not as steep
    # as exponential bucketing. Additionally this has a nice
    # property that generated values are guaranteed to be divisible
    # by base_step
    #
    # examples:
    # n=31, base_step=32
    #   => bucket_size = ceil(31^1/3) * 32 = 4 * 32 = 128
    #   => next_value = round_up(31, 128) = 128
    # n=4001, base_step=32
    #   => bucket_size = ceil(4001^1/3) * 32 = 16 * 32 = 512
    #   => next_value = round_up(4001, 512) = 4096
    bucket_size = math.ceil(math.pow(n, power)) * base_step
    return math.ceil(n / bucket_size) * bucket_size


class HPUBucketingManager():
    _instance = None
    prompt_buckets: List[Tuple[int, int, int]] = []
    decode_buckets: List[Tuple[int, int, int]] = []
    initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HPUBucketingManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, max_num_seqs, max_num_prefill_seqs, block_size, max_num_batched_tokens, max_model_len):
        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_hpu_blocks = None
        self.max_model_len = max_model_len
        self.initialized = True

        self.fallback_bs_base_step = 2
        self.fallback_seq_base_step = 32
        self.fallback_blocks_base_step = 32

    ### GENERATE BUCKETS FUNCTIONS ###

    def get_bucketing_strategy(self):
        strategy = None
        # TODO - we can use different strategies for decode and prompt
        use_exponential_bucketing = True if \
                get_config().VLLM_EXPONENTIAL_BUCKETING == None else \
                get_config().VLLM_EXPONENTIAL_BUCKETING

        if use_exponential_bucketing:
            from vllm_gaudi.extension.bucketing.exponential import (ExponentialBucketingStrategy)
            strategy = ExponentialBucketingStrategy()
        else:
            from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
            strategy = LinearBucketingStrategy()
        return strategy

    def generate_prompt_buckets(self):
        if self.initialized:
            strategy = self.get_bucketing_strategy()

            bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(max_num_prefill_seqs=self.max_num_prefill_seqs,
                                                                  block_size=self.block_size,
                                                                  max_num_batched_tokens=self.max_num_batched_tokens,
                                                                  max_model_len=self.max_model_len)

            bs_range = strategy.get_range(bs_cfg)
            query_range = strategy.get_range(query_cfg)
            ctx_range = strategy.get_range(ctx_cfg)

            self.prompt_buckets = generate_buckets(bs_range, query_range, ctx_range, True, self.max_model_len,
                                                   self.max_num_seqs, self.max_num_prefill_seqs,
                                                   self.max_num_batched_tokens, self.block_size, self.num_hpu_blocks)
            self.log_generate_info(True)
        else:
            logger().info("Bucketing is off - skipping prompt buckets generation")
            self.prompt_buckets = []
        return

    def generate_decode_buckets(self):
        if self.initialized:
            strategy = self.get_bucketing_strategy()

            bs_cfg, query_cfg, ctx_cfg = strategy.get_decode_cfgs(max_num_seqs=self.max_num_seqs,
                                                                  block_size=self.block_size,
                                                                  max_num_batched_tokens=self.max_num_batched_tokens,
                                                                  max_model_len=self.max_model_len,
                                                                  max_blocks=self.num_hpu_blocks)

            bs_range = strategy.get_range(bs_cfg)
            query_range = strategy.get_range(query_cfg)
            ctx_range = strategy.get_range(ctx_cfg)

            if get_config().use_contiguous_pa and ctx_range[-1] < self.num_hpu_blocks:
                ctx_range.append(self.num_hpu_blocks)

            print(ctx_range)

            self.decode_buckets = generate_buckets(bs_range, query_range, ctx_range, False, self.max_model_len,
                                                   self.max_num_seqs, self.max_num_prefill_seqs,
                                                   self.max_num_batched_tokens, self.block_size, self.num_hpu_blocks)
            self.log_generate_info(False)
        else:
            logger().info("Bucketing is off - skipping decode buckets generation")
            self.decode_buckets = []
        return

    def log_generate_info(self, is_prompt):
        phase = 'prompt' if is_prompt else 'decode'
        buckets = self.prompt_buckets if is_prompt else self.decode_buckets
        msg = (f"Generated {len(buckets)} "
               f"{phase} buckets [bs, query, num_blocks]: "
               f"{list(buckets)}")
        logger().info(msg)

    ### RETRIEVE BUCKETS FUNCTIONS ###

    def generate_fallback_bucket(self, batch_size, seq_len, ctx):
        assert self.max_num_batched_tokens is not None
        new_batch_size = calc_fallback_value(batch_size, self.fallback_bs_base_step)
        new_seq_len = min(calc_fallback_value(seq_len, self.fallback_seq_base_step), self.max_num_batched_tokens)
        if self.num_hpu_blocks is None:
            new_ctx = 0
        else:
            new_ctx = min(calc_fallback_value(ctx, self.fallback_blocks_base_step), self.num_hpu_blocks)
        return (new_batch_size, new_seq_len, new_ctx)

    def find_prompt_bucket(self, batch_size, seq_len, ctx=0):
        if self.initialized:
            found_bucket = find_equal_or_closest_greater_config(self.prompt_buckets, (batch_size, seq_len, ctx))
            if found_bucket is None:
                new_bucket = self.generate_fallback_bucket(batch_size, seq_len, ctx)
                logger().warning(f"Prompt bucket for {batch_size, seq_len, ctx}"
                                 f" was not prepared. Adding new bucket: {new_bucket}")
                self.prompt_buckets.append(new_bucket)
                self.prompt_buckets.sort()
                return new_bucket
            return found_bucket
        return (batch_size, seq_len, ctx)

    def find_decode_bucket(self, batch_size, num_blocks):
        if self.initialized:
            found_bucket = find_equal_or_closest_greater_config(self.decode_buckets, (batch_size, 1, num_blocks))
            if found_bucket is None:
                new_bucket = self.generate_fallback_bucket(batch_size, 1, num_blocks)
                logger().warning(f"Decode bucket for {batch_size, 1, num_blocks}"
                                 f" was not prepared. Adding new bucket: {new_bucket}")
                self.decode_buckets.append(new_bucket)
                self.decode_buckets.sort()
                return new_bucket
            return found_bucket
        return (batch_size, 1, num_blocks)

    def get_max_prompt_shape(self):
        return max(b[1] for b in self.prompt_buckets) \
               if len(self.prompt_buckets) > 0 else self.max_model_len

    @classmethod
    def get_instance(cls):
        """
        Retrieve the singleton instance of the class.
        """
        return cls._instance


def get_bucketing_manager():
    instance = HPUBucketingManager.get_instance()
    return instance


def generate_buckets(bs_range, query_range, ctx_range, is_prompt, max_model_len, max_num_seqs, max_num_prefill_seqs,
                     max_num_batched_tokens, block_size, max_blocks):
    use_merged_prefill = get_config().merged_prefill
    use_contiguous_pa = get_config().use_contiguous_pa

    def expand_to_neighbor_buckets(bs_idx, bs_range, query_idx, query_range, max_num_batched_tokens):
        '''
        Expand 2d bucket (bs, query) to include:
        - itself
        - next bs value (if any)
        - next query value (if any)
        - next bs and query values together (if both exists)
        This cover case when our configuration is in budget but between
        values that are in and out of budget:
        bs < edge_case_bs < next bs and query < edge_case_query < next query
        '''
        candidates = [(bs_idx, query_idx), (bs_idx + 1, query_idx), (bs_idx, query_idx + 1),
                      (bs_idx + 1, query_idx + 1)]
        valid = bs_range[bs_idx] * query_range[query_idx] <= max_num_batched_tokens
        if not valid:
            return {}
        valid_candidates = [(b_idx, q_idx) for b_idx, q_idx in candidates
                            if b_idx < len(bs_range) and q_idx < len(query_range)]
        return {(bs_range[b_idx], query_range[q_idx]) for b_idx, q_idx in valid_candidates}

    # filter rules for buckets
    # prompt
    def not_over_max_model_len(bs, query, ctx):
        return query + ctx * block_size <= max_model_len

    def ctx_not_over_max_ctx_for_merged_prefill(bs, query, ctx):
        return ctx <= max_num_prefill_seqs * math.ceil(
            (max_model_len - math.floor(query / max_num_prefill_seqs)) // block_size)

    # decode
    def block_not_greater_than_max_model_len(bs, query, ctx):
        return ctx <= bs * math.ceil(max_model_len / block_size)

    def batch_size_smaller_than_blocks(bs, query, ctx):
        return bs <= ctx

    filters_map = {
        "prompt": {
            # depends only on merged_prefill
            True: [ctx_not_over_max_ctx_for_merged_prefill],
            False: [not_over_max_model_len],
        },
        "decode": {
            # depends only on contiguous PA
            True: [],
            False: [block_not_greater_than_max_model_len, batch_size_smaller_than_blocks],
        }
    }

    def get_filters(is_prompt, use_merged_prefill, use_contiguous_pa):
        phase = "prompt" if is_prompt else "decode"
        if is_prompt:
            return filters_map[phase][use_merged_prefill]
        else:
            return filters_map[phase][use_contiguous_pa]
        return []

    buckets = set()
    buckets_2d = set()
    filters = get_filters(is_prompt, use_merged_prefill, use_contiguous_pa)
    for bs_idx, bs in enumerate(bs_range):
        for query_idx, query in enumerate(query_range):
            buckets_2d.update(
                expand_to_neighbor_buckets(bs_idx, bs_range, query_idx, query_range, max_num_batched_tokens))

    for bs, query in buckets_2d:
        for ctx in ctx_range:
            if all(bucket_filter(bs, query, ctx) for bucket_filter in filters):
                buckets.add((bs, query, ctx))
    return sorted(buckets)


def is_greater_or_equal(tuple1, tuple2):
    return tuple1[0] >= tuple2[0] and tuple1[1] >= tuple2[1] \
           and tuple1[2] >= tuple2[2]


def find_equal_or_closest_greater_config(sorted_list, target_tuple):
    idx = bisect.bisect_left(sorted_list, target_tuple)
    for i in range(idx, len(sorted_list)):
        if is_greater_or_equal(sorted_list[i], target_tuple):
            return sorted_list[i]
    return None
