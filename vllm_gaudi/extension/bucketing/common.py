import os
import bisect
import math
from typing import Dict
import inspect
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_gaudi.extension.logger import logger as logger
from vllm_gaudi.extension.runtime import get_config

class VisionBuckets:
    '''
    This class is used to bucket image tokens
    '''

    def __init__(self, is_batch_based):
        self.is_batch_based = is_batch_based
        envvar = os.environ.get('VLLM_MULTIMODAL_BUCKETS', "")
        if envvar == 'None':
            self.multimodal_buckets = None
        else:
            if envvar == "":
                if is_batch_based:
                    multimodal_buckets = [1, 2, 4, 8]  # batch sizes for gemma3
                else:
                    multimodal_buckets = [
                        1600, 3136, 4096, 6400, 7744, 9216, 12544
                    ]
            else:
                multimodal_buckets = [int(i) for i in envvar.split(',')]
            self.multimodal_buckets = self._process_buckets(multimodal_buckets)

    def _process_buckets(self, buckets):
        if not self.is_batch_based:
            for bucket in buckets:
                assert bucket % 8 == 0, (
                    'Buckets needs to be multiples 8 (slices of 64)')
        return sorted(buckets)

    def get_multimodal_bucket(self, curr_num_image_patches):
        if self.multimodal_buckets is not None:
            for mm_bucket in self.multimodal_buckets:
                if curr_num_image_patches <= mm_bucket:
                    return mm_bucket
            return curr_num_image_patches
        else:
            return 0

    def __repr__(self):
        return str(self.multimodal_buckets)
        

def calc_fallback_value(n: int, base_step: int):
    """ Calculate next bucket for yet unbucketized value"""
    if n <= 1:
        return n
    power = 1/3
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

    def initialize(self, max_num_seqs, max_num_prefill_seqs, block_size,
                   max_num_batched_tokens, max_model_len):
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

    def get_bucketing_strategy(self):
        strategy = None
        # TODO - we can use different strategies for decode and prompt
        use_exponential_bucketing = True if \
                get_config().VLLM_EXPONENTIAL_BUCKETING == None else \
                get_config().VLLM_EXPONENTIAL_BUCKETING
        
        if use_exponential_bucketing:
            from vllm_gaudi.extension.bucketing.exponential import (
                ExponentialBucketingStrategy)
            strategy = ExponentialBucketingStrategy()
        else:
            from vllm_gaudi.extension.bucketing.linear import LinearBucketingStrategy
            strategy = LinearBucketingStrategy()
        return strategy

    def generate_prompt_buckets(self):
        if self.initialized:
            strategy = self.get_bucketing_strategy()

            self.prompt_buckets = strategy.get_prompt_buckets(
                            max_num_prefill_seqs = self.max_num_prefill_seqs,
                            block_size = self.block_size,
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_model_len = self.max_model_len)
            self.log_generate_info(True)
        else:
            logger().info("Bucketing is off - skipping prompt buckets generation")
            self.prompt_buckets = []
        return

    def generate_decode_buckets(self):
        if self.initialized:
            strategy = self.get_bucketing_strategy()

            self.decode_buckets = strategy.get_decode_buckets(
                            max_num_seqs = self.max_num_seqs,
                            block_size = self.block_size, 
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_model_len = self.max_model_len, 
                            num_max_blocks = self.num_hpu_blocks)
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

    def generate_fallback_bucket(self, batch_size, seq_len, ctx):
        assert self.max_num_batched_tokens is not None
        new_batch_size = calc_fallback_value(batch_size, self.fallback_bs_base_step)
        new_seq_len = min(calc_fallback_value(seq_len, self.fallback_seq_base_step),
                          self.max_num_batched_tokens)
        if self.num_hpu_blocks is None:
            new_ctx = 0
        else:
            new_ctx = min(calc_fallback_value(ctx, self.fallback_blocks_base_step),
                      self.num_hpu_blocks)
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


def is_greater_or_equal(tuple1, tuple2):
    return tuple1[0] >= tuple2[0] and tuple1[1] >= tuple2[1] \
           and tuple1[2] >= tuple2[2]


def find_equal_or_closest_greater_config(sorted_list, target_tuple):
    idx = bisect.bisect_left(sorted_list, target_tuple)
    for i in range(idx, len(sorted_list)):
        if is_greater_or_equal(sorted_list[i], target_tuple):
            return sorted_list[i]
    return None

