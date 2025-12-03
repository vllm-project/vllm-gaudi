import os
import bisect
import math
import itertools
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
    unified_buckets: List[Tuple[int, int, int]] = []
    # Seed buckets are the buckets originally generated from bucketing configuration
    # Spec decode may automatically add new buckets based on the seed buckets
    seed_decode_buckets: List[Tuple[int, int, int]] = None
    initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HPUBucketingManager, cls).__new__(cls)
        return cls._instance

    def initialize(self,
                   max_num_seqs,
                   max_num_prefill_seqs,
                   block_size,
                   max_num_batched_tokens,
                   max_model_len,
                   num_speculative_tokens=0):
        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_hpu_blocks = None
        self.max_model_len = max_model_len
        self.num_speculative_tokens = num_speculative_tokens
        self.initialized = True
        self.fallback_bs_base_step = 2
        self.fallback_seq_base_step = 32
        self.fallback_blocks_base_step = 32

        self.use_sliding_window = get_config().PT_HPU_SDPA_QKV_SLICE_MODE_FWD
        if self.use_sliding_window:
            self.slice_size = get_config().PT_HPU_SDPA_BC_FACTOR if \
                get_config().PT_HPU_SDPA_BC_FACTOR is not None else 1024
            self.slice_thld = get_config().VLLM_FUSEDSDPA_SLIDE_THLD if \
                get_config().VLLM_FUSEDSDPA_SLIDE_THLD is not None else 8192

            msg = (
                f"use_sliding_window {self.use_sliding_window}, slice_size {self.slice_size}, threshold {self.slice_thld}"
            )
            logger().info(msg)

    ### GENERATE BUCKETS FUNCTIONS ###

    def read_from_file(self, is_prompt):
        file_name = get_config().VLLM_BUCKETING_FROM_FILE
        from vllm_gaudi.extension.bucketing.file_strategy import (FileBucketingStrategy)
        strategy = FileBucketingStrategy()
        return strategy.get_buckets(file_name, is_prompt)

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

    def generate_unified_buckets(self):
        if self.initialized:
            if get_config().VLLM_BUCKETING_FROM_FILE:
                assert "Unified attention doesn't support bucketing from file"
            from vllm_gaudi.extension.bucketing.unified import (UnifiedBucketingStrategy)
            strategy = UnifiedBucketingStrategy()

            query_cfg, shared_ctx_cfg, unique_ctx_cfg = strategy.get_unified_cfgs(
                bs=self.max_num_seqs,
                max_model_len=self.max_model_len,
                block_size=self.block_size,
                max_blocks=self.num_hpu_blocks,
                max_num_batched_tokens=self.max_num_batched_tokens)
            query_range = strategy.get_range(query_cfg)
            shared_ctx_range = strategy.get_range(shared_ctx_cfg)
            unique_ctx_range = strategy.get_range(unique_ctx_cfg)

            self.unified_buckets = generate_unified_buckets(query_range, shared_ctx_range, unique_ctx_range,
                                                            self.max_num_seqs, self.block_size, self.max_model_len)

            msg = (f"Generated {len(self.unified_buckets)} "
                   f"unified buckets [query, shared_blocks, unique_blocks]: "
                   f"{list(self.unified_buckets)}")
            logger().info(msg)
        else:
            logger().info("Bucketing is off - skipping prompt buckets generation")
            self.unified_buckets = []
        return

    def generate_prompt_buckets(self):
        if self.initialized:
            buckets_from_file = None
            bs_range = []
            query_range = []
            ctx_range = []
            if get_config().VLLM_BUCKETING_FROM_FILE:
                buckets_from_file = self.read_from_file(is_prompt=True)
            else:
                strategy = self.get_bucketing_strategy()

                bs_cfg, query_cfg, ctx_cfg = strategy.get_prompt_cfgs(
                    max_num_prefill_seqs=self.max_num_prefill_seqs,
                    block_size=self.block_size,
                    max_num_batched_tokens=self.max_num_batched_tokens,
                    max_model_len=self.max_model_len)

                bs_range = strategy.get_range(bs_cfg)
                query_range = strategy.get_range(query_cfg)
                ctx_range = strategy.get_range(ctx_cfg)

            self.prompt_buckets = generate_buckets(bs_range, query_range, ctx_range, True, self.max_model_len,
                                                   self.max_num_seqs, self.max_num_prefill_seqs,
                                                   self.max_num_batched_tokens, self.block_size, self.num_hpu_blocks,
                                                   buckets_from_file)
            self.log_generate_info(True)
            if self.use_sliding_window:
                self.prompt_buckets = [
                    t for t in self.prompt_buckets
                    if t[2] != 0 or (t[2] == 0 and (t[1] < self.slice_thld or
                                                    (t[1] >= self.slice_thld and t[1] % self.slice_size == 0)))
                ]
                self.log_generate_info(True)
        else:
            logger().info("Bucketing is off - skipping prompt buckets generation")
            self.prompt_buckets = []
        return

    def generate_decode_buckets(self):
        if self.initialized:
            buckets_from_file = None
            bs_range = []
            query_range = []
            ctx_range = []
            if get_config().VLLM_BUCKETING_FROM_FILE:
                buckets_from_file = self.read_from_file(is_prompt=False)
            else:
                strategy = self.get_bucketing_strategy()

                bs_cfg, query_cfg, ctx_cfg = strategy.get_decode_cfgs(
                    max_num_seqs=self.max_num_seqs,
                    block_size=self.block_size,
                    max_num_batched_tokens=self.max_num_batched_tokens,
                    max_model_len=self.max_model_len,
                    max_blocks=self.num_hpu_blocks)

                bs_range = strategy.get_range(bs_cfg)
                query_range = strategy.get_range(query_cfg)
                ctx_range = strategy.get_range(ctx_cfg)

                if get_config().use_contiguous_pa and ctx_range[-1] < self.num_hpu_blocks:
                    ctx_range.append(self.num_hpu_blocks)

            self.decode_buckets = generate_buckets(bs_range, query_range, ctx_range, False, self.max_model_len,
                                                   self.max_num_seqs, self.max_num_prefill_seqs,
                                                   self.max_num_batched_tokens, self.block_size, self.num_hpu_blocks,
                                                   buckets_from_file)
            if self.num_speculative_tokens:
                # The existing buckets are used as seed decode buckets
                self.seed_decode_buckets = self.decode_buckets
                # More buckets are added automatically for spec decode
                self.decode_buckets = self.generate_spec_decode_buckets(self.decode_buckets)

            self.log_generate_info(False)
        else:
            logger().info("Bucketing is off - skipping decode buckets generation")
            self.decode_buckets = []
        return

    def log_generate_info(self, is_prompt=False):
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
        if self.use_sliding_window and seq_len >= self.slice_thld:
            new_seq_len = math.ceil(seq_len / self.slice_size) * self.slice_size
        else:
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

    def find_decode_bucket(self, batch_size, num_blocks, seed_buckets: bool = False):
        if self.initialized:
            if seed_buckets and self.seed_decode_buckets is not None:
                found_bucket = find_equal_or_closest_greater_config(self.seed_decode_buckets,
                                                                    (batch_size, 1, num_blocks))
                if found_bucket is not None:
                    return found_bucket

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

    def find_unified_bucket(self, query, shared_ctx, unique_ctx, is_causal):
        if self.initialized:
            # TODO: handle is_causal
            found_bucket = find_equal_or_closest_greater_config(self.unified_buckets,
                                                                (query, shared_ctx, unique_ctx, is_causal))
            if found_bucket is None:
                logger().warning(f"No bucket found for: {(query, shared_ctx, unique_ctx)}")
                return (query, shared_ctx, unique_ctx)
            return found_bucket
        return (query, shared_ctx, unique_ctx)

    def get_max_prompt_shape(self):
        return max(b[1] for b in self.prompt_buckets) \
               if len(self.prompt_buckets) > 0 else self.max_model_len

    def generate_spec_decode_buckets(self, seed_decode_buckets):
        max_model_len = self.max_model_len
        block_size = self.block_size

        def no_corrections(bs, query, ctx):
            return (bs, query, ctx)

        def correct_for_max_model_len(bs, query, ctx):
            return (bs, query, min(ctx, bs * math.ceil(max_model_len / block_size)))

        def get_corrector(use_contiguous_pa):
            if use_contiguous_pa:
                return no_corrections
            else:
                return correct_for_max_model_len

        use_contiguous_pa = get_config().use_contiguous_pa
        corrector = get_corrector(use_contiguous_pa)

        # If spec decode enabled, generate buckets for batch_size * (1 + num_speculative_tokens)
        num_tokens = 1 + self.num_speculative_tokens
        buckets = set()
        for bucket in seed_decode_buckets:
            buckets.add(bucket)
            bs, query, ctx = bucket
            spec_decode_bs = bs * num_tokens
            if spec_decode_bs <= ctx:
                # Add a bucket with (batch_size * num_tokens, query, ctx)
                buckets.add(corrector(spec_decode_bs, query, ctx))
            # Add a bucket with (batch_size * num_tokens, query, ctx * num_tokens)
            buckets.add(corrector(spec_decode_bs, query, ctx * num_tokens))

        # Log the new generated spec decode buckets
        new_buckets = sorted(buckets - set(seed_decode_buckets))
        msg = (f"Generated {len(new_buckets)} "
               f"spec decode buckets [bs, query, num_blocks]: {list(new_buckets)}")
        logger().info(msg)

        return sorted(buckets)

    @classmethod
    def get_instance(cls):
        """
        Retrieve the singleton instance of the class.
        """
        return cls._instance


def get_bucketing_manager():
    instance = HPUBucketingManager.get_instance()
    return instance


def generate_buckets(bs_range,
                     query_range,
                     ctx_range,
                     is_prompt,
                     max_model_len,
                     max_num_seqs,
                     max_num_prefill_seqs,
                     max_num_batched_tokens,
                     block_size,
                     max_blocks,
                     file_buckets=None):
    use_merged_prefill = get_config().merged_prefill
    use_contiguous_pa = get_config().use_contiguous_pa

    def expand_to_neighbor_buckets(bs_idx, bs_range, ctx_idx, ctx_range, max_num_batched_tokens):
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

        candidates = [(bs_idx, ctx_idx), (bs_idx + 1, ctx_idx), (bs_idx, ctx_idx + 1), (bs_idx + 1, ctx_idx + 1)]
        valid_candidates = [(b_idx, q_idx) for b_idx, q_idx in candidates
                            if b_idx < len(bs_range) and q_idx < len(ctx_range)]
        return {(bs_range[b_idx], ctx_range[q_idx]) for b_idx, q_idx in valid_candidates}

    # filter rules for buckets
    # prompt
    def not_over_max_model_len(bs, query, ctx):
        smaller_than_limit = (query + ctx * block_size) <= max_model_len
        if not smaller_than_limit:
            omitted_buckets.add(
                ("condition: (query + ctx * block_size) <= max_model_len", "-> bs, query, ctx: ", bs, query, ctx))
        return smaller_than_limit

    def not_over_max_num_batched_tokens(bs, query, ctx):
        smaller_than_limit = bs * query <= max_num_batched_tokens
        if not smaller_than_limit:
            omitted_buckets.add(
                ("condition: bs * query <= max_num_batched_tokens", "-> bs, query, ctx: ", bs, query, ctx))
        return smaller_than_limit

    def ctx_not_over_max_ctx_for_merged_prefill(bs, query, ctx):
        smaller_than_limit = ctx <= max_num_prefill_seqs * math.ceil(
            (max_model_len - math.floor(query / max_num_prefill_seqs)) // block_size)
        if not smaller_than_limit:
            omitted_buckets.add((
                "ctx <= max_num_prefill_seqs * math.ceil((max_model_len - math.floor(query / max_num_prefill_seqs)) // block_size)",
                "-> bs, query, ctx: ", bs, query, ctx))
        return smaller_than_limit

    def no_corrections(bs, query, ctx):
        return (bs, query, ctx)

    def correct_for_max_model_len(bs, query, ctx):
        return (bs, query, min(ctx, bs * math.ceil(max_model_len / block_size)))

    def batch_size_smaller_than_blocks(bs, query, ctx):
        if not bs <= ctx:
            omitted_buckets.add(("condition: bs <= ctx, ", "-> bs, query, ctx: ", bs, query, ctx))
        return bs <= ctx

    filters_map = {
        "prompt": {
            # depends only on merged_prefill
            True: [ctx_not_over_max_ctx_for_merged_prefill],
            False: [not_over_max_model_len, not_over_max_num_batched_tokens],
        },
        "decode": {
            # depends only on contiguous PA
            True: [],
            False: [batch_size_smaller_than_blocks],
        }
    }

    def get_filters(is_prompt, use_merged_prefill, use_contiguous_pa):
        phase = "prompt" if is_prompt else "decode"
        if is_prompt:
            return filters_map[phase][use_merged_prefill]
        return filters_map[phase][use_contiguous_pa]

    def get_corrector(is_prompt, use_contiguous_pa):
        if is_prompt or use_contiguous_pa:
            return no_corrections
        else:
            return correct_for_max_model_len

    buckets = set()
    buckets_2d = set()
    omitted_buckets = set()
    filters = get_filters(is_prompt, use_merged_prefill, use_contiguous_pa)
    corrector = get_corrector(is_prompt, use_contiguous_pa)

    if file_buckets:
        for bs, query, blocks in file_buckets:
            if all(bucket_filter(bs, query, blocks) for bucket_filter in filters):
                buckets.add(corrector(bs, query, blocks))
    else:
        for bs_idx, bs in enumerate(bs_range):
            for ctx_idx, ctx in enumerate(ctx_range):
                local_buckets = expand_to_neighbor_buckets(bs_idx, bs_range, ctx_idx, ctx_range,
                                                           max_num_batched_tokens) if not is_prompt else {(bs, ctx)}
                buckets_2d.update(local_buckets)

        for bs, ctx in buckets_2d:
            for query in query_range:
                if all(bucket_filter(bs, query, ctx) for bucket_filter in filters):
                    buckets.add(corrector(bs, query, ctx))
    if not buckets:
        phase = 'prompt' if is_prompt else 'decode'
        for bucket in omitted_buckets:
            logger().error(bucket)
        raise RuntimeError(
            "Generated 0 " + phase +
            " buckets. Please use default exponential bucketing, VLLM_EXPONENTIAL_BUCKETING=true or generate linear warmup flags according to README"
        )

    return sorted(buckets)


def generate_unified_buckets(query_range, shared_ctx_range, unique_ctx_range, bs, block_size, max_model_len):
    buckets = set()
    is_causal = [0, 1]

    for query, shared_ctx, unique_ctx, causal in itertools.product(query_range, shared_ctx_range, unique_ctx_range,
                                                                   is_causal):
        if causal:
            max_bs = min(bs, query)
            if math.ceil(shared_ctx * block_size // max_bs) <= max_model_len:
                buckets.add((query, shared_ctx, unique_ctx, causal))
        elif query <= bs:
            # non causal query = current bs
            if shared_ctx > 0 or unique_ctx > 0:
                if shared_ctx == 0 or (math.ceil(shared_ctx * block_size // (query // 2)) <= max_model_len):
                    if shared_ctx > 0 or query <= unique_ctx:
                        buckets.add((query, shared_ctx, unique_ctx, causal))

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
