import itertools
import logging
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Set, Tuple

from vllm_gaudi.extension.logger import logger as logger
from vllm_gaudi.extension.runtime import get_config


class UnifiedBucketingStrategy():

    def get_unified_cfgs(self, bs, max_model_len, block_size, max_blocks, max_num_batched_tokens):
        # [min, max, turning_point]
        query_cfg = [1, max_num_batched_tokens, bs]
        max_shared_ctx = min(math.ceil(max_model_len // block_size), max_blocks)
        shared_ctx_cfg = [0, max_shared_ctx, bs]
        max_unique_ctx = max_blocks
        unique_ctx_cfg = [0, max_unique_ctx, bs]
        return query_cfg, shared_ctx_cfg, unique_ctx_cfg

    def get_range(self, cfg):
        range_for_cfg = warmup_unified_range(cfg)
        return sorted(range_for_cfg)


def warmup_unified_range(cfg):
    bmin, bmax, turning_point = cfg
    limit = 10
    round_up = 128

    buckets: Set[Tuple[int, int]] = set()

    if bmin == 0:
        buckets.add(bmin)
        bmin = 1

    num_buckets_exp = limit
    first_step = bmax

    for i in range(num_buckets_exp):
        power_unpadded = bmin * np.float_power(first_step / bmin, (1. / float(num_buckets_exp - 1)) * i)
        if i == limit - 1:
            bucket = bmax
        else:
            bucket = math.ceil(power_unpadded / round_up) * round_up
        buckets.add(bucket)

    return list(sorted(buckets))
