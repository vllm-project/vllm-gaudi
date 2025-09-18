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
        # [min, step, max, turning_point]
        query_cfg = [128, 128, max_num_batched_tokens, bs]
        max_shared_ctx = math.ceil(max_model_len // block_size) * bs
        shared_ctx_cfg = [0, 1, max_shared_ctx, bs]
        max_unique_ctx = max_blocks # // 2 - 500 # TODO: OOM
        unique_ctx_cfg = [0, 1, max_unique_ctx, bs]
        return query_cfg, shared_ctx_cfg, unique_ctx_cfg

    def get_range(self, cfg):
        range_for_cfg = warmup_unified_range(cfg)
        return sorted(range_for_cfg)

def exponential_range(number_of_values, minimum_value, rounding_value, maximum_value):
    values_range = []
    if number_of_values == 1:
        return [maximum_value]
    number_of_exponential_values = number_of_values
    for i in range(number_of_values):
        power_unpadded = minimum_value * np.float_power(maximum_value / minimum_value, (1. / float(number_of_exponential_values - 1)) * i)
        new_value = math.ceil(power_unpadded / rounding_value) * rounding_value
        values_range.append(new_value)
    return values_range
    

def warmup_unified_range(cfg):
    '''
    [min, step, max, limit, turning_point]
        bs
           ↑
           |                          max   \
           |                          .      |
           |                         .       |
           |                        .        |
           |                       .         | - exponantial with base1
           |                     .           |
           |                  .              |
           |              .                  |
           |          .                     /
           |         X - turning point               
           |         .                      \
           |        .                        |
           |      .                          | - exponential with base0
           |    .                            |
           | min                            /
           +-------------------------------------------→ 
    '''        
    bmin, bstep, bmax, turning_point = cfg

    buckets: Set[Tuple[int, int]] = set()

    if bmin == 0:
        buckets.add(bmin)

    # alpha version: [bs/4, bs/2, bs, bt/4, bt/2, bt]

    buckets.add(turning_point//4)
    buckets.add(turning_point//2)
    buckets.add(turning_point)
    buckets.add(bmax//4)
    buckets.add(bmax//2)
    buckets.add(bmax)

    return list(sorted(buckets))
