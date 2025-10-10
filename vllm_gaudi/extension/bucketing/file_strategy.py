import itertools
import operator
import os
import math
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_gaudi.extension.logger import logger as logger
from vllm_gaudi.extension.runtime import get_config


class FileBucketingStrategy:
    def get_buckets(self, file_name, is_prompt):
        prompt_buckets = []
        decode_buckets = []

        with open(file_name, 'r') as f:
            for line in f:
                bucket = line.strip()
                if not bucket or bucket.startswith('#'):
                    continue

                bucket = bucket.strip('()')
                cfg = bucket.split(',')

                x, y, z = map(int, cfg)

                if y == 1:
                    decode_buckets.append((x, y, z))
                else:
                    prompt_buckets.append((x, y, z))

        return sorted(prompt_buckets) if is_prompt else sorted(decode_buckets)
