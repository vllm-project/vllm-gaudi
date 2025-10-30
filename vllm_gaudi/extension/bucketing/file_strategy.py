import itertools
import operator
import os
import math
import ast
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
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    bucket = eval(line, {"__builtins__": None}, {"range": range})
                except Exception as e:
                    print(f"Skipping line due to eval error: {e} - {line}")
                    continue

                if not isinstance(bucket, tuple) or len(bucket) != 3:
                    print('Skipping line due to incorrect format - ', bucket)
                    continue

                x_num = ensure_is_list(bucket[0])
                y_num = ensure_is_list(bucket[1])
                z_num = ensure_is_list(bucket[2])

                for full_bucket in itertools.product(x_num, y_num, z_num):
                    x, y, z = map(int, full_bucket)
                    if y == 1:
                        decode_buckets.append((x, y, z))
                    else:
                        prompt_buckets.append((x, y, z))
        return sorted(prompt_buckets) if is_prompt else sorted(decode_buckets)


def ensure_is_list(value):
    if isinstance(value, list):
        return value
    elif isinstance(value, range):
        return list(value)
    else:
        return [value]
