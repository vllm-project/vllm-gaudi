import itertools
import operator
import os
import math
import ast
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_gaudi.extension.logger import logger as logger
from vllm_gaudi.extension.runtime import get_config

_RANGE_RE = re.compile(r'range\((\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+))?\)')


def _expand_ranges(line: str) -> str:
    """Replace range(start, stop[, step]) with list equivalents for safe parsing."""

    def _replace(m):
        start, stop = int(m.group(1)), int(m.group(2))
        step = int(m.group(3)) if m.group(3) else 1
        return repr(list(range(start, stop, step)))

    return _RANGE_RE.sub(_replace, line)


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
                    bucket = ast.literal_eval(_expand_ranges(line))
                except (ValueError, SyntaxError):
                    logger.warning("Skipping invalid bucketing line")
                    continue

                if not isinstance(bucket, tuple) or len(bucket) != 3:
                    logger.warning("Skipping line due to incorrect format")
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
