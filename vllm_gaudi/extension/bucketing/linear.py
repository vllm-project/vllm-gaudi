import os
import math
from typing import List, Tuple

from vllm_gaudi.extension.logger import logger as logger
from vllm_gaudi.extension.runtime import get_config


class LinearBucketingStrategy:

    def get_prompt_cfgs(self, max_num_prefill_seqs, block_size, max_num_batched_tokens, max_model_len):
        use_merged_prefill = get_config().merged_prefill

        prompt_bs_bucket_cfg = read_bucket_settings('prompt',
                                                    'bs',
                                                    min=1,
                                                    step=2,
                                                    max=max_num_prefill_seqs,
                                                    pad_max=16,
                                                    pad_percent=25)
        prompt_query_bucket_cfg = read_bucket_settings('prompt',
                                                       'query',
                                                       min=block_size,
                                                       step=block_size,
                                                       max=max_num_batched_tokens,
                                                       pad_max=max_num_batched_tokens,
                                                       pad_percent=25)
        max_ctx = math.ceil((max_model_len - prompt_query_bucket_cfg[0]) // block_size)
        prompt_ctx_bucket_cfg = read_bucket_settings('prompt',
                                                     'ctx',
                                                     min=0,
                                                     step=2,
                                                     max=max_ctx,
                                                     pad_max=max_num_batched_tokens // block_size,
                                                     pad_percent=25)

        if use_merged_prefill:
            prev_prompt_bs_bucket_cfg = tuple(prompt_bs_bucket_cfg)
            prev_prompt_query_bucket_cfg = tuple(prompt_query_bucket_cfg)
            prev_prompt_ctx_bucket_cfg = tuple(prompt_ctx_bucket_cfg)

            prompt_bs_bucket_cfg = (1, 1, 1, prev_prompt_bs_bucket_cfg[-2], prev_prompt_bs_bucket_cfg[-1])
            query_min, query_step, _, query_pad_max, query_pad_percent = prev_prompt_query_bucket_cfg
            prompt_query_bucket_cfg = (query_min, query_step * 4, max_num_batched_tokens, query_pad_max,
                                       query_pad_percent)
            prompt_ctx_bucket_cfg = read_bucket_settings('prompt',
                                                         'ctx',
                                                         min=0,
                                                         step=4,
                                                         max=max_ctx * max_num_prefill_seqs,
                                                         pad_max=max_num_batched_tokens // block_size,
                                                         pad_percent=25)

            msg = ('Merged prefill is enabled!\n'
                   'Overriding prompt bucketing settings!\n'
                   f'prompt bs cfg: {prev_prompt_bs_bucket_cfg} -> {prompt_bs_bucket_cfg}\n'
                   f'prompt query cfg: {prev_prompt_query_bucket_cfg} -> {prompt_query_bucket_cfg}\n'
                   f'prompt ctx cfg: {prev_prompt_ctx_bucket_cfg} -> {prompt_ctx_bucket_cfg}\n')
            logger().info(msg)

        msg = ("Prompt bucket config (min, step, max_warmup, pad_max, pad_percent) "
               f"bs:{prompt_bs_bucket_cfg}, "
               f"query:{prompt_query_bucket_cfg}, "
               f"blocks:{prompt_ctx_bucket_cfg}")
        logger().info(msg)

        return prompt_bs_bucket_cfg, prompt_query_bucket_cfg, prompt_ctx_bucket_cfg

    def get_decode_cfgs(self, max_num_seqs, block_size, max_num_batched_tokens, max_model_len, max_blocks):
        contiguous_pa = get_config().use_contiguous_pa

        decode_bs_bucket_cfg = read_bucket_settings('decode',
                                                    'bs',
                                                    min=1,
                                                    step=2,
                                                    max=max_num_seqs,
                                                    pad_max=32,
                                                    pad_percent=25)
        decode_query_bucket_cfg = [1, 1, 1, 1, 1]
        max_decode_blocks = max(math.ceil(max_model_len * max_num_seqs // block_size), block_size)
        if contiguous_pa:
            max_decode_blocks = max_blocks
        decode_block_bucket_cfg = read_bucket_settings('decode',
                                                       'block',
                                                       min=block_size,
                                                       step=block_size,
                                                       max=max_decode_blocks,
                                                       pad_max=max_num_batched_tokens * max_num_seqs // block_size,
                                                       pad_percent=25)
        if decode_block_bucket_cfg[2] > max_blocks:
            logger().info(
                f'VLLM_DECODE_BLOCK_BUCKET_MAX={decode_block_bucket_cfg[2]} is higher than max_blocks={max_blocks}. Your configuration VLLM_DECODE_BLOCK_BUCKET_MAX={decode_block_bucket_cfg[2]} will be overwritten to VLLM_DECODE_BLOCK_BUCKET_MAX={max_blocks}'
            )
            decode_block_bucket_cfg[2] = max_blocks
            if decode_block_bucket_cfg[0] > max_blocks:
                decode_block_bucket_min = max(1, max_blocks - decode_block_bucket_cfg[1])
                logger().info(
                    f'VLLM_DECODE_BLOCK_BUCKET_MIN={decode_block_bucket_cfg[0]} is higher than max_blocks={max_blocks}. Your configuration VLLM_DECODE_BLOCK_BUCKET_MIN={decode_block_bucket_cfg[0]} will be overwritten to VLLM_DECODE_BLOCK_BUCKET_MIN={decode_block_bucket_min}'
                )
                decode_block_bucket_cfg[0] = decode_block_bucket_min

        msg = ("Decode bucket config (min, step, max_warmup, pad_max, pad_percent) "
               f"bs:{decode_bs_bucket_cfg}, "
               f"blocks:{decode_block_bucket_cfg}")
        logger().info(msg)

        return decode_bs_bucket_cfg, decode_query_bucket_cfg, decode_block_bucket_cfg

    def get_range(self, cfg):
        range_for_cfg = warmup_range_with_limits(cfg)
        return sorted(range_for_cfg)


def read_bucket_settings(phase: str, dim: str, **defaults):
    """Read bucketing configuration from env variables.

    phase is either 'prompt' or 'decode'
    dim is either 'bs', 'query' or 'block'
    param is either 'min', 'step' or 'max'
    example env variable: VLLM_DECODE_BS_BUCKET_STEP=128
    """
    params = ['min', 'step', 'max', 'pad_max', 'pad_percent']
    env_vars = [f'VLLM_{phase}_{dim}_BUCKET_{p}'.upper() for p in params]
    default_values = [defaults[p] for p in params]
    values = []

    for p, e, d in zip(params, env_vars, default_values):
        val = os.environ.get(e)

        if val is None and dim == 'query':
            # Check if fallback 'seq' flag is set
            fallback_env = f'VLLM_{phase}_SEQ_BUCKET_{p}'.upper()
            fallback_val = os.environ.get(fallback_env)

            if fallback_val is not None:
                val = fallback_val
                logger().warning(f"{e} not set, using {fallback_env} value ({fallback_val}) instead. "
                                 "This fallback behavior is deprecated and will be removed in v0.12.0.")
        resolved_val = int(val) if val is not None else d
        logger().info(f'{e}={resolved_val} (default:{d})')
        values.append(resolved_val)

    return values


def warmup_range_with_limits(config: Tuple[int, int, int, int, int]) -> List[int]:
    """Generate a warmup range with absolute and relative padding limits.

    1. Starts from `bucket_min` and multiply by 2 (or +1 for 0) till to `bucket_step`.
    2. Add `bucket_step` to the values till to `bucket_max` and choose current bucket if:
        a. the next bucket exceeds the absolute padding limit `pad_max`,
        b. or the next bucket exceeds the padding ratio limit `pad_percent`,
        c. or the current bucket is a multiple of `pad_max`.
    3. Always include `bucket_max` as the last bucket.

    Example:
    1. for config = (0, 8, 64, 64, 0)
        ramp_up = [0, 1, 2, 4, 8]
        stable = [16, 24, 32, 40, 48, 56, 64]
        return [0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    2. for config = (0, 8, 64, 64, 50)
        ramp_up = [0, 1, 2, 4, 8]
        stable = [16, 24, 32, 48, 64]  # 40 and 56 are skipped due to padding ratio limit
        return [0, 1, 2, 4, 8, 16, 24, 32, 48, 64]
    3. for config = (0, 8, 64, 16, 50)
        ramp_up = [0, 1, 2, 4, 8]
        stable = [16, 32, 48, 64]  # 24, 40, 56 are skipped due to absolute padding limit
        return [0, 1, 2, 4, 8, 16, 32, 48, 64]
    4. for config = (16, 16, 128, 32, 25)
        stable = [16, 32, 48, 64, 80, 96, 112, 128]  # no ramp up phase
        return [16, 32, 48, 64, 80, 96, 112, 128]
    """
    bucket_min, bucket_step, bucket_max, pad_max, pad_percent = config
    assert bucket_min <= bucket_max, ("bucket_min cannot be greater than bucket_max. "
                                      "If you want to skip warmup, set VLLM_SKIP_WARMUP=true")
    assert bucket_step > 0, f"bucket_step must be positive, got: ({bucket_step})"
    assert 0 <= pad_percent <= 50, f"pad_percent must be between 0 and 50 percentage points, got: ({pad_percent})"

    buckets = [bucket_min]
    current_bucket = bucket_min
    while current_bucket <= bucket_max:
        last_bucket = buckets[-1]
        if current_bucket <= bucket_step:
            next_bucket = last_bucket * 2
            if next_bucket == 0:
                next_bucket += 1
            if next_bucket <= bucket_max:
                buckets.append(next_bucket)
        else:
            next_bucket = current_bucket + bucket_step
            max_padding = next_bucket - last_bucket - 1
            max_padding_ratio = max_padding / next_bucket
            keep_bucket = (
                max_padding_ratio > pad_percent / 100.0  # next bucket exceeds padding ratio limit
                or max_padding > pad_max  # next bucket exceeds absolute padding limit
                or current_bucket % pad_max == 0  # current bucket is a multiple of pad_max
            )
            if keep_bucket and current_bucket != last_bucket:
                buckets.append(current_bucket)
        current_bucket = next_bucket
    if buckets[-1] != bucket_max:
        buckets.append(bucket_max)

    return buckets
