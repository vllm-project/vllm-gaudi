# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from typing import Optional
from itertools import cycle
from unittest.mock import patch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm.utils.torch_utils import set_random_seed
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available

from vllm_gaudi.v1.worker.hpu_input_batch import InputBatch, CachedRequestState

VOCAB_SIZE = 1024
MAX_PROMPT_SIZE = 100
NUM_OUTPUT_TOKENS = 1
DEVICE = current_platform.device_type
SEED = 42


def _create_sampling_params(temperature: float = 0,
                            top_k: int = -1,
                            top_p: float = 1,
                            min_p: float = 0,
                            presence_penalty: float = 0,
                            repetition_penalty: float = 1,
                            frequency_penalty: float = 0,
                            min_tokens: int = 0,
                            seed: Optional[int] = None) -> SamplingParams:
    '''Create sampling parameters for text generation.
    Refer to: 
    https://docs.vllm.ai/en/stable/api/vllm/sampling_params.html#vllm.sampling_params.SamplingParams 
    for params'''
    return SamplingParams(temperature=temperature,
                          top_k=top_k,
                          top_p=top_p,
                          min_p=min_p,
                          presence_penalty=presence_penalty,
                          repetition_penalty=repetition_penalty,
                          frequency_penalty=frequency_penalty,
                          min_tokens=min_tokens,
                          seed=seed)


def _construct_cached_request_state(req_id_suffix: int,
                                    sampling_params: SamplingParams,
                                    generator: Optional[torch.Generator] = None) -> CachedRequestState:
    prompt_token_ids = [np.random.randint(0, VOCAB_SIZE) for _ in range(np.random.randint(0, MAX_PROMPT_SIZE))]
    output_token_ids = [np.random.randint(0, VOCAB_SIZE) for _ in range(np.random.randint(0, NUM_OUTPUT_TOKENS))]
    return CachedRequestState(
        req_id=f"req_id_{req_id_suffix}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=[],
        block_ids=([], ),
        generator=generator,
        num_computed_tokens=len(output_token_ids),
        output_token_ids=output_token_ids,
    )


def _create_logits(batch_size: int, init_value: float = 1e-2) -> torch.Tensor:
    logits = torch.full((batch_size, VOCAB_SIZE), init_value, device=DEVICE, dtype=torch.bfloat16)
    for i in range(batch_size):
        max_token_id = i % VOCAB_SIZE  # Different max token for each batch
        logits[i, max_token_id] = 1e2  # Clear maximum
    return logits


def _prepare_metadata(batch_size: int,
                      sampling_params: SamplingParams,
                      is_seeded_random: bool = False) -> SamplingMetadata:
    input_batch: InputBatch = InputBatch(
        max_num_reqs=batch_size,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        device=torch.device(DEVICE),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    if is_seeded_random:
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(SEED)
    else:
        generator = None

    # Add requests
    for req_index in range(batch_size):
        req: CachedRequestState = _construct_cached_request_state(req_index, sampling_params, generator)
        input_batch.add_request(req)

    # Generate the sampling metadata
    sampling_metadata = input_batch._make_sampling_metadata()

    return sampling_metadata


def _create_offset_logits(batch_size: int) -> torch.tensor:
    logits = []
    cycle_input = [i * 0.1 for i in range(VOCAB_SIZE)]
    cycle_iter = cycle(cycle_input)
    # creates logits list arranged like:
    # 0 1 2 ... VOCAB_Size - 1
    # 1 2 ... 0 etc
    for _ in range(batch_size):
        tmp = []
        for _ in range(VOCAB_SIZE):
            tmp.append(next(cycle_iter))
        logits.append(tmp)
        _ = next(cycle_iter)

    return torch.tensor(logits, device=DEVICE, dtype=torch.bfloat16)


@pytest.mark.parametrize("batch_size", [1, 32])
def test_sampler_greedy(batch_size: int) -> None:
    logits = _create_logits(batch_size)
    sampler = Sampler()
    sampling_params = _create_sampling_params(temperature=0)
    sampling_metadata = _prepare_metadata(batch_size, sampling_params)

    # Expected output: argmax of each logits row
    expected_tokens = torch.argmax(logits, dim=-1)

    # Perform sampling
    sampler_output = sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    ).sampled_token_ids.flatten()

    assert torch.equal(sampler_output, expected_tokens)


@pytest.mark.parametrize("batch_size", [1, 32])
def test_sampler_random(batch_size: int) -> None:
    set_random_seed(SEED)
    logits = _create_logits(batch_size)
    sampler = Sampler()
    sampling_params = _create_sampling_params(temperature=1.0, seed=SEED)
    sampling_metadata = _prepare_metadata(batch_size, sampling_params)

    # Final probs are [100%, 0%, 0%, ..., 0%] so we can use argmax
    expected_tokens = torch.argmax(logits, dim=-1)

    # Perform sampling
    sampler_output = sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    ).sampled_token_ids.flatten()

    assert torch.equal(sampler_output, expected_tokens)


@pytest.mark.parametrize("batch_size", [1, 32])
def test_sampler_random_seeded(batch_size: int) -> None:
    set_random_seed(SEED)
    # NOTE I don't believe def random_sample( is fully deterministic
    # due to q[i].exponential_( generating random numbers although
    # generator is set. If init_value is set closer to 1e2 this test will fail
    logits = _create_logits(batch_size, init_value=1e-2)
    sampler = Sampler()
    sampling_params = _create_sampling_params(temperature=1.0, seed=SEED)
    sampling_metadata = _prepare_metadata(batch_size, sampling_params, is_seeded_random=True)

    # Perform sampling
    sampler_output_first = sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    ).sampled_token_ids.flatten()

    sampler_output_second = sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    ).sampled_token_ids.flatten()

    # check seeded generation
    assert torch.equal(sampler_output_first, sampler_output_second)


# beware of using higher BS, it will multiply the time of the test
# and CI will run on G2 so it will be even slower in actual run (~4x)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("top_k", [-1, 1, 2])
@pytest.mark.parametrize("top_p", [0.1, 1])
@pytest.mark.parametrize("min_p", [0, 0.1])
def test_sampler_top_p_top_k_min_p(batch_size: int, top_k: int, top_p: float, min_p: float) -> None:
    set_random_seed(SEED)
    logits = _create_offset_logits(batch_size)
    sampler = Sampler()
    sampling_params = _create_sampling_params(temperature=1.0, seed=SEED, top_k=top_k, top_p=top_p, min_p=min_p)
    sampling_metadata = _prepare_metadata(batch_size, sampling_params)

    sample_probs = None

    def _mock_random_sample(
        probs: torch.Tensor,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor:
        nonlocal sample_probs
        sample_probs = probs
        q = torch.empty_like(probs)

        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
        return probs.div_(q).argmax(dim=-1).view(-1)

    with patch("vllm.v1.sample.ops.topk_topp_sampler.random_sample", _mock_random_sample):
        sampler_output = sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

    sampled_ids = sampler_output.sampled_token_ids.flatten()
    idx_of_nonzero = torch.nonzero(sample_probs, as_tuple=False)
    no_of_nonzero_probs_to_sample = torch.count_nonzero(sample_probs, dim=None).item()

    # only in top_k we know how many samples we can expect
    if top_p == 1 and min_p == 0:
        assert top_k*batch_size <= no_of_nonzero_probs_to_sample, \
            f'''Expected at least {top_k*batch_size} non-zero probabilities, 
                got {no_of_nonzero_probs_to_sample}'''

    # Change [[0, 1023], [0, 1024], [1, 1022], ...]
    # to a [[1023, 1024], [1022, 1023], ...]
    # to compare with expected result for each sample
    expected_nonzero_idx = [[] for _ in range(batch_size)]  # type: list[list[int]]
    for prompt_no, idx in idx_of_nonzero.to("cpu").tolist():
        expected_nonzero_idx[prompt_no].append(idx)

    for i in range(len(expected_nonzero_idx)):
        assert sampled_ids[i] in expected_nonzero_idx[i], \
            f'''Expected sampled token ids to be in {expected_nonzero_idx[i]}, 
                got {sampled_ids[i]}'''
