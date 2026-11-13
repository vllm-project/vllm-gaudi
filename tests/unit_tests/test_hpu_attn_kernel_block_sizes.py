# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from vllm.v1.worker.utils import select_common_block_size

import vllm_gaudi.v1.attention.backends.hpu_attn as hpu_attn
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionBackendV1

ALL_SIZES = [16, 128, 528, 768]


def _set_prefer_128(monkeypatch, enabled):
    monkeypatch.setattr(hpu_attn, "get_config", lambda: SimpleNamespace(prefer_128_kernel_blocks=enabled))


@pytest.mark.parametrize("block_size", [128, 2176, 4224, 8448])
def test_prefer_128_keeps_128_when_it_divides_the_block(monkeypatch, block_size):
    _set_prefer_128(monkeypatch, True)
    spec = SimpleNamespace(block_size=block_size)
    assert HPUAttentionBackendV1.get_supported_kernel_block_sizes(spec) == [128]
    assert select_common_block_size(block_size, [HPUAttentionBackendV1], [spec]) == 128


@pytest.mark.parametrize("block_size, selected", [(528, 528), (1056, 528), (16, 16)])
def test_prefer_128_keeps_the_other_sizes_when_128_does_not_divide(monkeypatch, block_size, selected):
    _set_prefer_128(monkeypatch, True)
    spec = SimpleNamespace(block_size=block_size)
    assert HPUAttentionBackendV1.get_supported_kernel_block_sizes(spec) == ALL_SIZES
    assert select_common_block_size(block_size, [HPUAttentionBackendV1], [spec]) == selected


@pytest.mark.parametrize("block_size, selected", [(4224, 528), (768, 768), (2176, 128)])
def test_without_prefer_128_the_largest_dividing_size_wins(monkeypatch, block_size, selected):
    _set_prefer_128(monkeypatch, False)
    spec = SimpleNamespace(block_size=block_size)
    assert HPUAttentionBackendV1.get_supported_kernel_block_sizes(spec) == ALL_SIZES
    assert select_common_block_size(block_size, [HPUAttentionBackendV1], [spec]) == selected


def test_no_spec_lists_every_size_without_reading_the_config(monkeypatch):

    def no_config():
        raise AssertionError("get_config must not be read without a KV cache spec")

    monkeypatch.setattr(hpu_attn, "get_config", no_config)
    assert HPUAttentionBackendV1.get_supported_kernel_block_sizes() == ALL_SIZES
