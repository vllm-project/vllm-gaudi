# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats, )


def test_build_kv_connector_stats_with_none():
    stats = OffloadingConnectorStats.build(None)
    assert stats is None


def test_build_kv_connector_stats_with_empty_dict():
    stats = OffloadingConnectorStats.build({})
    assert stats is None


def test_build_kv_connector_stats_reconstructs_offload_stats():
    original = OffloadingConnectorStats(
        num_cpu_blocks_stored=4,
        num_cpu_blocks_loaded=5,
        num_cpu_blocks_removed=6,
        num_cpu_blocks_all=7,
        num_gpu_blocks_stored=8,
        num_gpu_blocks_loaded=9,
        num_gpu_blocks_preempted=10,
    )
    restored = OffloadingConnectorStats.build(original.to_dict())
    assert restored == original


def test_aggregate_same_connector():
    a = OffloadingConnectorStats(
        num_cpu_blocks_stored=1,
        num_cpu_blocks_loaded=2,
        num_cpu_blocks_removed=3,
        num_cpu_blocks_all=4,
        num_gpu_blocks_stored=5,
        num_gpu_blocks_loaded=6,
        num_gpu_blocks_preempted=7,
    )
    b = OffloadingConnectorStats(
        num_cpu_blocks_stored=10,
        num_cpu_blocks_loaded=20,
        num_cpu_blocks_removed=30,
        num_cpu_blocks_all=40,
        num_gpu_blocks_stored=50,
        num_gpu_blocks_loaded=60,
        num_gpu_blocks_preempted=70,
    )
    result = OffloadingConnectorStats.aggregate([a, b])
    assert result.num_cpu_blocks_stored == 11
    assert result.num_cpu_blocks_loaded == 22
    assert result.num_cpu_blocks_removed == 33
    assert result.num_cpu_blocks_all == 44
    assert result.num_gpu_blocks_stored == 55
    assert result.num_gpu_blocks_loaded == 66
    assert result.num_gpu_blocks_preempted == 77


def test_reduce():
    a = OffloadingConnectorStats(
        num_cpu_blocks_stored=1,
        num_cpu_blocks_loaded=2,
        num_cpu_blocks_removed=3,
        num_cpu_blocks_all=4,
        num_gpu_blocks_stored=5,
        num_gpu_blocks_loaded=6,
        num_gpu_blocks_preempted=7,
    )
    b = OffloadingConnectorStats(
        num_cpu_blocks_stored=10,
        num_cpu_blocks_loaded=20,
        num_cpu_blocks_removed=30,
        num_cpu_blocks_all=40,
        num_gpu_blocks_stored=50,
        num_gpu_blocks_loaded=60,
        num_gpu_blocks_preempted=70,
    )
    result = a.reduce_down(b)
    assert result.num_cpu_blocks_stored == -9
    assert result.num_cpu_blocks_loaded == -18
    assert result.num_cpu_blocks_removed == -27
    assert result.num_cpu_blocks_all == -36
    assert result.num_gpu_blocks_stored == -45
    assert result.num_gpu_blocks_loaded == -54
    assert result.num_gpu_blocks_preempted == -63


def test_reset():
    stats = OffloadingConnectorStats(
        num_cpu_blocks_stored=1,
        num_cpu_blocks_loaded=2,
        num_cpu_blocks_removed=3,
        num_cpu_blocks_all=4,
        num_gpu_blocks_stored=5,
        num_gpu_blocks_loaded=6,
        num_gpu_blocks_preempted=7,
    )
    zero = stats.reset()
    assert zero == OffloadingConnectorStats(
        num_cpu_blocks_stored=0,
        num_cpu_blocks_loaded=0,
        num_cpu_blocks_removed=0,
        num_cpu_blocks_all=0,
        num_gpu_blocks_stored=0,
        num_gpu_blocks_loaded=0,
        num_gpu_blocks_preempted=0,
    )
