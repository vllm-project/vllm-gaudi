# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``OffloadingConnectorStats``.

Upstream vLLM PR #35669 ("Feature/offloading manager stats") rewrote
``OffloadingConnectorStats`` from a per-direction
``{"CPU_to_GPU": [{op_size, op_time}], "GPU_to_CPU": [...]}`` list payload to a
self-describing flat structure keyed by ``{"types": {...}, "data": {...}}`` with
flat prometheus metric names.

Upstream vLLM PR #45957 ("[KV Offloading] Add labeled metrics support") then
nested every per-metric value under a label-values tuple, so each ``data``
entry is now ``{labelvalues: value}`` instead of a bare value. Unlabeled
metrics use ``()`` as their label-values tuple. These tests track the new
contract so the HPU CPU-offloading connector keeps verifying against the live
upstream API.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
    OffloadingConnectorStats,
)

# Flat metric names emitted by the upstream connector. Kept as module-level
# constants so the tests read against names instead of magic strings.
LOAD_BYTES = "vllm:kv_offload_load_bytes"
LOAD_TIME = "vllm:kv_offload_load_time"
LOAD_SIZE = "vllm:kv_offload_load_size"
STORE_BYTES = "vllm:kv_offload_store_bytes"
STORE_SIZE = "vllm:kv_offload_store_size"


def _make_populated_stats() -> OffloadingConnectorStats:
    """Build a stats object exercising counter, gauge, and histogram metrics."""
    stats = OffloadingConnectorStats()
    stats.increase_counter(LOAD_BYTES, 16)
    stats.increase_counter(LOAD_BYTES, 8)
    stats.increase_counter(LOAD_TIME, 1.5)
    stats.observe_histogram(LOAD_SIZE, 16)
    stats.observe_histogram(LOAD_SIZE, 8)
    return stats


def test_build_kv_connector_stats_with_none():
    """``build_kv_connector_stats(None)`` returns an empty, self-describing stats."""
    stats = OffloadingConnector.build_kv_connector_stats(data=None)

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    # The flat payload always carries the "types"/"data" envelope, and is
    # considered empty while the inner "data" map has no observations.
    assert stats.is_empty()
    assert stats.data["data"] == {}
    assert stats.data["types"] == {}


def test_build_kv_connector_stats_with_empty_dict():
    """An explicit empty dict is normalized to the self-describing envelope."""
    stats = OffloadingConnector.build_kv_connector_stats(data={})

    assert stats is not None
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.is_empty()
    assert stats.data["data"] == {}
    assert stats.data["types"] == {}


def test_build_kv_connector_stats_reconstructs_offload_stats():
    """A serialized flat payload round-trips back into an equivalent stats."""
    serialized_data = {
        "types": {
            LOAD_BYTES: "counter",
            LOAD_SIZE: "histogram",
        },
        "data": {
            LOAD_BYTES: {
                (): 24
            },
            LOAD_SIZE: {
                (): [16, 8]
            },
        },
    }

    stats = OffloadingConnector.build_kv_connector_stats(data=serialized_data)

    assert isinstance(stats, OffloadingConnectorStats)
    assert not stats.is_empty()
    assert stats.data["data"][LOAD_BYTES] == {(): 24}
    assert stats.data["data"][LOAD_SIZE] == {(): [16, 8]}
    assert stats.data["types"][LOAD_BYTES] == "counter"
    assert stats.data["types"][LOAD_SIZE] == "histogram"


def test_aggregate_same_connector():
    """Aggregating sums counters and concatenates histogram observations."""
    stats1 = _make_populated_stats()

    stats2 = OffloadingConnectorStats()
    stats2.increase_counter(LOAD_BYTES, 10)
    stats2.observe_histogram(LOAD_SIZE, 3)
    stats2.increase_counter(STORE_BYTES, 16)

    result = stats1.aggregate(stats2)

    assert result is stats1  # aggregate mutates and returns self
    # Counters accumulate across both stats, nested under the unlabeled tuple.
    assert result.data["data"][LOAD_BYTES] == {(): 34}
    assert result.data["data"][STORE_BYTES] == {(): 16}
    # Histogram observations are concatenated in order.
    assert result.data["data"][LOAD_SIZE] == {(): [16, 8, 3]}


def test_aggregate_empty_other_is_noop():
    """Aggregating an empty stats leaves the receiver unchanged."""
    stats = _make_populated_stats()
    empty = OffloadingConnectorStats()

    result = stats.aggregate(empty)

    assert result is stats
    assert result.data["data"][LOAD_BYTES] == {(): 24}
    assert result.data["data"][LOAD_SIZE] == {(): [16, 8]}


def test_reduce():
    """``reduce()`` flattens counters as-is and histograms to count/sum pairs."""
    stats = OffloadingConnectorStats()
    stats.increase_counter(LOAD_BYTES, 34)
    stats.increase_counter(LOAD_TIME, 2.6)
    stats.increase_counter(STORE_BYTES, 19)
    stats.observe_histogram(LOAD_SIZE, 16)
    stats.observe_histogram(LOAD_SIZE, 8)

    reduced = stats.reduce()

    assert isinstance(reduced, dict)
    # Counters pass through unchanged under their flat metric name.
    assert reduced[LOAD_BYTES] == 34
    assert reduced[LOAD_TIME] == 2.6
    assert reduced[STORE_BYTES] == 19
    # Histograms reduce to a count and a sum suffix.
    assert reduced[f"{LOAD_SIZE}_count"] == 2
    assert reduced[f"{LOAD_SIZE}_sum"] == 24


def test_reset():
    """``reset()`` clears all observations back to the empty envelope."""
    stats = _make_populated_stats()

    assert not stats.is_empty()

    stats.reset()

    assert stats.is_empty()
    assert stats.data["data"] == {}
    assert stats.data["types"] == {}
