# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gaudi.extension.runtime import eager_deferred_sampling_warning

# The receipt table below pins the closed-form ask that issue #1846 measured
# byte-exactly six times on its reporting deployment
# (ask = 16,384 B x blockpad128(prompt + output - 64), worst case at the cap;
# blockpad128 rounds up to a 128-token multiple). The same constant must
# separate every observed PASSED row from every observed FAILED row of that
# report: warn only when the worst-case ask exceeds the free device memory.

_GIB = 1024**3


def test_warns_on_the_786k_death_receipt(monkeypatch):
    # FAILED x2 at mml 786,432, ask 12,884,901,888 B, free ~11.8 GiB class.
    warning = eager_deferred_sampling_warning(max_model_len=786432, free_device_bytes=int(11.8 * _GIB))
    assert warning is not None
    assert str(12_884_901_888) in warning
    assert 'issues/1846' in warning


def test_warns_on_the_1m_low_headroom_death_receipt():
    # FAILED at mml 1,000,000 with KV util 0.85; ask 16,382,951,424 B.
    warning = eager_deferred_sampling_warning(max_model_len=1_000_000, free_device_bytes=int(6.9 * _GIB))
    assert warning is not None
    assert str(16_382_951_424) in warning


def test_silent_on_the_1m_served_receipt():
    # The same 1M-class request SERVED clean at KV util 0.82
    # (21.46 GiB/worker static margin): the ask fits, no warning.
    assert eager_deferred_sampling_warning(max_model_len=1_000_000, free_device_bytes=int(21.46 * _GIB)) is None


def test_silent_on_the_262k_passed_receipt():
    # PASSED x2 at the 262,144+512 class: ask 4,295,817,216 B fits.
    assert eager_deferred_sampling_warning(max_model_len=262_144, free_device_bytes=int(21.46 * _GIB)) is None


def test_silent_when_headroom_covers_the_ask():
    # The dose-response bracket upper edge: past the free-memory cliff the
    # configuration just serves; the warning must not fire.
    assert eager_deferred_sampling_warning(max_model_len=786_432, free_device_bytes=int(13.9 * _GIB)) is None


def test_short_context_never_warns():
    assert eager_deferred_sampling_warning(max_model_len=4096, free_device_bytes=int(1 * _GIB)) is None
    assert eager_deferred_sampling_warning(max_model_len=64, free_device_bytes=0) is None


def test_warning_names_the_remedy_and_the_knobs():
    warning = eager_deferred_sampling_warning(max_model_len=1_000_000, free_device_bytes=1)
    assert 'VLLM_BUCKETING_STRATEGY=pad' in warning
    assert 'VLLM_HPU_FSDPA_SLICE_ENABLED=0' in warning
    assert 'issues/1854' in warning
    assert 'max-model-len' in warning
