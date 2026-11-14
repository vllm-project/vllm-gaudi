# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_gaudi.extension.runtime import eager_deferred_sampling_warning

# The tests below pin the closed-form ask that issue #1846 measured
# byte-exactly six times on its reporting deployment
# (ask = 16,384 B x blockpad128(prompt + output - 64), worst case at the cap;
# blockpad128 rounds up to a 128-token multiple) and the advisory's one-sided
# semantics: warn exactly when the cap-class ask exceeds free device memory,
# stay silent otherwise. Silence is not a survival claim -- the report's own
# 1M failure booted with free memory above its cap-class ask and died to
# pre-ask consumption between boot and the first sampling pass, which no
# boot-time comparison can see.

_GIB = 1024**3


def test_warns_on_the_786k_death_receipt(monkeypatch):
    # The 786,432-cap class: ask 12,884,901,888 B (12,288 MiB in the issue's
    # dose-response table). The free value is a probe point inside the
    # report's measured death region for this class -- the report brackets
    # this ask "crosses free device memory between ~10.2 and 12.3 GiB" -- and
    # it also sits below the 12.0 GiB ask, so the comparison is unambiguous.
    warning = eager_deferred_sampling_warning(max_model_len=786432, free_device_bytes=int(11.8 * _GIB))
    assert warning is not None
    assert str(12_884_901_888) in warning
    assert 'issues/1846' in warning


def test_warns_below_the_1m_cap_class_ask():
    # The 1M-cap class: ask 16,382,951,424 B (15.26 GiB). The free value is a
    # below-the-ask probe of the comparison, not a measured headroom: the
    # report's 1M failure (KV util 0.85) booted with free memory ABOVE this
    # ask -- our re-measurement of that boot held 17.9 GiB free against the
    # 15.26 GiB ask -- and died to un-modeled pre-ask consumption, exactly
    # the case where the boot-time advisory stays silent by design.
    warning = eager_deferred_sampling_warning(max_model_len=1_000_000, free_device_bytes=int(6.9 * _GIB))
    assert warning is not None
    assert str(16_382_951_424) in warning


def test_silent_on_the_1m_served_receipt():
    # The same 1M-class request SERVED clean at KV util 0.82
    # (21.46 GiB/worker static margin, per the report): the ask fits, no
    # warning.
    assert eager_deferred_sampling_warning(max_model_len=1_000_000, free_device_bytes=int(21.46 * _GIB)) is None


def test_silent_on_the_262k_passed_receipt():
    # A smaller cap stays silent whenever its cap-class ask fits: at
    # max_model_len 262,144 the closed-form cap-class ask is 4,294,967,296 B
    # (4 GiB). The report's passing ~4,103 MiB class -- 4,303,355,904 B, the
    # ask a 262,144+512-token request pays -- was served on an mml 1M boot,
    # where the 1M cap-class ask dominates it; it fits this free point a
    # fortiori.
    assert eager_deferred_sampling_warning(max_model_len=262_144, free_device_bytes=int(21.46 * _GIB)) is None


def test_silent_when_headroom_covers_the_ask():
    # A free point above both the report's measured death region for the
    # 786,432 cap (~10.2-12.3 GiB) and the 12.0 GiB cap-class ask: past the
    # free-memory cliff the report shows the class serving; the warning must
    # not fire.
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
