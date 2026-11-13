# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the vendored V1 tests. Lives outside upstream/, which vendor_v1.py regenerates."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_default_device():
    """Restore the default device: the vendored sampler tests set it to HPU, which would leak into later tests."""
    # get_default_device() reports cpu when none is set; restore "unset" as None so no device context stays on.
    unset = getattr(getattr(torch, "_GLOBAL_DEVICE_CONTEXT", None), "device_context", None) is None
    device = None if unset else torch.get_default_device()
    try:
        yield
    finally:
        torch.set_default_device(device)
