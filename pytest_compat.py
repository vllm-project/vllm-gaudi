"""Standalone pytest plugin for HPU compatibility with upstream vLLM tests.

Registered via ``pytest11`` entry-point.  Intentionally placed outside the
``vllm_gaudi`` package so that importing it does **not** trigger
``vllm_gaudi/__init__.py`` (which pulls in ``habana_frameworks.torch`` and
would break pytest on non-HPU machines).
"""

import functools
from unittest.mock import MagicMock, patch

import torch


def pytest_configure(config):
    """Apply HPU compatibility patches after all imports are resolved."""
    _patch_hf3fs_mock_client()


def _patch_hf3fs_mock_client():
    """Guard CUDA sync in the HF3FS mock client on non-CUDA platforms."""
    if torch.cuda.is_available():
        return

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils import (
            hf3fs_mock_client, )
    except ImportError:
        return

    _orig_batch_write = hf3fs_mock_client.Hf3fsClient.batch_write

    @functools.wraps(_orig_batch_write)
    def _safe_batch_write(self, offsets, tensors, event):
        with patch("torch.cuda.current_stream", return_value=MagicMock()):
            return _orig_batch_write(self, offsets, tensors, event)

    hf3fs_mock_client.Hf3fsClient.batch_write = _safe_batch_write
