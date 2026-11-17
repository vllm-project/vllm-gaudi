# SPDX-License-Identifier: Apache-2.0

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# Mock habana_frameworks before any vllm_gaudi imports
if "habana_frameworks" not in sys.modules:
    _hf = MagicMock()
    _hf.torch.utils.internal.is_lazy.return_value = False
    sys.modules["habana_frameworks"] = _hf
    sys.modules["habana_frameworks.torch"] = _hf.torch
    sys.modules["habana_frameworks.torch.utils"] = _hf.torch.utils
    sys.modules["habana_frameworks.torch.utils.internal"] = _hf.torch.utils.internal


# ── ops_selector ─────────────────────────────────────────────────────────


class TestOpsSelectorRuntime:

    def test_use_pytorch_runtime_default(self):
        """Default (unset) returns False."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_MAMBA_USE_PYTORCH", None)
            from vllm_gaudi.ops.ops_selector import _use_pytorch_runtime
            assert _use_pytorch_runtime() is False

    def test_use_pytorch_runtime_enabled(self):
        with patch.dict(os.environ, {"VLLM_MAMBA_USE_PYTORCH": "1"}):
            from vllm_gaudi.ops.ops_selector import _use_pytorch_runtime
            assert _use_pytorch_runtime() is True

    def test_use_pytorch_runtime_disabled(self):
        with patch.dict(os.environ, {"VLLM_MAMBA_USE_PYTORCH": "0"}):
            from vllm_gaudi.ops.ops_selector import _use_pytorch_runtime
            assert _use_pytorch_runtime() is False


class TestWrapSelectiveStateUpdateRef:

    def test_simple_path_no_indices(self):
        """When no batch indices, calls the inner fn directly."""
        from vllm_gaudi.ops.ops_selector import _wrap_selective_state_update_ref

        mock_fn = MagicMock(return_value="result")
        wrapped = _wrap_selective_state_update_ref(mock_fn)

        result = wrapped("state", "x", "dt", "A", "B", "C", D="D", z="z")
        assert result == "result"
        mock_fn.assert_called_once_with(
            "state", "x", "dt", "A", "B", "C",
            D="D", z="z", dt_bias=None, dt_softplus=False
        )

    def test_simple_path_with_out(self):
        """When out is provided, copies result into it."""
        from vllm_gaudi.ops.ops_selector import _wrap_selective_state_update_ref

        mock_fn = MagicMock(return_value="result")
        out = MagicMock()
        wrapped = _wrap_selective_state_update_ref(mock_fn)

        result = wrapped("state", "x", "dt", "A", "B", "C", out=out)
        out.copy_.assert_called_once_with("result")
        assert result is out


# ── platform helpers ─────────────────────────────────────────────────────


class TestRetainEnvs:

    def test_hpu_vars_retained(self):
        from vllm_gaudi.platform import retain_envs
        assert retain_envs("HPU_VISIBLE_DEVICES") is True
        assert retain_envs("MY_HPU_CONFIG") is True

    def test_ray_vars_retained(self):
        from vllm_gaudi.platform import retain_envs
        assert retain_envs("RAY_ADDRESS") is True

    def test_vllm_vars_retained(self):
        from vllm_gaudi.platform import retain_envs
        assert retain_envs("VLLM_USE_FAKE_HPU") is True

    def test_specific_allowed_vars(self):
        from vllm_gaudi.platform import retain_envs
        assert retain_envs("GLOO_SOCKET_IFNAME") is True
        assert retain_envs("HCCL_SOCKET_IFNAME") is True
        assert retain_envs("NCCL_SOCKET_IFNAME") is True

    def test_unrelated_var_not_retained(self):
        from vllm_gaudi.platform import retain_envs
        assert retain_envs("HOME") is False
        assert retain_envs("PATH") is False
        assert retain_envs("LD_LIBRARY_PATH") is False


class TestHpuPlatformPureMethods:

    def test_is_async_output_supported(self):
        from vllm_gaudi.platform import HpuPlatform
        assert HpuPlatform.is_async_output_supported(enforce_eager=True) is True
        assert HpuPlatform.is_async_output_supported(enforce_eager=False) is True

    def test_get_device_name(self):
        from vllm_gaudi.platform import HpuPlatform
        assert HpuPlatform.get_device_name() == "hpu"
        assert HpuPlatform.get_device_name(device_id=1) == "hpu"

    def test_is_pin_memory_available(self):
        from vllm_gaudi.platform import HpuPlatform
        assert HpuPlatform.is_pin_memory_available() is False

    def test_get_punica_wrapper(self):
        from vllm_gaudi.platform import HpuPlatform
        result = HpuPlatform.get_punica_wrapper()
        assert "PunicaWrapperHPU" in result

    def test_support_hybrid_kv_cache(self):
        from vllm_gaudi.platform import HpuPlatform
        assert HpuPlatform.support_hybrid_kv_cache() is True

    def test_get_device_communicator_cls(self):
        from vllm_gaudi.platform import HpuPlatform
        result = HpuPlatform.get_device_communicator_cls()
        assert "HpuCommunicator" in result

    def test_supports_structured_output(self):
        from vllm_gaudi.platform import HpuPlatform
        assert HpuPlatform.supports_structured_output() is True

    def test_supports_v1(self):
        from vllm_gaudi.platform import HpuPlatform
        assert HpuPlatform.supports_v1(model_config=None) is True

    def test_get_nixl_supported_devices(self):
        from vllm_gaudi.platform import HpuPlatform
        result = HpuPlatform.get_nixl_supported_devices()
        assert "hpu" in result
        assert "cpu" in result["hpu"]
        assert "hpu" in result["hpu"]

    def test_is_kv_cache_dtype_supported(self):
        from vllm_gaudi.platform import HpuPlatform
        assert HpuPlatform.is_kv_cache_dtype_supported("fp8_inc", None) is True
        assert HpuPlatform.is_kv_cache_dtype_supported("fp8", None) is False
        assert HpuPlatform.is_kv_cache_dtype_supported("auto", None) is False

    def test_get_device_total_memory(self):
        from vllm_gaudi.platform import HpuPlatform
        assert HpuPlatform.get_device_total_memory() == 0

    def test_get_nixl_memory_type_default(self):
        from vllm_gaudi.platform import HpuPlatform
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_NIXL_DEVICE_TO_DEVICE", None)
            assert HpuPlatform.get_nixl_memory_type() == "DRAM"

    def test_get_nixl_memory_type_device_to_device(self):
        from vllm_gaudi.platform import HpuPlatform
        with patch.dict(os.environ, {"VLLM_NIXL_DEVICE_TO_DEVICE": "1"}):
            assert HpuPlatform.get_nixl_memory_type() == "VRAM"

    def test_get_nixl_memory_type_disabled(self):
        from vllm_gaudi.platform import HpuPlatform
        with patch.dict(os.environ, {"VLLM_NIXL_DEVICE_TO_DEVICE": "0"}):
            assert HpuPlatform.get_nixl_memory_type() == "DRAM"

    def test_use_sync_weight_loader_default(self):
        from vllm_gaudi.platform import HpuPlatform
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_WEIGHT_LOAD_FORCE_SYNC", None)
            assert HpuPlatform.use_sync_weight_loader() is True

    def test_use_sync_weight_loader_true(self):
        from vllm_gaudi.platform import HpuPlatform
        with patch.dict(os.environ, {"VLLM_WEIGHT_LOAD_FORCE_SYNC": "true"}):
            assert HpuPlatform.use_sync_weight_loader() is True

    def test_use_sync_weight_loader_one(self):
        from vllm_gaudi.platform import HpuPlatform
        with patch.dict(os.environ, {"VLLM_WEIGHT_LOAD_FORCE_SYNC": "1"}):
            assert HpuPlatform.use_sync_weight_loader() is True

    def test_use_sync_weight_loader_false(self):
        from vllm_gaudi.platform import HpuPlatform
        with patch.dict(os.environ, {"VLLM_WEIGHT_LOAD_FORCE_SYNC": "false"}):
            assert HpuPlatform.use_sync_weight_loader() is False

    def test_use_sync_weight_loader_zero(self):
        from vllm_gaudi.platform import HpuPlatform
        with patch.dict(os.environ, {"VLLM_WEIGHT_LOAD_FORCE_SYNC": "0"}):
            assert HpuPlatform.use_sync_weight_loader() is False


class TestHpuPlatformAttnBackend:

    def test_sparse_raises(self):
        from vllm_gaudi.platform import HpuPlatform
        config = MagicMock()
        config.use_sparse = True
        with pytest.raises(NotImplementedError, match="Sparse Attention"):
            HpuPlatform.get_attn_backend_cls(None, config)

    def test_unified_attn_non_mla(self):
        from vllm_gaudi.platform import HpuPlatform
        config = MagicMock()
        config.use_sparse = False
        config.use_mla = False
        with patch("vllm_gaudi.platform.get_config") as mock_cfg:
            mock_cfg.return_value.unified_attn = True
            result = HpuPlatform.get_attn_backend_cls(None, config)
            assert "HPUUnifiedAttentionBackend" in result

    def test_unified_attn_mla(self):
        from vllm_gaudi.platform import HpuPlatform
        config = MagicMock()
        config.use_sparse = False
        config.use_mla = True
        with patch("vllm_gaudi.platform.get_config") as mock_cfg:
            mock_cfg.return_value.unified_attn = True
            result = HpuPlatform.get_attn_backend_cls(None, config)
            assert "HPUUnifiedMLABackend" in result

    def test_non_unified_non_mla(self):
        from vllm_gaudi.platform import HpuPlatform
        config = MagicMock()
        config.use_sparse = False
        config.use_mla = False
        with patch("vllm_gaudi.platform.get_config") as mock_cfg:
            mock_cfg.return_value.unified_attn = False
            result = HpuPlatform.get_attn_backend_cls(None, config)
            assert "HPUAttentionBackendV1" in result

    def test_non_unified_mla(self):
        from vllm_gaudi.platform import HpuPlatform
        config = MagicMock()
        config.use_sparse = False
        config.use_mla = True
        with patch("vllm_gaudi.platform.get_config") as mock_cfg:
            mock_cfg.return_value.unified_attn = False
            result = HpuPlatform.get_attn_backend_cls(None, config)
            assert "HPUMLAAttentionBackend" in result
