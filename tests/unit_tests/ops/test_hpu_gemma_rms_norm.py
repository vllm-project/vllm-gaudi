# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import unittest
from unittest.mock import patch
import torch

from vllm.ir import ops as ir_ops
from vllm_gaudi.extension import kernels
from vllm_gaudi.ops.hpu_layernorm import HPUGemmaRMSNorm


class _ReferenceHPUKernel:
    @staticmethod
    def apply(x, weight, epsilon):
        # Model the fused kernels accumulation and output dtype on CPU.
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)
        return (value * weight.float()).to(torch.promote_types(x.dtype, weight.dtype))


class TestHPUGemmaRMSNorm(unittest.TestCase):
    def test_matches_upstream_precision(self):
        with patch.object(kernels, "rms_norm", return_value=_ReferenceHPUKernel):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                for use_residual in (False, True):
                    with self.subTest(dtype=dtype, residual=use_residual):
                        generator = torch.Generator().manual_seed(413)
                        x = torch.randn(2, 3, 256, generator=generator).to(dtype)
                        residual = torch.randn(x.shape, generator=generator).to(dtype) if use_residual else None
                        layer = SimpleNamespace(
                            weight=(torch.randn(256, generator=generator) * 0.3).to(dtype),
                            variance_epsilon=1e-6,
                        )
                        weight = layer.weight.float() + 1.0
                        if use_residual:
                            expected = ir_ops.fused_add_rms_norm.impls["native"].impl_fn(
                                x, residual, weight, layer.variance_epsilon)
                        else:
                            expected = ir_ops.rms_norm.impls["native"].impl_fn(x, weight, layer.variance_epsilon)
                        actual = HPUGemmaRMSNorm.forward_oot(layer, x, residual)
                        if use_residual:
                            for result, reference in zip(actual, expected):
                                self.assertEqual(result.dtype, dtype)
                                torch.testing.assert_close(result, reference, rtol=0, atol=0)
                        else:
                            self.assertEqual(actual.dtype, dtype)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
