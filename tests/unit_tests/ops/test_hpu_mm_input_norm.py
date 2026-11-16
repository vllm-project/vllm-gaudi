# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest
import torch
from torchvision.transforms.v2 import functional as tvf

from vllm.config import VllmConfig, set_current_vllm_config
from vllm_gaudi.ops.hpu_mm_input_norm import HPUFusedMMInputNorm

PARAMETERS = [
    ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 1 / 255),
    ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711], 1 / 255),
    ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0),
]


class TestHPUMMInputNorm(unittest.TestCase):

    def _check_processor(self, execution):
        for mean, std, scale in PARAMETERS:
            for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                with self.subTest(mean=mean, dtype=dtype, execution=execution):
                    device = "cpu" if execution == "cpu" else "hpu"
                    # Cover every byte value in every channel, with distinct patterns.
                    pixels = torch.stack([torch.arange(256).roll(c * 17) for c in range(3)]).to(torch.uint8)
                    pixels = pixels.reshape(3, 16, 16)
                    fused_mean = torch.tensor(mean, dtype=torch.float32) * (1.0 / scale)
                    fused_std = torch.tensor(std, dtype=torch.float32) * (1.0 / scale)
                    expected = tvf.normalize(pixels.float(), fused_mean, fused_std).to(dtype)
                    raw = pixels.reshape(1, -1).to(device)
                    with set_current_vllm_config(VllmConfig()):
                        op = HPUFusedMMInputNorm(mean, std, scale).to(device)
                        forward = op.forward_oot
                        if execution == "hpu_compile":
                            torch._dynamo.reset()
                            forward = torch.compile(forward, backend="hpu_backend", fullgraph=True)
                        actual = forward(raw, dtype).cpu().reshape_as(expected)
                    self.assertEqual(actual.dtype, dtype)
                    # Allow a small relative error for HPU FP32 division.
                    # Keep emitted BF16/FP16 values strictly identical.
                    rtol = 2 * torch.finfo(torch.float32).eps if dtype == torch.float32 and device == "hpu" else 0
                    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0)

    def test_uint8_cpu_matches_processor(self):
        self._check_processor("cpu")

    @unittest.skipUnless(torch.hpu.is_available(), "HPU required")
    def test_uint8_hpu_matches_processor(self):
        self._check_processor("hpu")

    @unittest.skipUnless(torch.hpu.is_available(), "HPU required")
    def test_uint8_compiled_hpu_matches_processor(self):
        self._check_processor("hpu_compile")

    def test_bf16_rounding_near_mean(self):
        with set_current_vllm_config(VllmConfig()):
            op = HPUFusedMMInputNorm([0.5] * 3, [0.5] * 3, 1 / 255)
            pixels = torch.full((1, 3), 127, dtype=torch.uint8)
            actual = op.forward_oot(pixels, torch.bfloat16)
            expected = torch.tensor(-1 / 255).bfloat16()
            self.assertTrue(torch.all(actual == expected))
            self.assertFalse(torch.equal(actual, op.forward_native(pixels, torch.bfloat16)))

    def test_non_uint8_keeps_native_contract(self):
        with set_current_vllm_config(VllmConfig()):
            op = HPUFusedMMInputNorm([0.5] * 3, [0.5] * 3, 1 / 255)
            for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                with self.subTest(dtype=dtype):
                    pixels = torch.arange(18, dtype=dtype).reshape(2, 9)
                    torch.testing.assert_close(op.forward_oot(pixels, dtype),
                                               op.forward_native(pixels, dtype),
                                               rtol=0,
                                               atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
