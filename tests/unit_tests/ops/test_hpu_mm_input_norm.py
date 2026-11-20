# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

import pytest
import torch
from torchvision.transforms.v2 import functional as tvf
from vllm.config import VllmConfig, set_current_vllm_config
import habana_frameworks.torch as htorch
from utils import temporary_op_registry_oot, register_op
from vllm_gaudi.ops.hpu_mm_input_norm import HPUFusedMMInputNorm
from vllm_gaudi.utils import HPUCompileConfig
from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm
from vllm.platforms import current_platform

# CLIP mean/std: distinct per channel, so a channel mix-up changes the result.
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]
RESCALE = 1 / 255
DEVICE = [current_platform.device_type]

# (layout, input shape): Gemma 4 channel-last rows, with a pixel count that is a
# multiple of 256 and one that is not; other narrow rows; Mistral 3 / Kimi-K2.5
# (patches, C*14*14); Qwen-style (patches, C*2*14*14); NCHW images.
SHAPES = [
    ("pcl", (2 * 280 * 256, 3)),
    ("pcl", (7 * 17, 3)),
    ("pcl", (64, 3 * 2)),
    ("pcl", (64, 3 * 4)),
    ("pcl", (64, 3 * 14 * 14)),
    ("pcl", (96, 3 * 2 * 14 * 14)),
    ("bchw", (2, 3, 32, 48)),
]


@pytest.mark.parametrize("layout,shape", SHAPES)
@pytest.mark.parametrize("input_dtype", [torch.uint8, torch.float32])
@pytest.mark.parametrize("visual_dtype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("device", DEVICE)
def test_fused_mm_input_norm(
    default_vllm_config: None,
    layout: str,
    shape: tuple[int, ...],
    visual_dtype: torch.dtype,
    input_dtype: torch.dtype,
    device: str,
) -> None:
    with temporary_op_registry_oot():
        native = FusedMMInputNorm(IMAGE_MEAN, IMAGE_STD, RESCALE, layout=layout).to(device)
        assert not isinstance(native, HPUFusedMMInputNorm)

        register_op(FusedMMInputNorm, HPUFusedMMInputNorm)
        oot = FusedMMInputNorm(IMAGE_MEAN, IMAGE_STD, RESCALE, layout=layout).to(device)
        assert isinstance(oot, HPUFusedMMInputNorm)

        if not htorch.utils.internal.is_lazy():
            compile_config = HPUCompileConfig()
            oot = torch.compile(oot, **compile_config.get_compile_args())

        pixel_values = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)
        pixel_values = pixel_values.to(input_dtype)
        if input_dtype == torch.uint8:
            x3, out_shape = native._unpack(pixel_values)
            fused_mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32) * (1.0 / RESCALE)
            fused_std = torch.tensor(IMAGE_STD, dtype=torch.float32) * (1.0 / RESCALE)
            ref_out = tvf.normalize(x3.cpu().float().unsqueeze(-1), fused_mean, fused_std)
            ref_out = ref_out.reshape(out_shape).to(device=device, dtype=visual_dtype)
        else:
            ref_out = native.forward_native(pixel_values, visual_dtype)
        out = oot(pixel_values, visual_dtype)

        assert out.shape == pixel_values.shape
        assert out.dtype == visual_dtype
        if input_dtype == torch.uint8:
            rtol = 2 * torch.finfo(torch.float32).eps if visual_dtype == torch.float32 else 0
            torch.testing.assert_close(out, ref_out, rtol=rtol, atol=0)
        else:
            # torch.compile may fuse the multiply-add, which differs from eager by 1 ulp.
            torch.testing.assert_close(out, ref_out)


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
