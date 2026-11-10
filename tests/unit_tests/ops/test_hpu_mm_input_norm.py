# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
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
# multiple of 256 and one that is not; Qwen-style (patches, C*L); NCHW images.
SHAPES = [
    ("pcl", (2 * 280 * 256, 3)),
    ("pcl", (7 * 17, 3)),
    ("pcl", (64, 3 * 2)),
    ("pcl", (96, 3 * 2 * 14 * 14)),
    ("bchw", (2, 3, 32, 48)),
]


@pytest.mark.parametrize("layout,shape", SHAPES)
@pytest.mark.parametrize("visual_dtype", [torch.bfloat16, torch.float])
@pytest.mark.parametrize("device", DEVICE)
def test_fused_mm_input_norm(
    default_vllm_config: None,
    layout: str,
    shape: tuple[int, ...],
    visual_dtype: torch.dtype,
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
        ref_out = native.forward_native(pixel_values, visual_dtype)
        out = oot(pixel_values, visual_dtype)

        assert out.shape == pixel_values.shape
        assert out.dtype == visual_dtype
        # torch.compile may fuse the multiply-add, which differs from eager by 1 ulp.
        torch.testing.assert_close(out, ref_out)
