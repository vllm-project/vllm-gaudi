# SPDX-License-Identifier: Apache-2.0
"""HPU override for the Gemma4 multimodal dummy-inputs builder.

Warmup needs to compile the vision graph at the *production* image resolution
(e.g. 864x480), but upstream ``Gemma4DummyInputsBuilder`` inherits
``_get_dummy_images`` from ``BaseDummyInputsBuilder``, which clamps any
``ImageDummyOptions`` width/height override down to the model's default
(224, since the Gemma4 image processor exposes ``size=None``). That makes
warmup build a 224x224 dummy and compile the wrong vision-tower shape.

The clamp is removed for Gemma4 by swapping ``_get_dummy_images`` at import
time (the same approach ``kimi_k25_vit.py`` / ``qwen3_5.py`` use). Guarded so
an upstream rename makes this a no-op rather than an error.
"""

from PIL import Image

import vllm.model_executor.models.gemma4_mm as _gemma4


def _get_dummy_images(self, *, width, height, num_images, overrides=None):
    """Honor width/height overrides instead of clamping to the model default.

    Mirrors ``BaseDummyInputsBuilder._get_dummy_images`` but, when an override
    is larger than the default, uses the override so warmup compiles the real
    production resolution.
    """
    if num_images == 0:
        return []
    if overrides is not None:
        if getattr(overrides, "width", None):
            width = overrides.width
        if getattr(overrides, "height", None):
            height = overrides.height
    image = Image.new("RGB", (width, height), color=255)
    return [image] * num_images


_builder_cls = getattr(_gemma4, "Gemma4DummyInputsBuilder", None)
if _builder_cls is not None and hasattr(_builder_cls, "_get_dummy_images"):
    _builder_cls._get_dummy_images = _get_dummy_images
