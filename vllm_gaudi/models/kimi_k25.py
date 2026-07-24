# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPU dummy-input fix + processor re-registration for Kimi-K2.5/K2.6.

Upstream ``KimiK25DummyInputsBuilder`` has a count mismatch that only surfaces
during warmup when ``limit_per_prompt['vision_chunk'] > 1``:

* ``get_dummy_text`` emits ``num_media`` media tokens (one ``<|media_pad|>`` per
  requested item), but
* ``get_dummy_mm_data`` always returns a SINGLE dummy chunk.

``KimiK25Processor`` expands each media token by popping a per-chunk token count
from a list built from the provided chunks, so ``num_media`` tokens against 1
chunk raises ``IndexError: pop from empty list`` →
``ValueError: Failed to apply KimiK25Processor``. This crashes engine init for
any run configured with a vision_chunk limit above 1 (e.g. the offline batch
script uses ``limit_mm_per_prompt={"vision_chunk": 20}``); the OpenAI server did
not hit it because it left the limit at the default.

The fix replicates the dummy chunk to match ``num_media`` so token count and
chunk count agree. Vision-tower HPU overrides live separately in
``kimi_k25_vit.py``; this module only touches dummy-input building and
re-registers the processor on the upstream model class.
"""

from collections.abc import Mapping

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.kimi_k25 import (
    KimiK25DummyInputsBuilder,
    KimiK25ForConditionalGeneration,
    KimiK25MultiModalProcessor,
    KimiK25ProcessingInfo,
)


class HpuKimiK25DummyInputsBuilder(KimiK25DummyInputsBuilder):
    """Dummy-input builder that emits one chunk per requested media token.

    Upstream ``get_dummy_mm_data`` returns a single chunk regardless of
    ``mm_counts``; that mismatches the ``num_media`` media tokens emitted by
    ``get_dummy_text`` and makes the processor's per-chunk ``pop`` run dry when
    the vision_chunk limit is > 1. Replicate the upstream dummy item
    ``num_media`` times so the two counts agree.
    """

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_media = mm_counts.get("vision_chunk", 0)
        # ``get_dummy_mm_items()`` returns a single-element list (the larger of
        # the image/video-chunk dummy); replicate it to match ``num_media``.
        return {"vision_chunk": self.get_dummy_mm_items() * num_media}


# Re-register the multimodal processor on the upstream model class with the
# fixed dummy-inputs builder. ``register_processor`` overwrites the existing
# ``_processor_factory`` (it logs a warning on override, which is expected).
MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=KimiK25ProcessingInfo,
    dummy_inputs=HpuKimiK25DummyInputsBuilder,
)(KimiK25ForConditionalGeneration)
