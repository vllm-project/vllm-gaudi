# vLLM Gaudi Plugin v0.19.1.post1 Release Notes

## Overview

This is a post-release patch on top of [v0.19.1](release_notes_v0.19.1.md) and continues to support [Intel® Gaudi® Software v1.24.0](https://docs.habana.ai/en/v1.24.0/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.10.

The release fixes a **device mismatch error** during repetition/presence/frequency penalty sampling on HPU.

For the full set of features delivered in the v0.19.x line, see the [v0.19.0 release notes](release_notes_v0.19.0.md).

---

## Fixes

- Fixed `prompt_token_ids` device placement in selective sampling metadata creation — `prompt_token_ids` is now moved to `self.device` for both `skip_copy` paths, preventing runtime device mismatch errors during repetition/presence/frequency penalty application. ([#1542](https://github.com/vllm-project/vllm-gaudi/pull/1542), cherry-pick of [#1466](https://github.com/vllm-project/vllm-gaudi/pull/1466))

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1542](https://github.com/vllm-project/vllm-gaudi/pull/1542) | Port of: Fix HPU prompt_token_ids device placement for penalty sampling #1466 | @iboiko-habana |
