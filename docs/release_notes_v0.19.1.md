# vLLM Gaudi Plugin v0.19.1 Release Notes

## Overview

This is a minor patch release on top of [v0.19.0](release_notes_v0.19.0.md) and continues to support [Intel® Gaudi® Software v1.24.0](https://docs.habana.ai/en/v1.24.0/Release_Notes/GAUDI_Release_Notes.html) with PyTorch 2.10.

The release lifts the **`transformers < 5` upper-bound constraint**, allowing users to install Hugging Face Transformers v5 alongside the plugin.

For the full set of features delivered in the v0.19.x line, see the [v0.19.0 release notes](release_notes_v0.19.0.md).

---

## Highlights

- Removed the **`transformers >= 4.56.0, <5`** dependency constraint from `requirements.txt`, enabling installation of Hugging Face Transformers v5.
- Refreshed the pinned **upstream vLLM stable commit** used by build scripts and CI.

---

## Plugin Core

- Removed the `transformers >= 4.56.0, <5` dependency entry entirely from `requirements.txt` (the plugin no longer pins a `transformers` version), and dropped the now-redundant `add_bos_token=True` test setting in `tests/models/language/generation/test_common.py`. ([#1420](https://github.com/vllm-project/vllm-gaudi/pull/1420))

---

## Full Changelog

| PR | Title | Author |
| --- | --- | --- |
| [#1420](https://github.com/vllm-project/vllm-gaudi/pull/1420) | Transformers v5 upgrade | @iboiko-habana |
