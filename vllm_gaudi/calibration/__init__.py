# SPDX-License-Identifier: Apache-2.0
"""Unified FP8 (Intel Neural Compressor) calibration for vLLM on Intel Gaudi.

One tool calibrates every model type. It drives the model with lm-eval tasks in two
separate processes: a ``MEASURE`` pass that collects maxabs statistics and a ``QUANTIZE``
pass that turns them into scales. Run it with ``vllm-gaudi-calibrate`` or
``python -m vllm_gaudi.calibration``.

The orchestrating process never imports vllm, torch or habana_frameworks, because
``QUANT_CONFIG`` is read when the HPU model runner is imported. Each phase therefore
runs in a fresh interpreter with its own environment.
"""
