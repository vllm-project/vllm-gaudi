# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import unittest

import torch
import habana_frameworks.torch.core  # noqa: F401


class _RouterGate(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, value):
        return torch.nn.functional.linear(value, self.weight), None


@unittest.skipUnless(torch.hpu.is_available() and os.environ.get("KEEP_NUMERICS", "false").lower() == "true",
                     "Requires HPU hardware and KEEP_NUMERICS=true")
class TestHPUMoERouterPrecision(unittest.TestCase):
    def test_compiled_router_preserves_native_rounding(self):
        generator = torch.Generator().manual_seed(413)
        weight = (torch.randn(256, 2048, generator=generator) * 0.03).to(torch.bfloat16)
        gate = _RouterGate(weight.to("hpu"))
        target = torch.randn(2048, generator=generator).to(torch.bfloat16)

        def route(logits):
            probabilities = logits.softmax(-1, dtype=torch.float32)
            weights, ids = probabilities.topk(8, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
            return weights.to(torch.bfloat16), ids

        def forward(value):
            logits = gate(value)[0]
            return route(logits)

        compiled = torch.compile(forward, backend="hpu_backend", fullgraph=True, dynamic=False)
        baseline = None
        with torch.no_grad():
            for batch in (1, 6, 45, 128, 2048):
                with self.subTest(batch=batch):
                    value = torch.randn(batch, 2048, generator=generator).to(torch.bfloat16)
                    row = min(2, batch - 1)
                    value[row] = target
                    value = value.to("hpu")
                    expected = tuple(result.cpu()[row] for result in route(gate(value)[0]))
                    actual = tuple(result.cpu()[row] for result in compiled(value))
                    for result, reference in zip(actual, expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    if baseline is None:
                        baseline = actual
                    for result, reference in zip(actual, baseline):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
