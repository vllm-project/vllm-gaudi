# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the compact-GDN prefix-cache tests.

Shared prose prefix and state-tensor tolerance rule used by the state and
TP>1 correctness checks.
"""
import torch

# Coherent, varied paragraphs. Distinct enough that rotations share no long
# common prefix, so each rotation occupies its own kv blocks.
PARAS = (
    "The history of computing spans several centuries, beginning with mechanical "
    "calculators and progressing through electromechanical relays to the first "
    "electronic digital machines built during the middle of the twentieth century. ",
    "Early designs separated storage from processing, an idea that remains central "
    "to nearly every general-purpose computer in use today, from tiny embedded "
    "controllers to large distributed clusters spread across many datacenters. ",
    "As transistors replaced vacuum tubes, machines became smaller, cheaper, and "
    "far more reliable, enabling the personal computing revolution and eventually "
    "the mobile devices that billions of people now carry in their pockets. ",
    "Programming languages evolved from raw machine code toward high-level "
    "abstractions that let engineers express complex ideas concisely while "
    "compilers handled the tedious mapping down to individual instructions. ",
    "Networking then connected isolated machines into a global fabric, giving us "
    "electronic mail, the world wide web, streaming media, and the interconnected "
    "services that define modern digital life for people everywhere. ",
    "More recently, accelerators built for dense linear algebra made it practical "
    "to train enormous statistical models on unprecedented quantities of text, "
    "images, and audio gathered from across the public internet. ",
)


def assert_state_close(ref: torch.Tensor, got: torch.Tensor, name: str, tag) -> None:
    """Compare a restored state against its fresh-compute reference.

    Mirrors upstream check_mamba_state_equal: a bf16-scale tolerance, and a
    handful (<1%) of divergent elements from rounding are tolerated.
    """
    atol = rtol = 1e-2
    got = got[:ref.shape[0]]
    if torch.allclose(ref, got, atol=atol, rtol=rtol):
        return
    diff = (~torch.isclose(ref, got, atol=atol, rtol=rtol)).sum().item()
    if diff * 100 < ref.numel():
        print(f"[WARN] {name}@{tag}: {diff * 100 / ref.numel():.3f}% elements differ")
        return
    raise AssertionError(f"{name} state at {tag} diverged from fresh-compute "
                         f"reference: {diff}/{ref.numel()} elements exceed tol")
