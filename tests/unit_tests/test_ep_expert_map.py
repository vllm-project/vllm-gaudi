# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-id translation for expert parallelism.

``torch.ops.hpu.mixture_of_experts`` indexes its local weight list directly by
the entry it finds in ``expert_routing_table``. The table must therefore hold
LOCAL expert ids (``0..local_num_experts-1``, ``-1`` for tokens owned by
another rank), and ``experts_min``/``experts_max`` must be the local range.

Passing GLOBAL ids together with a global window
(``ep_shift .. ep_shift + local - 1``) happens to work for ``ep_rank == 0``,
because the two conventions coincide there. Every other rank skips the experts
it owns and picks up ones it does not -- silently, since nothing raises.

These tests model the kernel's selection rule in plain PyTorch so the
convention is pinned down without requiring a Gaudi card or a distributed run.
"""

import pytest
import torch

GLOBAL_EXPERTS = 8
EP_SIZE = 4
LOCAL_EXPERTS = GLOBAL_EXPERTS // EP_SIZE
ROUTING = [2, 3, 0, 5, 2, 7]


def build_expert_map(ep_rank: int) -> torch.Tensor:
    """What ExpertMapManager builds: global -> local, -1 when not local."""
    m = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.long)
    lo = ep_rank * LOCAL_EXPERTS
    m[lo : lo + LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS)
    return m


def kernel_slots(routing_table, experts_min, experts_max):
    """The kernel's rule: compute a slot iff its id is inside the window, and
    use that id to index this rank's local weight list."""
    slots = []
    for rid in routing_table.tolist():
        if experts_min <= rid <= experts_max and 0 <= rid < LOCAL_EXPERTS:
            slots.append(rid)
        else:
            slots.append(None)
    return slots


def owned_slots(ep_rank: int):
    """Ground truth: the rank computes exactly the experts it owns."""
    lo, hi = ep_rank * LOCAL_EXPERTS, (ep_rank + 1) * LOCAL_EXPERTS - 1
    return [(g - lo) if lo <= g <= hi else None for g in ROUTING]


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_local_ids_route_to_the_owned_experts(ep_rank):
    """With the translation in place every rank computes what it owns."""
    routing = build_expert_map(ep_rank)[torch.tensor(ROUTING)]
    slots = kernel_slots(routing, 0, LOCAL_EXPERTS - 1)
    assert slots == owned_slots(ep_rank)


def test_expert_map_marks_remote_experts_with_minus_one():
    """The sentinel is what keeps non-local tokens out of the kernel."""
    for ep_rank in range(EP_SIZE):
        m = build_expert_map(ep_rank)
        lo = ep_rank * LOCAL_EXPERTS
        local = m[lo : lo + LOCAL_EXPERTS]
        assert local.tolist() == list(range(LOCAL_EXPERTS))
        remote = torch.cat([m[:lo], m[lo + LOCAL_EXPERTS :]])
        assert (remote == -1).all()


@pytest.mark.parametrize("ep_rank", range(1, EP_SIZE))
def test_untranslated_global_ids_route_wrongly(ep_rank):
    """Regression guard: global ids plus a global window drop owned experts.

    Only ranks above 0 are affected, which is why single-rank and non-EP runs
    never surfaced this.
    """
    ep_shift = ep_rank * LOCAL_EXPERTS
    slots = kernel_slots(torch.tensor(ROUTING), ep_shift, LOCAL_EXPERTS + ep_shift - 1)
    assert slots != owned_slots(ep_rank)
