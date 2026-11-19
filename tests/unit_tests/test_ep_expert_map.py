# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing ids under expert parallelism.

``HPUFp8MoEMethod`` hands ``mixture_of_experts`` only this rank's weights, so a
routed id must land on the slot of the expert it names. It now translates the
routing table through ``layer.expert_map`` -- global id to local slot, ``-1``
for an expert another rank owns -- and passes the local window ``(0, n - 1)``.

Two readings of how the op picks a slot are in play: by ``id - experts_min``
(what the repo's PyTorch stand-ins do), or by the id itself (what was measured
on Gaudi 2). With ``experts_min = 0`` they coincide, so the new convention is
checked against both. The former window, ``ep_rank * local_num_experts``
onwards, is wrong under either reading once the split is uneven or the
placement is round-robin.
"""

import pytest
import torch
from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map

EP_SIZE = 4


def slot_by_offset(rid, experts_min, experts_max, num_local):
    return rid - experts_min if experts_min <= rid <= experts_max else None


def slot_by_id(rid, experts_min, experts_max, num_local):
    return rid if 0 <= rid < num_local else None


READINGS = {"by_offset": slot_by_offset, "by_id": slot_by_id}


def owned_slots(expert_map):
    return [None if s < 0 else s for s in expert_map.tolist()]


@pytest.mark.parametrize("reading", READINGS)
@pytest.mark.parametrize("num_experts,placement", [(8, "linear"), (290, "linear"), (8, "round_robin")])
@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_local_ids_reach_the_owned_slot(reading, num_experts, placement, ep_rank):
    num_local, expert_map, _ = determine_expert_map(EP_SIZE, ep_rank, num_experts, placement)
    routing = torch.arange(num_experts)
    local_ids = expert_map[routing].to(torch.int64)  # as apply_monolithic does

    slots = [READINGS[reading](rid, 0, num_local - 1, num_local) for rid in local_ids.tolist()]

    assert local_ids.dtype == torch.int64
    assert slots == owned_slots(expert_map)


@pytest.mark.parametrize("reading", READINGS)
@pytest.mark.parametrize("num_experts,placement", [(290, "linear"), (8, "round_robin")])
def test_rank_offset_window_misses_uneven_and_round_robin(reading, num_experts, placement):
    ep_rank = 2
    num_local, expert_map, _ = determine_expert_map(EP_SIZE, ep_rank, num_experts, placement)
    ep_shift = ep_rank * num_local

    slots = [READINGS[reading](g, ep_shift, ep_shift + num_local - 1, num_local) for g in range(num_experts)]

    assert slots != owned_slots(expert_map)


@pytest.mark.parametrize("ep_rank", range(EP_SIZE))
def test_local_ids_on_the_op(ep_rank):
    """The new convention on the real op, bf16, 8 global / 2 local experts."""
    pytest.importorskip("habana_frameworks.torch")
    import torch.nn.functional as F

    num_local, expert_map, _ = determine_expert_map(EP_SIZE, ep_rank, 8)
    torch.manual_seed(100 + ep_rank)
    w13 = [torch.randn(16, 16, device="hpu", dtype=torch.bfloat16) * 0.1 for _ in range(num_local)]
    w2 = [torch.randn(16, 8, device="hpu", dtype=torch.bfloat16) * 0.1 for _ in range(num_local)]
    routing = [2, 3, 0, 5, 2, 7]
    torch.manual_seed(42)
    x = torch.randn(len(routing), 16, device="hpu", dtype=torch.bfloat16)
    local_ids = expert_map[torch.tensor(routing)].to(torch.int64).view(-1, 1).to("hpu")

    out = torch.ops.hpu.mixture_of_experts(
        hidden_states=x,
        expert_routing_table=local_ids,
        router_weights=torch.ones(len(routing), 1, device="hpu", dtype=torch.bfloat16),
        w12=tuple(w13),
        w3=tuple(w2),
        permuted_weights=True,
        activation="silu",
        experts_min=0,
        experts_max=num_local - 1,
    ).float().cpu()

    ref = torch.zeros_like(out)
    for t, slot in enumerate(owned_slots(expert_map)[g] for g in routing):
        if slot is not None:
            gate, up = (x[t].float() @ w13[slot].float().t()).chunk(2, dim=-1)
            ref[t] = ((F.silu(gate) * up) @ w2[slot].float().t()).cpu()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=0)
