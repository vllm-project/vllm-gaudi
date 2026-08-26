# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression guard for Mamba/GDN state aliasing after vLLM PR #51718.

#51718 renamed ``KVCacheTensor.shared_by`` to ``.layers`` and inverted the
meaning: the field now lists *every* layer of a group, each at its own byte
offset, instead of only the layers that genuinely share one page. Propagating a
single state tensor across that list therefore collapses a whole group of
recurrent layers onto one state -- no exception, just destroyed generation
quality (gsm8k accuracy 0.0 on Qwen3.5/3.6-35B-A3B).

The contract that must hold, per the upstream docstring, is: layers *within* a
group occupy distinct regions, while cache groups overlay each other (sound
because a block ID is owned by one group at a time).
"""

from types import SimpleNamespace

import habana_frameworks.torch  # noqa: F401
import torch
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec)

from vllm_gaudi.v1.worker import hpu_model_runner as hmr
from vllm_gaudi.v1.worker.hpu_model_runner import HPUModelRunner, _GDN_MAMBA_TYPES

BLOCK_SIZE = 128
NUM_BLOCKS = 8
MAX_NUM_SEQS = 4
NUM_GDN_GROUPS = 3
LAYERS_PER_GROUP = 10
STATE_SHAPES = ((2, 2), (2, 2))
STATE_DTYPES = (torch.bfloat16, torch.bfloat16)


class _StubModelRunner:
    """Runs the real ``initialize_kv_cache`` with the heavyweight steps stubbed.

    Only KV-cache allocation is under test, so model loading, attention-backend
    setup and input-batch construction are replaced by no-ops.
    """

    initialize_kv_cache = HPUModelRunner.initialize_kv_cache

    def __init__(self, device: str):
        self.device = device
        self.model = None
        self.num_gdn = NUM_GDN_GROUPS * LAYERS_PER_GROUP
        self.num_mamba_like_layers = self.num_gdn
        self.use_hybrid_cache = False
        self.use_naive_mamba_cache_sharing = True
        self._compact_gdn_enabled = True
        self._compact_gdn_group_ids: set[int] = set()
        self._compact_gdn_group_offset: dict[int, int] = {}
        self._gdn_req_to_base_slot: dict[str, int] = {}
        self._gdn_slot_free_list: list[int] = []
        self._num_gdn_groups = 0
        self._original_max_num_seqs = MAX_NUM_SEQS
        self.block_size = BLOCK_SIZE
        self.attn_block_size = BLOCK_SIZE
        self.enable_bucketing = False
        self.is_encoder_only_attn = False
        self.runner_only_attn_layers: set[str] = set()
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_caches: list = []
        self.attn_groups: list = []
        self.attn_backend = SimpleNamespace(
            get_kv_cache_shape=lambda num_blocks, block_size, num_kv_heads, head_size: (num_blocks, 1, 1, 1))
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
            compilation_config=SimpleNamespace(static_forward_context={}),
            kv_transfer_config=None,
        )

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(self, kv_cache_config: KVCacheConfig) -> None:
        pass

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        pass

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        pass

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]) -> None:
        pass


def _gdn_layer_names(group_idx: int) -> list[str]:
    return [f"gdn.{group_idx}.{pos}" for pos in range(LAYERS_PER_GROUP)]


def _attn_layer_names() -> list[str]:
    return [f"attn.{pos}" for pos in range(LAYERS_PER_GROUP)]


def _build_kv_cache_config() -> KVCacheConfig:
    """Mirror the post-#51718 config for a 3:1 GDN hybrid (e.g. Qwen3.5-35B-A3B).

    Upstream coalesces each group into a single KVCacheTensor listing all of the
    group's layers, so ``layers`` has ``LAYERS_PER_GROUP`` entries per group.
    """
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=STATE_SHAPES,
        dtypes=STATE_DTYPES,
        mamba_type=_GDN_MAMBA_TYPES[0],
    )
    attn_spec = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.bfloat16)

    groups = [
        KVCacheGroupSpec(layer_names=_gdn_layer_names(group_idx), kv_cache_spec=mamba_spec)
        for group_idx in range(NUM_GDN_GROUPS)
    ]
    groups.append(KVCacheGroupSpec(layer_names=_attn_layer_names(), kv_cache_spec=attn_spec))

    tensors = []
    for group in groups:
        page_size = group.kv_cache_spec.page_size_bytes
        tensors.append(
            KVCacheTensor(size=page_size * NUM_BLOCKS * LAYERS_PER_GROUP,
                          layers=list(group.layer_names),
                          layer_stride=page_size * NUM_BLOCKS,
                          block_stride=page_size))

    return KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=tensors, kv_cache_groups=groups)


def _allocate(monkeypatch) -> tuple[_StubModelRunner, dict]:
    """Run ``initialize_kv_cache``; return the runner and per-layer caches."""
    captured: dict = {}

    def _capture_bind(kv_caches, forward_context, runner_kv_caches):
        captured.update(kv_caches)

    monkeypatch.setattr(hmr, "bind_kv_cache", _capture_bind)
    monkeypatch.setattr(hmr, "maybe_set_mamba_kv_cache_groups_ids", lambda model, kv_cache_config: None)
    monkeypatch.setattr(hmr, "prepare_kernel_block_sizes",
                        lambda kv_cache_config, attn_groups: [BLOCK_SIZE] * len(kv_cache_config.kv_cache_groups))
    monkeypatch.setattr(hmr, "has_kv_transfer_group", lambda: False)
    monkeypatch.delenv("VLLM_PROFILE_PROMPT", raising=False)
    monkeypatch.delenv("VLLM_PROFILE_DECODE", raising=False)

    runner = _StubModelRunner(current_platform.device_type)
    runner.initialize_kv_cache(_build_kv_cache_config())
    return runner, captured


def test_gdn_layers_within_a_group_get_distinct_states(monkeypatch):
    """Every recurrent layer of a group needs its own state storage.

    Before the fix, the ``.layers`` propagation loop assigned one state tuple to
    all layers of a group, so the group's layers overwrote each other's
    recurrent state on every step.
    """
    _, kv_caches = _allocate(monkeypatch)

    for group_idx in range(NUM_GDN_GROUPS):
        states = [kv_caches[name] for name in _gdn_layer_names(group_idx)]
        distinct = {id(state[0]) for state in states}
        assert len(distinct) == LAYERS_PER_GROUP, (f"group {group_idx}: {len(distinct)} distinct states for "
                                                   f"{LAYERS_PER_GROUP} layers")


def test_gdn_groups_overlay_at_equal_layer_position(monkeypatch):
    """Groups overlay each other; slot separation comes from the group offset.

    Sharing storage across groups at the same layer position is the layout
    upstream describes and is safe because compact GDN indexes state slots as
    ``base_slot * num_gdn_groups + group_offset + 1``.
    """
    _, kv_caches = _allocate(monkeypatch)

    for pos in range(LAYERS_PER_GROUP):
        first = kv_caches[f"gdn.0.{pos}"]
        for group_idx in range(1, NUM_GDN_GROUPS):
            other = kv_caches[f"gdn.{group_idx}.{pos}"]
            assert all(a is b for a, b in zip(first, other)), f"position {pos} not overlaid across groups"


def test_gdn_state_slots_are_unique_per_layer(monkeypatch):
    """The (state tensor, slot) pair must be unique for all 30 GDN layers."""
    _, kv_caches = _allocate(monkeypatch)

    runner_groups = NUM_GDN_GROUPS
    states = set()
    for group_idx in range(NUM_GDN_GROUPS):
        for pos, name in enumerate(_gdn_layer_names(group_idx)):
            # Compact GDN slot for base_slot 0 in this group.
            slot = 0 * runner_groups + group_idx + 1
            states.add((id(kv_caches[name][0]), slot))

    assert len(states) == NUM_GDN_GROUPS * LAYERS_PER_GROUP


def test_compact_gdn_group_bookkeeping(monkeypatch):
    """Only the GDN groups take the compact path, each with its own offset."""
    runner, _ = _allocate(monkeypatch)

    assert runner._num_gdn_groups == NUM_GDN_GROUPS
    assert runner._compact_gdn_group_ids == set(range(NUM_GDN_GROUPS))
    assert sorted(runner._compact_gdn_group_offset.values()) == list(range(NUM_GDN_GROUPS))
