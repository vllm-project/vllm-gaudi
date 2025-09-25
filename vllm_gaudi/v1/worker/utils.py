from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec


def initialize_kv_cache_for_kv_sharing(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list["KVCacheGroupSpec"],
    kv_caches: dict[str, torch.Tensor],
) -> None:
    """
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
        kv_caches: The allocated kv_caches with layer names as keys.
            Note that layers in shared_kv_cache_layers.keys() are not
            originally included as it only contains layers which have its own
            KV cache allocation.
    """
    # Record index of KV cache group for each layer that allocates a KV cache.
    layer_to_kv_cache_group_idx: dict[str, int] = {}
    for i, kv_cache_group in enumerate(kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group_idx[layer_name] = i

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]
        group_idx = layer_to_kv_cache_group_idx[target_layer_name]
        kv_cache_groups[group_idx].layer_names.append(layer_name)
