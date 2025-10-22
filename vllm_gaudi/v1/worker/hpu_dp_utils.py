import torch
from contextlib import contextmanager
from vllm.config import VllmConfig
from dataclasses import dataclass
from typing import Optional
from vllm.platforms import current_platform
import habana_frameworks.torch as htorch


@dataclass
class HPUDPMetadata:
    hidden_states_across_dp: torch.Tensor
    router_logits_across_dp: torch.Tensor
    local_hidden_states: torch.Tensor

    @staticmethod
    def make(
        vllm_config: VllmConfig,
        num_tokens: int,
    ) -> "HPUDPMetadata":
        hidden_size = vllm_config.model_config.get_hidden_size()
        dp_size = vllm_config.parallel_config.data_parallel_size
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        num_tokens_across_dp = num_tokens * dp_size

        dtype = vllm_config.model_config.dtype
        device = current_platform.device_type

        num_expert_names = [
            "moe_num_experts",  # Dbrx
            "num_experts",  # Jamba
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
        ]
        num_experts = 0
        for name in num_expert_names:
            num_experts = getattr(vllm_config.model_config.hf_text_config, name, 0)
            if num_experts > 0:
                break
        assert num_experts > 0, \
            "No expert found in the model config. Please check the model config."

        hidden_states_across_dp = torch.empty(
            (num_tokens_across_dp, hidden_size),
            dtype=dtype,
            device=device,
        )
        router_logits_across_dp = torch.empty(
            (num_tokens_across_dp, num_experts),
            dtype=dtype,
            device=device,
        )
        local_num_tokens = (num_tokens //
                            tp_size) if vllm_config.parallel_config.use_sequence_parallel_moe else num_tokens
        local_hidden_states = torch.empty((local_num_tokens, hidden_size), dtype=dtype, device=device)

        return HPUDPMetadata(hidden_states_across_dp, router_logits_across_dp, local_hidden_states)


_hpu_dp_metadata: Optional[HPUDPMetadata] = None


@contextmanager
def override_hpu_dp_metadata(hpu_dp_metadata: Optional[HPUDPMetadata]):
    """A context manager that overrides the current HPU DP metadata.
    This is used to override the HPU DP metadata for a specific
    forward pass.
    """
    global _hpu_dp_metadata
    prev_metadata = _hpu_dp_metadata
    _hpu_dp_metadata = hpu_dp_metadata
    try:
        yield
    finally:
        _hpu_dp_metadata = prev_metadata


@contextmanager
def set_hpu_dp_metadata(
    vllm_config: VllmConfig,
    num_tokens: int,
):
    dp_metadata = None
    if htorch.utils.internal.is_lazy(
    ) and not vllm_config.model_config.enforce_eager and vllm_config.parallel_config.data_parallel_size > 1:
        dp_metadata = HPUDPMetadata.make(vllm_config, num_tokens)

    try:
        with override_hpu_dp_metadata(dp_metadata):
            yield
    finally:
        pass


def get_hpu_dp_metadata() -> Optional[HPUDPMetadata]:
    """Get the current HPU DP metadata."""
    return _hpu_dp_metadata
