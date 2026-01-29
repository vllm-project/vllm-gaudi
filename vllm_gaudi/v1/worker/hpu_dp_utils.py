import torch
from contextlib import contextmanager
from vllm.config import VllmConfig
from dataclasses import dataclass
from typing import Optional
from vllm.distributed import get_dp_group, get_ep_group
from vllm.platforms import current_platform
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.utils import has_quant_config
import habana_frameworks.torch as htorch


@dataclass
class HPUDPMetadata:
    hidden_states_across_dp: torch.Tensor
    topk_ids_across_dp: torch.Tensor
    topk_weights_across_dp: torch.Tensor
    local_hidden_states: torch.Tensor

    @staticmethod
    def make(
        vllm_config: VllmConfig,
        num_tokens: int,
    ) -> "HPUDPMetadata":
        hidden_size = vllm_config.model_config.get_hidden_size()
        dp_size = vllm_config.parallel_config.data_parallel_size
        tp_size = vllm_config.parallel_config.tensor_parallel_size

        if vllm_config.parallel_config.use_sequence_parallel_moe and (num_tokens % tp_size != 0):
            # make sure num_tokens is enough to be divided by tp_size for
            # sequence parallel MOE
            num_tokens = (num_tokens // tp_size + 1) * tp_size

        num_tokens_across_dp = num_tokens * dp_size

        dtype = vllm_config.model_config.dtype
        device = current_platform.device_type

        num_experts_per_tok = getattr(vllm_config.model_config.hf_text_config, "num_experts_per_tok", 0)
        assert num_experts_per_tok > 0, (
            "num_experts_per_tok must be greater than 0 in model config. Please check the model config.")

        is_quant_with_inc = has_quant_config(vllm_config.model_config) and get_config().use_dispatch_fn
        hidden_states_dtype = (torch.float8_e4m3fn if is_quant_with_inc else dtype)
        hidden_states_across_dp = torch.empty(
            (num_tokens_across_dp, hidden_size),
            dtype=hidden_states_dtype,
            device=device,
        )
        topk_ids_across_dp = torch.empty(
            (num_tokens_across_dp, num_experts_per_tok),
            dtype=torch.int64,
            device=device,
        )
        topk_weights_across_dp = torch.empty(
            (num_tokens_across_dp, num_experts_per_tok),
            dtype=dtype,
            device=device,
        )
        local_num_tokens = (num_tokens //
                            tp_size) if vllm_config.parallel_config.use_sequence_parallel_moe else num_tokens
        local_hidden_states = torch.empty((local_num_tokens, hidden_size), dtype=dtype, device=device)

        return HPUDPMetadata(hidden_states_across_dp, topk_ids_across_dp, topk_weights_across_dp, local_hidden_states)


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


def dispatch_tensor(input, output: torch.Tensor | None = None, is_sequence_parallel: bool = False) -> torch.Tensor:
    assert get_dp_group() is not None
    assert input.dim() == 2, "Input must be 2D"

    if output is None:
        # create output tensor
        input_size = input.size()
        # Allocate output tensor.
        output_size = list(input_size)
        if is_sequence_parallel:
            # if sequence parallel enabled, input was already being chunked by sp_size
            output_size[0] *= get_ep_group().world_size
        else:
            output_size[0] *= get_dp_group().world_size
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    torch.distributed.all_gather_into_tensor(
        output, input, group=get_ep_group().device_group if is_sequence_parallel else get_dp_group().device_group)

    return output


def dispatch_hidden_states(input, is_sequence_parallel):
    dp_metadata = get_hpu_dp_metadata()
    hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
    return dispatch_tensor(input, hidden_states_across_dp, is_sequence_parallel)
