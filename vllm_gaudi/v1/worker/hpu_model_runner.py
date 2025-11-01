# SPDX-License-Identifier: Apache-2.0
import collections
import contextlib
import functools
from functools import partial
import itertools
import math
import os
import time
from dataclasses import dataclass, field, fields
from typing import (TYPE_CHECKING, Any, Callable, Optional, TypeAlias, Union, cast)

import habana_frameworks.torch as htorch
import habana_frameworks.torch.internal.bridge_config as bc
import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn as nn
import vllm_gaudi.extension.environment as environment
from vllm_gaudi.extension.bucketing.common import HPUBucketingManager
from vllm_gaudi.extension.defragmentation import OnlineDefragmenter
from vllm_gaudi.extension.profiler import (HabanaHighLevelProfiler, HabanaMemoryProfiler, HabanaProfilerCounterHelper,
                                           format_bytes, setup_profiler)
from vllm_gaudi.extension.runtime import finalize_config, get_config
from vllm_gaudi.extension.unified import (create_unified_batch)
from vllm_gaudi.extension.utils import align_and_pad, pad_list, with_default
from vllm_gaudi.extension.debug import init_debug_logger
from vllm_gaudi.v1.worker.hpu_dp_utils import set_hpu_dp_metadata

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.attention.layer import MLAAttention
from vllm.attention.selector import get_attn_backend
from vllm.config import (VllmConfig, update_config)
from vllm.distributed.kv_transfer import (get_kv_transfer_group, has_kv_transfer_group)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (VocabParallelEmbedding)
from vllm.model_executor.model_loader import get_model, get_model_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargs, MultiModalKwargsItem)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.multimodal.inputs import PlaceholderRange
from vllm.sampling_params import SamplingType
from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.utils.import_utils import LazyLoader
from vllm.utils.jsontree import json_map_leaves
from vllm_gaudi.utils import (HPUCompileConfig, is_fake_hpu, async_h2d_copy)
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheSpec, KVCacheTensor, MLAAttentionSpec)
from vllm.v1.worker.kv_connector_model_runner_mixin import (KVConnectorModelRunnerMixin)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, LogprobsTensors, DraftTokenIds, ModelRunnerOutput,
                             AsyncModelRunnerOutput, KVConnectorOutput)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.utils import bind_kv_cache
from vllm.v1.utils import CpuGpuBuffer
from vllm_gaudi.v1.worker.hpu_input_batch import InputBatch, CachedRequestState
from vllm.distributed.parallel_state import get_pp_group, get_dp_group
from vllm.model_executor.models.interfaces import (SupportsMultiModal, supports_eagle3, supports_transcription)
from vllm.model_executor.models.interfaces_base import (VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.transformers_utils.config import is_interleaved
from vllm.v1.worker.utils import (gather_mm_placeholders, sanity_check_mm_encoder_outputs, scatter_mm_placeholders)
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.logits_processor import build_logitsprocs
from torch.nn.utils.rnn import pad_sequence
from vllm.v1.core.sched.output import NewRequestData
from vllm.sampling_params import SamplingParams
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor.models import supports_lora, supports_multimodal
from vllm_gaudi.extension.ops import LoraMask as LoraMask
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks

if TYPE_CHECKING:
    import xgrammar as xgr
    import xgrammar.kernels.apply_token_bitmask_inplace_torch_compile as xgr_torch_compile  # noqa: E501
    import xgrammar.kernels.apply_token_bitmask_inplace_cpu as xgr_cpu
    from vllm.v1.core.scheduler import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_cpu = LazyLoader("xgr_cpu", globals(), "xgrammar.kernels.apply_token_bitmask_inplace_cpu")
    xgr_torch_compile = LazyLoader("xgr_torch_compile", globals(),
                                   "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile")

from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()

_TYPE_CACHE: dict[str, dict[str, Any]] = {}

hpu_buffer: list[list[torch.Tensor]] = []


class BucketingFailedException(Exception):
    pass


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncHPUModelRunnerOutput(AsyncModelRunnerOutput):

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.hpu.Stream,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids

        self._async_copy_ready_event = torch.hpu.Event()
        default_stream = torch.hpu.current_stream()
        with torch.hpu.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._sampled_token_ids_cpu = self._sampled_token_ids.to('cpu', non_blocking=True)
            self._async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        
        This function blocks until the copy is finished.
        """

        # Release the device tensor once the copy has completed
        self._async_copy_ready_event.synchronize()

        valid_sampled_token_ids = self._sampled_token_ids_cpu.tolist()
        del self._sampled_token_ids
        for i in self._invalid_req_indices:
            if i < len(valid_sampled_token_ids):
                valid_sampled_token_ids[i].clear()

        output = self._model_runner_output
        output.sampled_token_ids[:len(valid_sampled_token_ids)] = valid_sampled_token_ids
        return output


@dataclass
class PromptDecodeInfo:
    prompt_req_ids: list[str]
    decode_req_ids: list[str]
    prompt_scheduled_tokens: list[int]


@dataclass
class PromptData:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: HPUAttentionMetadataV1


@dataclass
class DecodeData:
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    attn_metadata: Optional[HPUAttentionMetadataV1] = None


empty_list: Callable[[], list] = lambda: field(default_factory=list)


@dataclass
class BatchContents:
    req_ids: list[str] = empty_list()
    token_ids: list[list[int]] = empty_list()
    context_lens: list[int] = empty_list()
    prompt_lens: list[int] = empty_list()
    blocks: list[list[int]] = empty_list()
    logits_positions: list[list[int]] = empty_list()

    def get_num_tokens(self):
        return [len(t) for t in self.token_ids]

    def clone(self):
        return BatchContents(req_ids=self.req_ids.copy(),
                             token_ids=[t.copy() for t in self.token_ids],
                             context_lens=self.context_lens.copy(),
                             blocks=[b.copy() for b in self.blocks],
                             logits_positions=[lp.copy() for lp in self.logits_positions])


# TODO(kzawora): remove this
@dataclass
class PrefillInputData:
    request_ids: list = empty_list()
    prompt_lens: list = empty_list()
    token_ids: list = empty_list()
    position_ids: list = empty_list()
    attn_metadata: list = empty_list()
    logits_indices: list = empty_list()
    logits_requests: list = empty_list()


# TODO(kzawora): remove this
@dataclass
class DecodeInputData:
    num_decodes: int
    token_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    attn_metadata: Optional[HPUAttentionMetadataV1] = None
    logits_indices: Optional[torch.Tensor] = None
    spec_decode_metadata: Optional[SpecDecodeMetadata] = None


def bool_helper(value):
    value = value.lower()
    return value in ("y", "yes", "t", "true", "on", "1")


Mergeable: TypeAlias = Union[BatchContents, PrefillInputData]


def shallow_tuple(obj: Mergeable) -> tuple:
    """Returns a shallow tuple with dataclass field values"""
    # Unfortunately dataclasses.astuple deepcopies the data
    # se we can't use it
    return tuple(getattr(obj, field.name) for field in fields(obj))


def merge_contents(lhs: Mergeable, *rhs: Mergeable):
    """Extends all internal lists of a dataclass with """
    """values from given objects"""
    lhs_type = type(lhs)
    lhs_tuple = shallow_tuple(lhs)
    for other in rhs:
        assert lhs_type is type(other), \
            'Only objects of the same type can be merged'
        for dst, src in zip(lhs_tuple, shallow_tuple(other)):
            dst.extend(src)


def flatten(in_list):
    """Return a flattened representation of a list"""
    return list(itertools.chain(*in_list))


def gather_list(input, indices, v):
    """Gather values from input using indices"""
    return [input[i] if i is not None else v for i in indices]


def ensure_decodes_first(b: InputBatch):
    num_reqs = b.num_reqs
    while True:
        # Find the first prompt index
        first_prompt_index = None
        for i in range(num_reqs):
            if b.num_computed_tokens_cpu[i] < b.num_prompt_tokens[i]:
                first_prompt_index = i
                break
        if first_prompt_index is None:
            break

        # Find the last decode index
        last_decode_index = None
        for i in reversed(range(num_reqs)):
            if b.num_computed_tokens_cpu[i] >= b.num_prompt_tokens[i]:
                last_decode_index = i
                break
        if last_decode_index is None:
            break

        # Sanity
        assert first_prompt_index != last_decode_index

        # Check if done
        if first_prompt_index > last_decode_index:
            break

        # Swap
        b.swap_states(first_prompt_index, last_decode_index)


def get_target_layer_suffix_list(model_type) -> list[str]:
    # This sets the suffix for the hidden layer name, which is controlled by
    # VLLM_CONFIG_HIDDEN_LAYERS. The default suffix is "DecoderLayer," which is
    # applicable for most language models such as LLaMA, Qwen, and BART. If the
    # model's decoder layer name differs from the default, it will need to
    # be specified here.
    decoder_layer_table = {
        "gpt_bigcode": "BigCodeBlock",
    }

    return [decoder_layer_table.get(model_type, "DecoderLayer"), "EncoderLayer"]


def modify_model_layers(module: torch.nn.Module, suffix_list: list[str], n=1, counter=None):
    """Currently add mark_step at the end of specified layers.
    """

    def forward_hook(module, args, output):
        htorch.core.mark_step()
        return output

    if counter is None:
        counter = [0]

    for child_name, child_module in module.named_children():
        if any(child_module.__class__.__name__.endswith(layer) for layer in suffix_list):
            counter[0] += 1
            if counter[0] % n == 0:
                child_module.register_forward_hook(forward_hook)
        else:
            modify_model_layers(child_module, suffix_list, n, counter)


def is_mm_optimized(model):
    return 'Gemma3ForConditionalGeneration' in str(type(model.model)) \
        if hasattr(model, 'model') else \
        'Gemma3ForConditionalGeneration' in str(type(model))


class HpuModelAdapter(torch.nn.Module, KVConnectorModelRunnerMixin):

    def __init__(self, model, vllm_config):
        super().__init__()
        self.model = model
        self.prefill_use_fusedsdpa = get_config().prompt_attn_impl == 'fsdpa_impl'
        self.recompute_cos_sin = os.getenv('VLLM_COS_SIN_RECOMPUTE', 'false').lower() in ['1', 'true']
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.dtype = vllm_config.model_config.dtype
        self._rotary_embed_module = self._get_rotary_embedding_module(self.model)
        self._rotary_prepare_cos_sin = self._get_prepare_cos_sin()
        self.unified_attn = get_config().unified_attn
        self.flatten_input = get_config().flatten_input
        self.is_mm_optimized = is_mm_optimized(self.model)
        self.sliding_window = vllm_config.model_config.get_sliding_window()
        self.interleaved_sliding_window = is_interleaved(vllm_config.model_config.hf_text_config)
        if self.interleaved_sliding_window:
            self.use_window_sdpa = os.getenv("PT_HPU_SDPA_QKV_SLICE_MODE_FWD", "false").strip().lower() in ("1", "true")
            self.slice_size = int(os.getenv("PT_HPU_SDPA_BC_FACTOR", "1024"))
            self.slice_thld = int(os.environ.get('VLLM_FUSEDSDPA_SLIDE_THLD', '8192'))

        # for DP
        self.dummy_num_input_tokens = -1
        self.num_tokens_across_dp = [self.dummy_num_input_tokens] * self.vllm_config.parallel_config.data_parallel_size
        self.dummy_num_tokens_across_dp_cpu = torch.tensor(self.num_tokens_across_dp, device='cpu', dtype=torch.int32)

        # Vision embedding can be also wrapped in HPU graph once all the dynamic shape is removed.
        # Performance can be greatly improved.
        if htorch.utils.internal.is_lazy() and \
           MULTIMODAL_REGISTRY.supports_multimodal_inputs(vllm_config.model_config) and self.is_mm_optimized:
            if hasattr(self.model, 'vision_tower'):
                self.model.vision_tower = htorch.hpu.wrap_in_hpu_graph(self.model.vision_tower,
                                                                       disable_tensor_cache=False)
            if hasattr(self.model, 'multi_modal_projector'):
                self.model.multi_modal_projector = \
                        htorch.hpu.wrap_in_hpu_graph( \
                        self.model.multi_modal_projector, \
                        disable_tensor_cache=True)

    def _get_rotary_embedding_module(self, model: torch.nn.Module):
        """
        Dynamically get the RotaryEmbedding layer in the model.
        This function will recursively search through the module
        hierarchy to find and return a RotaryEmbedding layer.
        If no such layer is found, it returns None.
        """
        if model is None:
            return None

        if model.__class__.__name__.endswith("RotaryEmbedding"):
            return model

        if hasattr(model, 'children'):
            for child in model.children():
                result = self._get_rotary_embedding_module(child)
                if result is not None:
                    return result
        return None

    def _get_prepare_cos_sin(self):
        if self._rotary_embed_module is not None and hasattr(self._rotary_embed_module, 'prepare_cos_sin'):
            return self._rotary_embed_module.prepare_cos_sin
        return None

    def _reset_rotary_cos_sin(self):
        if hasattr(self._rotary_embed_module, "cos"):
            delattr(self._rotary_embed_module, "cos")
        if hasattr(self._rotary_embed_module, "sin"):
            delattr(self._rotary_embed_module, "sin")

    def _set_attn_bias(self, attn_metadata, batch_size, seq_len, device, dtype):
        if (attn_metadata is None or (self.prefill_use_fusedsdpa and attn_metadata.block_list is None)
                or not attn_metadata.is_prompt):
            return attn_metadata

        if attn_metadata.attn_bias is not None:
            return attn_metadata

        prefill_metadata = attn_metadata

        seq_lens_t = prefill_metadata.seq_lens_tensor
        context_lens_t = prefill_metadata.context_lens_tensor

        block_list = attn_metadata.block_list
        max_context_len = (block_list.size(-1) // batch_size if block_list is not None else 0)
        max_context_len = max_context_len * self.block_size
        past_mask = torch.arange(0, max_context_len, dtype=torch.int32, device=device)
        past_mask = (past_mask.view(1, -1).expand(batch_size, -1).ge(context_lens_t.view(-1, 1)).view(
            batch_size, 1, -1).expand(batch_size, seq_len, -1).view(batch_size, 1, seq_len, -1))

        len_mask = (torch.arange(0, seq_len, device=device, dtype=torch.int32).view(1, seq_len).ge(
            seq_lens_t.unsqueeze(-1)).view(batch_size, 1, 1, seq_len))
        causal_mask = torch.triu(torch.ones((batch_size, 1, seq_len, seq_len), device=device, dtype=torch.bool),
                                 diagonal=1)
        mask = causal_mask.logical_or(len_mask)
        mask = torch.concat((past_mask, mask), dim=-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf))
        attn_metadata = custom_tuple_replace(prefill_metadata, "TrimmedAttentionMetadata", attn_bias=attn_bias)
        return attn_metadata

    def _set_attn_bias_for_sliding_window(self, attn_metadata, batch_size, seq_len, window_size, device, dtype):

        if (attn_metadata is None or not attn_metadata.is_prompt):
            return attn_metadata

        prefill_metadata = attn_metadata
        shift = 0

        # FusedSDPA with window_size is only supported when the seq_len is multiple of the slice_size
        if self.prefill_use_fusedsdpa and self.use_window_sdpa and \
            seq_len >= self.slice_thld and self.slice_size != 0 and \
            seq_len % self.slice_size == 0 and attn_metadata.block_list is None:
            # no need to set sliding window mask, just use built-in window-sdpa
            return attn_metadata

        if self.prefill_use_fusedsdpa and attn_metadata.block_list is not None:
            context_lens_t = prefill_metadata.context_lens_tensor

            block_list = attn_metadata.block_list
            max_context_len = (block_list.size(-1) // batch_size if block_list is not None else 0)
            max_context_len = max_context_len * self.block_size

            invalid_lens_t = context_lens_t - window_size + torch.arange(seq_len, device=device) - 1
            past_indices = torch.arange(max_context_len, device=device)
            past_mask = ((past_indices.unsqueeze(0) > invalid_lens_t.unsqueeze(-1)) &
                         (past_indices.unsqueeze(0) < context_lens_t.unsqueeze(-1).unsqueeze(0))).unsqueeze(1)

            # Create boolean sliding window mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=shift)
            causal_mask = torch.triu(causal_mask, diagonal=shift - window_size + 1)
            causal_mask = causal_mask.view(batch_size, 1, seq_len, seq_len)

            # TODO: Investigate further - Removing Padding cause accuracy issue
            # seq_lens_t = prefill_metadata.seq_lens_tensor
            # len_mask = (torch.arange(0, seq_len, device=device, dtype=torch.int32).view(1, seq_len).lt(
            #     seq_lens_t.unsqueeze(-1)).view(batch_size, 1, 1, seq_len))
            # causal_mask = causal_mask.logical_and(len_mask)

            mask = torch.concat((past_mask, causal_mask), dim=-1)
            attn_bias = torch.where(mask, torch.tensor(0.0, dtype=dtype, device=device),
                                    torch.tensor(float('-inf'), dtype=dtype, device=device))
        else:
            # CAUSAL MASK without removing padding (CAUSAL+sliding window)
            # removing padding cause accuracy issue for images input
            tensor = torch.full((batch_size, 1, seq_len, seq_len), device=device, dtype=dtype, fill_value=1)
            mask = torch.tril(tensor, diagonal=shift)
            mask = torch.triu(mask, diagonal=shift - window_size + 1)
            attn_bias = torch.log(mask)

        attn_metadata = prefill_metadata._replace(window_attn_bias=attn_bias)
        return attn_metadata

    def _set_block_mapping(self, metadata, batch_size, device, dtype, is_window_block=False):
        if is_window_block:
            block_usage = metadata.window_block_usage
            block_groups = metadata.window_block_groups
        else:
            block_usage = metadata.block_usage
            block_groups = metadata.block_groups

        mask = torch.arange(0, self.block_size, device=device, dtype=torch.int32).unsqueeze(0)
        mask = mask >= block_usage.unsqueeze(-1)
        attn_bias = (torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, -math.inf))

        if not is_fake_hpu():
            block_mapping = torch.nn.functional.one_hot(block_groups, num_classes=batch_size)
        else:
            # Unfortunately one_hot on CPU
            # doesn't handle out of bounds classes so we need to convert
            # all negative values to 0 (block_mapping) or bs (block_groups)
            block_groups = block_groups.to(torch.long)
            block_mapping = torch.nn.functional.relu(block_groups)
            block_mapping = torch.nn.functional.one_hot(block_mapping, num_classes=batch_size)
            oob_values = block_groups.lt(0)
            block_mapping.masked_fill_(oob_values.unsqueeze(-1), 0)
            block_groups.masked_fill_(oob_values, batch_size)
            if is_window_block:
                metadata = custom_tuple_replace(metadata, "TrimmedAttentionMetadata", window_block_groups=block_groups)
            else:
                metadata = custom_tuple_replace(metadata, "TrimmedAttentionMetadata", block_groups=block_groups)
        block_mapping = block_mapping.to(dtype)
        if is_window_block:
            metadata = custom_tuple_replace(metadata,
                                            "TrimmedAttentionMetadata",
                                            window_block_mapping=block_mapping,
                                            window_attn_bias=attn_bias)
        else:
            metadata = custom_tuple_replace(metadata,
                                            "TrimmedAttentionMetadata",
                                            block_mapping=block_mapping,
                                            attn_bias=attn_bias)
        return metadata

    def _update_metadata(self, attn_metadata, batch_size, seq_len, device, dtype):
        if attn_metadata.is_prompt:
            attn_metadata = self._set_attn_bias(attn_metadata, batch_size, seq_len, device, dtype)
            if self.interleaved_sliding_window:
                attn_metadata = self._set_attn_bias_for_sliding_window(attn_metadata, batch_size, seq_len,
                                                                       self.sliding_window, device, dtype)
        else:
            attn_metadata = self._set_block_mapping(attn_metadata, batch_size, device, dtype)
            if self.interleaved_sliding_window:
                attn_metadata = self._set_block_mapping(attn_metadata, batch_size, device, dtype, True)
        return attn_metadata

    def forward(self, *args, **kwargs):
        # TODO(kzawora): something goes VERY WRONG when operating on
        # kwargs['attn_metadata'].slot_mapping, compared to untrimmed metadata
        kwargs = kwargs.copy()
        #        selected_token_indices = kwargs.pop('selected_token_indices')
        if 'lora_mask' in kwargs:
            lora_mask = kwargs['lora_mask']
            LoraMask.setLoraMask(lora_mask)
            kwargs.pop('lora_mask')
        if 'warmup_mode' in kwargs:
            kwargs.pop('warmup_mode')
        input_ids = kwargs['input_ids']
        if not self.unified_attn:
            kwargs['attn_metadata'] = self._update_metadata(kwargs['attn_metadata'], input_ids.size(0),
                                                            input_ids.size(1), input_ids.device, self.dtype)
        if self._rotary_prepare_cos_sin is not None:
            self._rotary_prepare_cos_sin(kwargs['positions'], recompute_cos_sin=self.recompute_cos_sin)
        attn_meta = kwargs.pop('attn_metadata')
        if 'kv_caches' in kwargs:
            kwargs.pop('kv_caches')

        # If multimodal inputs, update kwargs
        model_mm_kwargs = kwargs.pop('model_mm_kwargs', None)
        if model_mm_kwargs is not None:
            kwargs.update(model_mm_kwargs)

        num_real_tokens = input_ids.size(0) * input_ids.size(1)

        if self.flatten_input:
            kwargs['input_ids'] = input_ids.view(-1)
        # here num_tokens and num_tokens_across_dp are dummy values which are
        # used to skip sync in forward_context between DP ranks
        with set_forward_context(attn_meta,
                                 self.vllm_config,
                                 num_tokens=self.dummy_num_input_tokens,
                                 num_tokens_across_dp=self.dummy_num_tokens_across_dp_cpu), set_hpu_dp_metadata(
                                     self.vllm_config, num_real_tokens):
            hidden_states = self.model(*args, **kwargs)
            if self._rotary_prepare_cos_sin is not None:
                self._reset_rotary_cos_sin()
        return hidden_states

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, is_multimodal=False):
        return self.model.get_input_embeddings(input_ids=input_ids,
                                               multimodal_embeddings=multimodal_embeddings,
                                               is_multimodal=is_multimodal)

    def get_multimodal_embeddings(self, **batched_mm_inputs):
        return self.model.get_multimodal_embeddings(**batched_mm_inputs)

    def compute_logits(self, *args, **kwargs):
        return self.model.compute_logits(*args, **kwargs)

    # def sample(self, *args, **kwargs):
    #    return self.sampler(*args, **kwargs)

    def generate_proposals(self, *args, **kwargs):
        return self.model.generate_proposals(*args, **kwargs)

    # sampler property will be used by spec_decode_worker
    # don't rename
    # @property
    # def sampler(self):
    #    return self.model.sampler


def _maybe_wrap_in_hpu_graph(*args, **kwargs):
    return htorch.hpu.wrap_in_hpu_graph(HpuModelAdapter(
        *args, **kwargs), disable_tensor_cache=True) if htorch.utils.internal.is_lazy() else HpuModelAdapter(
            *args, **kwargs)


def subtuple(obj: object, typename: str, to_copy: list[str], to_override: Optional[dict[str, object]] = None):
    if obj is None:
        return None
    if to_override is None:
        to_override = {}
    fields = set(to_copy) | set(to_override.keys())
    if type(obj) is dict:
        values = {key: obj[key] for key in fields if key in obj}
    else:
        values = {f: to_override.get(f, getattr(obj, f)) for f in fields}
    if typename not in _TYPE_CACHE:
        _TYPE_CACHE[typename] = {'type': collections.namedtuple(typename, ' '.join(fields)), 'fields': fields}
    return _TYPE_CACHE[typename]['type'](**values)  # type: ignore


def custom_tuple_replace(obj: object, typename: str, **to_override):
    # Torch compile dynamo doesn't support calling any named tuple
    # dynamic methods other than len and get_attr. This function is
    # a torch.compile friendly version of tuple._replace

    cached_type = _TYPE_CACHE[typename]['type']
    fields = _TYPE_CACHE[typename]['fields']
    values = {
        field: getattr(obj, field)
        for field in fields  # type: ignore
    }
    values.update(to_override)
    return cached_type(**values)  # type: ignore


def trim_attn_metadata(metadata: HPUAttentionMetadataV1) -> object:
    # NOTE(kzawora): To anyone working on this in the future:
    # Trimming metadata is required when using HPUGraphs.
    # Attention metadata is going to be hashed by PT bridge, and
    # appropriate HPUGraphs will be matched based on all inputs' hash.

    # Before you put more keys in here, make sure you know their
    # value type and make sure you know how it's going to be hashed.
    # You can find that information in input_hash function
    # in habana_frameworks/torch/hpu/graphs.py. You can also hash
    # it manually with torch.hpu.graphs.input_hash(attention_metadata)

    # If you use primitive types here - they will get hashed based
    # on their value. You *will* get lots of excessive graph captures
    # (and an OOM eventually) if you decide to put something like
    # seq_len int here.
    # If you absolutely need a scalar, put it in a tensor. Tensors
    # get hashed using their metadata, not their values:
    # input_hash(torch.tensor(123)) == input_hash(torch.tensor(321))
    # input_hash(123) != input_hash(321)
    # input_hash("abc") != input_hash("cba")
    attention_metadata = subtuple(metadata, 'TrimmedAttentionMetadata', [
        'attn_bias',
        'seq_lens_tensor',
        'context_lens_tensor',
        'block_list',
        'block_mapping',
        'block_usage',
        'slot_mapping',
        'is_prompt',
        'block_size',
        'block_groups',
        'window_block_list',
        'window_block_mapping',
        'window_block_usage',
        'window_block_groups',
        'window_attn_bias',
    ])
    return attention_metadata


def round_up(value: int, k: int):
    return (value + k - 1) // k * k


def get_dp_padding(num_tokens: int, dp_size: int, dp_rank: int) -> int:
    if dp_size == 1:
        return 0

    group = get_dp_group().cpu_group

    num_tokens_across_dp = [0] * dp_size
    num_tokens_across_dp[dp_rank] = num_tokens
    num_tokens_tensor = torch.tensor(num_tokens_across_dp, dtype=torch.int32)
    torch.distributed.all_reduce(num_tokens_tensor, group=group)

    max_tokens_across_dp_cpu = torch.max(num_tokens_tensor).item()
    return max_tokens_across_dp_cpu - num_tokens


class HPUModelRunner(KVConnectorModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device = 'hpu',
        is_driver_worker: bool = False,
    ):
        # TODO: use ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        environment.set_vllm_config(vllm_config)
        finalize_config()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.is_driver_worker = is_driver_worker
        self.use_aux_hidden_state_outputs = False
        self.supports_mm_inputs = False

        self.sampler = Sampler()

        # NOTE(kzawora) update_env is a hack to work around VLLMKVCache in
        # hpu-extension which selects fetch_from_cache implementation based
        # on env vars... this should be fixed in the future
        self.enable_bucketing = get_config().use_bucketing
        self.use_contiguous_pa = get_config().use_contiguous_pa
        self.skip_warmup = get_config().skip_warmup

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        self.is_pooling_model = model_config.pooler_config is not None

        self.sliding_window = model_config.get_sliding_window()
        self.interleaved_sliding_window = is_interleaved(vllm_config.model_config.hf_text_config)
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        # Override settings when profiling a single prefill/decode
        # We can do such barbaric changes because we close vllm after the profiling
        prompt_profile_cfg, decode_profile_cfg = self._read_profiling_cfg()
        if prompt_profile_cfg or decode_profile_cfg:
            self.scheduler_config.max_num_seqs = self.max_model_len
            if prompt_profile_cfg:
                self.scheduler_config.max_num_batched_tokens = prompt_profile_cfg[0] * prompt_profile_cfg[1]
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        # Cached outputs.
        ## universal buffer for input_ids and positions ##
        ## necessary being used by spec decode by following GPU impl ##
        self._draft_token_ids: Optional[Union[list[list[int]], torch.Tensor]] = None
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        ###############################################################

        # Model-related.
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(self.parallel_config, "attention")
        self.num_query_heads = self.model_config.get_num_attention_heads(self.parallel_config)
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.hidden_size = self.model_config.get_hidden_size()
        self.is_pooling_model = model_config.pooler_config is not None
        logger.debug("model config: ", self.model_config)

        self.attn_backend = get_attn_backend(
            self.head_size,
            self.dtype,
            self.kv_cache_dtype,
            self.block_size,
            use_mla=self.model_config.use_mla,
        )

        # Mult-modal-related.
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(model_config)
        if self.supports_mm_inputs:
            self.is_mm_embed = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.is_multimodal_raw_input_supported = (model_config.is_multimodal_raw_input_only_model)

        # Lazy initialization
        # self.model: nn.Module  # set after load_model
        self.kv_caches: list[torch.Tensor] = []
        self.inc_initialized_successfully = False
        self._is_inc_finalized = False

        # mm_hash -> encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}
        # Set up speculative decoding.
        # NOTE(Chendi): Speculative decoding is only enabled for the last rank
        # in the pipeline parallel group.
        if self.speculative_config:
            if self.speculative_config.num_speculative_tokens > 1:
                raise NotImplementedError("Speculative decoding with num_speculative_tokens > 1 is "
                                          "not supported on HPU.")
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                if self.speculative_config.num_speculative_tokens > 1:
                    logger.warning("EagleProposer only supports num_speculative_tokens=1. "
                                   "Overriding the config.")
                    self.speculative_config.num_speculative_tokens = 1
                self.drafter = EagleProposer(self.vllm_config, self.device, self)  # type: ignore
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == "medusa":
                raise NotImplementedError("Medusa speculative decoding is not supported on HPU.")
            else:
                raise ValueError("Unknown speculative decoding method: "
                                 f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler(self.sampler)

        # Keep in int64 to avoid overflow with long context
        self.max_num_reqs = self.scheduler_config.max_num_seqs

        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens), dtype=np.int64)

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.block_size],
            kernel_block_sizes=[self.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(self.vllm_config, self.device, self.pin_memory, self.is_pooling_model,
                                          self.vllm_config.model_config.logits_processors),
        )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        # Cache token ids on device to avoid h2d copies
        self.input_ids_hpu = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=self.device,
            pin_memory=self.pin_memory) if self.use_async_scheduling else None
        self.async_output_copy_stream = torch.hpu.Stream() if \
            self.use_async_scheduling else None
        assert not (self.use_async_scheduling and (self.speculative_config is not None)), \
            "Speculative decoding is not supported with async scheduling."
        self.mem_margin = None
        self.unified_attn = get_config().unified_attn
        self.use_merged_prefill = get_config().merged_prefill

        self.use_hpu_graph = not self.model_config.enforce_eager
        self.max_batch_size = self.scheduler_config.max_num_seqs
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        if prompt_profile_cfg:
            self.max_prefill_batch_size = prompt_profile_cfg[0]
        else:
            self.max_prefill_batch_size = with_default(get_config().VLLM_PROMPT_BS_BUCKET_MAX, 1)
        self.seen_configs: set = set()
        self.max_num_batched_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.use_prefix_caching = (self.vllm_config.cache_config.enable_prefix_caching)
        self.bucketing_manager = HPUBucketingManager()
        max_num_prefill_seqs = self.max_num_seqs if self.use_merged_prefill \
                               else self.max_prefill_batch_size
        if self.enable_bucketing:
            logger.info("Bucketing is ON.")
            self.bucketing_manager.initialize(max_num_seqs=self.max_num_seqs,
                                              max_num_prefill_seqs=max_num_prefill_seqs,
                                              block_size=self.block_size,
                                              max_num_batched_tokens=self.max_num_batched_tokens,
                                              max_model_len=self.max_model_len)
            self.graphed_buckets: set[Any] = set()
            self.graphed_multimodal_buckets: set[Any] = set()
        else:
            logger.info("Bucketing is OFF.")
        self._PAD_SLOT_ID = -1
        self._PAD_BLOCK_ID = -1
        self._tokenizer = init_tokenizer_from_configs(model_config=vllm_config.model_config)

        if self.vllm_config.parallel_config.data_parallel_size > 1 and htorch.utils.internal.is_lazy(
        ) and not self.model_config.enforce_eager:
            from vllm import envs
            # disable device group for dp synchronization when hpu graph is
            # turned on since it's not captured and causes issues
            envs.VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION = True

        self.logits_rounding = 1
        # High-level profiler
        self.profiler = HabanaHighLevelProfiler()
        self.profiler_counter_helper = HabanaProfilerCounterHelper()

        self.defragmenter = OnlineDefragmenter()
        self.debug_fwd = init_debug_logger('fwd')

        self.get_dp_padding = partial(get_dp_padding,
                                      dp_size=self.parallel_config.data_parallel_size,
                                      dp_rank=self.parallel_config.data_parallel_rank)

        assert not (self.unified_attn and not self.use_contiguous_pa), 'Unified attn requires contiguous_pa!'
        assert not (self.unified_attn and not self.use_merged_prefill), 'Unified attn requires merged_prefill!'

    def _make_buffer(self, *size: Union[int, torch.SymInt], dtype: torch.dtype, numpy: bool = True) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size, dtype=dtype, device=self.device, pin_memory=self.pin_memory, with_numpy=numpy)

    def unified_bucketing_fn(self, is_causal, query_len, shared_blocks, unique_blocks, logits):
        if not get_config().use_bucketing:
            return query_len, shared_blocks, unique_blocks, logits

        new_bucket = self.bucketing_manager.find_unified_bucket(query_len, shared_blocks, unique_blocks, is_causal)
        return (new_bucket[0], new_bucket[1], new_bucket[2], self.max_num_seqs)

    def create_lora_mask(self, input_tokens: torch.Tensor, lora_ids: list[int], is_prompt: bool):
        '''
        This is a helper function to create the mask for lora computations.
        Lora Mask is needed to ensure we match the correct lora weights for the
        for the request.
        For Prompt phase we have
        lora_mask with shape (batch_size * seq_len, max_loras * max_rank)
        lora_logits_mask with shape (batch_size, max_loras * max_rank)
        For Decode phase we have both
        lora_mask and lora_logits_mask with shape
        (batch_size, max_loras * max_rank)
        '''
        lora_mask: torch.Tensor = None
        lora_logits_mask: torch.Tensor = None
        lora_index = 0

        if self.lora_config:
            if is_prompt:
                lora_mask = torch.zeros(
                    input_tokens.shape[0] * input_tokens.shape[1],
                    (self.lora_config.max_loras) *\
                        self.lora_config.max_lora_rank,
                    dtype=self.lora_config.lora_dtype)
                lora_logits_mask = torch.zeros(input_tokens.shape[0],
                                               (self.lora_config.max_loras) * self.lora_config.max_lora_rank,
                                               dtype=self.lora_config.lora_dtype)

                ones = torch.ones(input_tokens.shape[1],
                                  self.lora_config.max_lora_rank,
                                  dtype=self.lora_config.lora_dtype)
                logit_ones = torch.ones(1, self.lora_config.max_lora_rank, dtype=self.lora_config.lora_dtype)

                for i in range(len(lora_ids)):
                    if lora_ids[i] == 0:
                        continue
                    lora_index = self.lora_manager._adapter_manager.\
                        lora_index_to_id.index(lora_ids[i])
                    start_row = i * input_tokens.shape[1]
                    end_row = start_row + input_tokens.shape[1]
                    start_col = lora_index * self.lora_config.max_lora_rank
                    end_col = start_col + self.lora_config.max_lora_rank
                    lora_mask[start_row:end_row, start_col:end_col] = ones
                    lora_logits_mask[i, start_col:end_col] = logit_ones
                lora_mask = lora_mask.to('hpu')
                lora_logits_mask = lora_logits_mask.to('hpu')
            else:
                lora_mask = torch.zeros(input_tokens.shape[0],
                                        (self.lora_config.max_loras) * self.lora_config.max_lora_rank,
                                        dtype=self.lora_config.lora_dtype)
                ones = torch.ones(1, self.lora_config.max_lora_rank, dtype=self.lora_config.lora_dtype)
                for i in range(len(lora_ids)):
                    if lora_ids[i] == 0:
                        continue
                    lora_index = self.lora_manager._adapter_manager.\
                        lora_index_to_id.index(lora_ids[i])
                    start_pos = lora_index * self.lora_config.max_lora_rank
                    end_pos = start_pos + self.lora_config.max_lora_rank
                    lora_mask[i, start_pos:end_pos] = ones
                lora_mask = lora_mask.to('hpu')
                lora_logits_mask = lora_mask

        return lora_mask, lora_logits_mask

    def load_lora_model(self, model: nn.Module, vllm_config: VllmConfig, device: str) -> nn.Module:
        if not supports_lora(model):
            raise ValueError(f"{model.__class__.__name__} does not support LoRA yet.")

        if supports_multimodal(model):
            logger.warning("Regarding multimodal models, vLLM currently "
                           "only supports adding LoRA to language model.")

        # Add LoRA Manager to the Model Runner
        self.lora_manager = LRUCacheWorkerLoRAManager(
            vllm_config,
            device,
            model.embedding_modules,
            model.embedding_padding_modules,
        )
        return self.lora_manager.create_lora_manager(model)

    def set_active_loras(self, lora_requests: set[LoRARequest], lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)

    def remove_all_loras(self):
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.remove_all_adapters()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype
        for layer_name, attn_module in forward_ctx.items():
            if isinstance(attn_module, FusedMoE):
                continue

            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention
            if isinstance(attn_module, Attention):
                if attn_module.attn_type == AttentionType.DECODER:
                    kv_cache_spec[layer_name] = FullAttentionSpec(block_size=block_size,
                                                                  num_kv_heads=attn_module.num_kv_heads,
                                                                  head_size=attn_module.head_size,
                                                                  dtype=self.kv_cache_dtype)
                elif attn_module.attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown attention type: {attn_module.attn_type}")
            elif isinstance(attn_module, MLAAttention):
                if layer_name in kv_cache_spec:
                    continue
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    cache_dtype_str=cache_dtype_str,
                )

        return kv_cache_spec

    def _update_states(self, scheduler_output: "SchedulerOutput") -> bool:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: list[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)
            if req_id in self.input_batch.req_type:
                del self.input_batch.req_type[req_id]

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: list[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params
            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None
            if pooling_params:
                assert (task := pooling_params.task) is not None, ("You did not set `task` in the API")

                model = cast(VllmModelForPooling, self.model)
                to_update = model.pooler.get_pooling_updates(task)
                assert to_update is not None, (f"{pooling_params.task=} is not supported by the model")
                to_update.apply(pooling_params)

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                image_grid_thw = []
                video_grid_thw = []
                second_per_grid_ts = []
                audio_feature_lengths = []
                use_audio_in_video = False
                for mm_feature in self.requests[req_id].mm_features:
                    mm_item = mm_feature.data
                    mm_input = mm_item.get_data()
                    if mm_input.get("image_grid_thw") is not None:
                        image_grid_thw.append(mm_input["image_grid_thw"].tolist())
                    if mm_input.get("video_grid_thw") is not None:
                        video_grid_thw.append(mm_input["video_grid_thw"].tolist())
                    if mm_input.get("second_per_grid_ts") is not None:
                        second_per_grid_ts.append(mm_input["second_per_grid_ts"])
                    if mm_input.get("audio_feature_lengths") is not None:
                        audio_feature_lengths.append(mm_input["audio_feature_lengths"])
                    if mm_input.get("use_audio_in_video") is True:
                        use_audio_in_video = True

                hf_config = self.model_config.hf_config

                self.requests[req_id].mrope_positions, \
                    self.requests[req_id].mrope_position_delta = \
                    self.model.model.get_mrope_input_positions(
                        self.requests[req_id].prompt_token_ids,
                        hf_config=hf_config,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        second_per_grid_ts=second_per_grid_ts,
                        audio_feature_lengths=audio_feature_lengths,
                        use_audio_in_video=use_audio_in_video,
                )

            req_ids_to_add.append(req_id)
        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) - req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (num_computed_tokens)
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[req_index, start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                # NOTE(woosuk): `num_tokens` here may include spec decode tokens
                self.input_batch.num_tokens[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = \
                scheduler_output.scheduled_spec_decode_tokens.get(
                    req_id, ())
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            req_index = removed_req_indices.pop() if removed_req_indices else None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()
        return batch_changed

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if self.is_multimodal_raw_input_supported:  # noqa: SIM102
            if scheduler_output:
                mm_kwargs = list[MultiModalKwargsItem]()
                for req in scheduler_output.scheduled_new_reqs:
                    req_mm_kwargs = req.mm_kwargs
                    if not isinstance(req_mm_kwargs, list):
                        req_mm_kwargs = list(req_mm_kwargs)
                    mm_kwargs.extend(req_mm_kwargs)

                # Input all modalities at once
                self.model.model = cast(SupportsMultiModal, self.model.model)
                mm_kwargs_combined: BatchedTensorInputs = {}
                for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                        mm_kwargs,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        merge_by_field_config=self.model.model.merge_by_field_config,
                ):
                    mm_kwargs_combined.update(mm_kwargs_group)

                return mm_kwargs_combined

        return {}

    # source: vllm/v1/worker/gpu_model_runner.py
    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput", req_ids: list[str]):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # List of tuple (mm_hash, pos_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id in req_ids:
            encoder_input_ids = scheduled_encoder_inputs.get(req_id, None)
            if not encoder_input_ids:
                continue
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        if not mm_kwargs:
            return

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.

        # TODO (attafosu): Follow-up on the resolution to this.
        # The ordering of the encoder outputs needs to match the request ids
        # after fetching the embeddings.
        # For now, we'll restrict mm support to just a single prefill at a time - # noqa E501
        # Or that requests in the batch should have distinct modalities,

        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        encoder_outputs = []
        self.model.model = cast(SupportsMultiModal, self.model.model)
        for _, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=self.model.model.merge_by_field_config,
        ):
            # Run the encoder.
            # `curr_group_outputs` is either of the following:
            # 1. A tensor of shape (num_items, feature_size, hidden_size)
            # in case feature_size is fixed across all multimodal items.
            # 2. A list or tuple (length: num_items) of tensors, each of shape
            # (feature_size, hidden_size) in case the feature size is dynamic
            # depending on the input multimodal items.
            curr_group_outputs = self.model.get_multimodal_embeddings(**mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )

            for output in curr_group_outputs:
                encoder_outputs.append(output)

        # FIXME (attafosu) Reorder the encoder outputs to match the request ids.
        # This will be necessary after mm prefill batching constraints are removed # noqa E501

        # Cache the encoder outputs.
        for (mm_hash, pos_info), output in zip(
                mm_hashes_pos,
                encoder_outputs,
        ):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}

            self.encoder_cache[mm_hash] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed.to(
                    device=output.device) if pos_info.is_embed is not None else pos_info.is_embed,
            )

    # modified from: vllm/v1/worker/gpu_model_runner.py
    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        req_ids: list[str],
        shift_computed_tokens: int = 0,
        total_num_scheduled_tokens: Optional[int] = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = total_num_scheduled_tokens or scheduler_output.total_num_scheduled_tokens

        mm_embeds = list[torch.Tensor]()
        is_mm_embed = self.is_mm_embed.cpu
        is_mm_embed[:total_num_scheduled_tokens] = False

        req_start_idx = 0
        for req_id in req_ids:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = \
                req_state.num_computed_tokens + shift_computed_tokens
            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(num_computed_tokens - start_pos + num_scheduled_tokens, num_encoder_tokens)
                assert start_idx < end_idx
                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None,\
                      f"Encoder cache miss for {mm_hash}."
                encoder_output = self.encoder_cache[mm_hash]

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                is_mm_embed[req_start_pos+start_idx:req_start_pos + end_idx] \
                    = True

                # Only whole mm items are processed
                mm_embeds.append(mm_embeds_item)
            req_start_idx += num_scheduled_tokens

        is_mm_embed = self.is_mm_embed.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> torch.nn.Module:
        if isinstance(self.model, HpuModelAdapter):
            return self.model.model
        assert self.model is not None
        return self.model

    def is_decoder_only(self, req_id) -> bool:
        return bool(req_id in self.input_batch.req_type and \
            self.input_batch.req_type[req_id] == "decode")

    def _get_prompts_and_decodes(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> PromptDecodeInfo:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        #TODO: remove later

        requests_type = {}
        if scheduler_output.kv_connector_metadata:
            for req in scheduler_output.kv_connector_metadata.reqs_to_save:
                requests_type[req] = 'prefill'
            for req in scheduler_output.kv_connector_metadata.reqs_to_recv:
                requests_type[req] = 'decode'
            requests = scheduler_output.kv_connector_metadata.reqs_to_save | \
                        scheduler_output.kv_connector_metadata.reqs_to_recv
        else:
            requests = None

        # Traverse decodes first
        decode_req_ids = []
        num_computed_tokens_decode = []
        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None
            # P case assigment
            if requests is not None and req_id not in self.input_batch.req_type:
                for request in requests:
                    if request == req_id:
                        self.input_batch.req_type[req_id] = requests_type[req_id]
                        break

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]

            if num_computed_tokens < num_prompt_tokens and \
                not self.is_decoder_only(req_id):
                # This is prompt
                break

            # This is decode
            # NOTE(chendi): To support spec decode,
            # we don't assume num_scheduled_tokens == 1.

            decode_req_ids.append(req_id)
            num_computed_tokens_decode.append(int(num_computed_tokens + 1))

        if self.profiler.enabled:
            self.profiler_counter_helper.capture_decode_seq_stats(num_computed_tokens_decode)

        # Traverse prompts
        prompt_req_ids = []
        prompt_scheduled_tokens = []
        for i in range(len(decode_req_ids), num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt_tokens = self.input_batch.num_prompt_tokens[i]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Must be prompt
            assert num_computed_tokens < num_prompt_tokens
            num_output_tokens = len(self.requests[req_id].output_token_ids)
            if not has_kv_transfer_group():
                #P case num_output_tokens has non 0
                assert num_output_tokens == 0, \
                    f'req_id: {req_id}, {num_output_tokens}'

            prompt_req_ids.append(req_id)
            prompt_scheduled_tokens.append(num_scheduled_tokens)

        return PromptDecodeInfo(prompt_req_ids, decode_req_ids, prompt_scheduled_tokens)

    def _generate_req_id_output_token_ids_lst(self,
                                              request_ids: Optional[list[str]] = None,
                                              pad_to: Optional[int] = None,
                                              logits_reqs=None):
        req_id_output_token_ids: dict[str, list[int]] = \
            {req_id: req.output_token_ids
                for req_id, req in self.requests.items()}
        if request_ids is not None:
            req_id_output_token_ids = {req_id: req_id_output_token_ids[req_id] for req_id in request_ids}
        req_id_output_token_ids_lst = list(req_id_output_token_ids.items())
        if logits_reqs and len(req_id_output_token_ids_lst) > len(logits_reqs):
            # Merged prefill case: remove requests without logits
            req_id_output_token_ids_lst = [r for r in req_id_output_token_ids_lst if r[0] in logits_reqs]
        else:
            if pad_to is not None and len(req_id_output_token_ids_lst) > 0:
                while len(req_id_output_token_ids_lst) < pad_to:
                    req_id_output_token_ids_lst.append(req_id_output_token_ids_lst[0])
        return req_id_output_token_ids_lst

    def _prepare_sampling(self,
                          batch_changed: bool,
                          request_ids: Union[None, list[str]] = None,
                          pad_to: Optional[int] = None,
                          logits_reqs=None) -> SamplingMetadata:
        # Create the sampling metadata.
        req_id_output_token_ids_lst = \
            self._generate_req_id_output_token_ids_lst(request_ids, \
                                                       pad_to, logits_reqs)
        sampling_metadata = self.input_batch.make_selective_sampling_metadata(req_id_output_token_ids_lst,
                                                                              skip_copy=not batch_changed)
        return sampling_metadata

    def get_habana_paged_attn_buffers(self, block_tables, slot_mapping, batch_size):
        last_block_usage = [slot[0] % self.block_size + 1 for slot in slot_mapping]
        block_groups = [[i] * len(bt) for i, bt in enumerate(block_tables)]
        block_usage = [[self.block_size] * (len(bt) - 1) + [lbu] for bt, lbu in zip(block_tables, last_block_usage)
                       if bt]
        block_list = flatten(block_tables)
        block_groups = flatten(block_groups)
        block_usage = flatten(block_usage)
        assert len(block_list) == len(block_groups)
        assert len(block_list) == len(block_usage)

        padding_fn = None
        block_bucket_size: int
        if self.use_contiguous_pa:
            block_bucket_size = max(max(block_list) + 1, len(block_list))
            block_bucket_size = \
                self.bucketing_manager.find_decode_bucket(batch_size,
                                                          block_bucket_size)[2]
            block_bucket_size += self.get_dp_padding(block_bucket_size)

            indices: list[Any]
            indices = [None] * block_bucket_size
            for i, bid in enumerate(block_list):
                indices[bid] = i

            def padding_fn(tensor, pad_value):
                return gather_list(tensor, indices, pad_value)
        else:
            block_bucket_size = \
                self.bucketing_manager.find_decode_bucket(batch_size,
                                                          len(block_list))[2]
            block_bucket_size += self.get_dp_padding(block_bucket_size)

            def padding_fn(tensor, pad_value):
                return pad_list(tensor, block_bucket_size, itertools.repeat(pad_value))

        block_list = padding_fn(block_list, self._PAD_BLOCK_ID)
        block_groups = padding_fn(block_groups, -1)
        block_usage = padding_fn(block_usage, 1)

        block_list = torch.tensor(block_list, dtype=torch.long, device='cpu')
        block_groups = torch.tensor(block_groups, dtype=torch.long, device='cpu')
        block_usage = torch.tensor(block_usage, dtype=self.model_config.dtype, device='cpu')
        return block_list, block_groups, block_usage

    def _align_and_pad_mrope_positions(self, req_ids: list[str], context_lens: list[int], query_lens: list[int],
                                       bucketing: tuple[int, int], padding_gen: int) -> torch.Tensor:
        target_bs, target_len = bucketing
        out_shape = (3, target_len) if target_bs == 1 \
            else (target_bs, target_len)

        mrope_position_tensor = torch.full(out_shape, padding_gen, dtype=torch.int32, device='cpu')
        dst_start = 0
        dst_end = dst_start
        for b_idx, req_id in enumerate(req_ids):
            cl = context_lens[b_idx]
            qsl = query_lens[b_idx]
            assert self.requests[req_id].mrope_positions is not None
            input_mrope_position = \
                self.requests[req_id].mrope_positions[:, cl:cl + qsl] # type: ignore[index]
            dst_end = dst_start + qsl
            mrope_position_tensor[:, dst_start:dst_end].copy_(input_mrope_position, non_blocking=True)

            # Update dst_start depending on if pos_ids of requests are meant to be adjacent # noqa 501
            if target_bs == 1:
                dst_start = dst_end
            else:
                dst_start += target_len
        return mrope_position_tensor

    def _skip_bucketing(self, seq_lens, num_blocks):
        return (len(seq_lens), 0, 0)

    def _bucketize_merged_prompt(self, seq_lens, num_blocks):
        seq = sum(seq_lens)
        num_blocks = sum(num_blocks)
        seq = self.bucketing_manager.find_prompt_bucket(1, seq, num_blocks)[1]
        num_blocks = round_up(num_blocks, 32)
        return (1, seq, num_blocks)

    def _bucketize_2d_prompt(self, seq_lens, num_blocks):
        bs = len(seq_lens)
        if bs > self.max_prefill_batch_size:
            raise BucketingFailedException
        seq = max(seq_lens)
        num_blocks = max(num_blocks) if len(num_blocks) > 0 else 0
        bs, seq, num_blocks = self.bucketing_manager.find_prompt_bucket(bs, seq, num_blocks)
        return (bs, seq, num_blocks)

    def _get_prompt_bucketing_fn(self):
        if self.unified_attn:
            return self._skip_bucketing
        elif self.use_merged_prefill:
            return self._bucketize_merged_prompt
        else:
            return self._bucketize_2d_prompt

    def _can_merge_prefill_contents(self, lhs, rhs):
        combined_num_tokens = lhs.get_num_tokens() + rhs.get_num_tokens()
        bucketing_fn = self._get_prompt_bucketing_fn()
        try:
            target_bs, target_seq, target_blocks = bucketing_fn(combined_num_tokens, [])
        except BucketingFailedException:
            return False
        target_bs, target_seq, target_blocks = bucketing_fn(combined_num_tokens, [])
        return target_bs <= self.max_prefill_batch_size and\
            target_bs * target_seq <= self.max_num_tokens

    def _extract_prefill_batch_contents(self, num_prefills, num_decodes, num_scheduled_tokens, warmup=False):
        # DECODES are the first num_decodes REQUESTS.
        # PREFILLS are the next num_reqs - num_decodes REQUESTS.
        num_reqs = num_prefills + num_decodes
        block_table_cpu_tensor = self.input_batch.block_table[0].get_cpu_tensor()
        all_batch_contents = [BatchContents()]

        for batch_idx in range(num_decodes, num_reqs):
            req_id = self.input_batch.req_ids[batch_idx]
            context_len = self.input_batch.num_computed_tokens_cpu[batch_idx]
            query_len = num_scheduled_tokens[batch_idx]

            token_ids = self.input_batch.token_ids_cpu[batch_idx, context_len:context_len + query_len].tolist()

            num_blocks = round_up(context_len + query_len, self.block_size) // self.block_size
            blocks = block_table_cpu_tensor[batch_idx, :num_blocks].tolist()
            if not warmup:
                blocks = [self.defragmenter.resolve(b) for b in blocks]

            prompt_tokens = self.input_batch.num_prompt_tokens[batch_idx]
            # TODO: Fix non-prompt case
            num_output_logits = max(0, context_len + query_len - prompt_tokens + 1)
            logits_positions = list(range(query_len - num_output_logits, query_len))

            new_batch_contents = BatchContents(
                req_ids=[req_id],
                token_ids=[token_ids],
                context_lens=[context_len],
                prompt_lens=[prompt_tokens],
                blocks=[blocks],
                logits_positions=[logits_positions],
            )
            if self._can_merge_prefill_contents(all_batch_contents[-1], new_batch_contents):
                merge_contents(all_batch_contents[-1], new_batch_contents)
            else:
                all_batch_contents.append(new_batch_contents)

        num_real_prefill_batches = 0
        for content in all_batch_contents:
            if len(content.req_ids) > 0:
                num_real_prefill_batches += 1

        num_pad_across_dp = self.get_dp_padding(num_real_prefill_batches)
        return all_batch_contents, num_pad_across_dp

    def _make_attn_bias(self, context_groups, token_groups):
        dtype = self.dtype
        is_causal = True  # TODO: add support for non-causal tasks
        context_groups = torch.tensor(context_groups, device='cpu', dtype=torch.int16)
        context_groups = context_groups.repeat_interleave(self.block_size, dim=-1)
        context_len = context_groups.size(-1)
        token_groups = torch.tensor(token_groups, device='cpu', dtype=torch.int16)
        num_queries = token_groups.size(-1)
        seq_groups = torch.cat([context_groups, token_groups], dim=-1)
        attn_mask = seq_groups.unflatten(-1, (1, -1)) != token_groups.unflatten(-1, (-1, 1))
        if is_causal:
            causal_mask = torch.ones(num_queries, num_queries, device='cpu', dtype=torch.bool)
            causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0)
            attn_mask[:, :, context_len:].logical_or_(causal_mask)
        attn_mask = attn_mask.to(dtype).masked_fill_(attn_mask, -math.inf)

        return attn_mask.unflatten(0, (1, -1))

    def _form_prefill_batch(self, contents):
        if len(contents.req_ids) == 0:
            return PrefillInputData()

        token_ids = contents.token_ids
        req_ids = contents.req_ids
        query_lens = [len(tids) for tids in contents.token_ids]
        if self.profiler.enabled:
            self.profiler_counter_helper.capture_prompt_seq_stats(query_lens)
        context_lens = contents.context_lens

        token_positions = [list(range(cl, cl + ql)) for cl, ql in zip(context_lens, query_lens)]

        block_assignment = [[divmod(pos, self.block_size) for pos in positions] for positions in token_positions]

        token_slots = [[blocks[bi] * self.block_size + bo for bi, bo in assignment]
                       for blocks, assignment in zip(contents.blocks, block_assignment)]
        token_groups = [[i] * len(tid) for i, tid in enumerate(token_ids)]
        num_context_blocks = [round_up(ctx_len, self.block_size) // self.block_size for ctx_len in context_lens]
        context_blocks: list = [blocks[:num] for blocks, num in zip(contents.blocks, num_context_blocks)]
        num_context_blocks = [len(b) for b in context_blocks]
        context_groups = [[i] * b for i, b in enumerate(num_context_blocks)]
        has_context = sum(context_lens) > 0
        target_bs, target_seq, target_blocks = self._get_prompt_bucketing_fn()(query_lens, num_context_blocks)

        target_bs += self.get_dp_padding(target_bs)
        target_seq += self.get_dp_padding(target_seq)
        target_blocks += self.get_dp_padding(target_blocks)

        # NOTE: If model does not support multimodal inputs, we pad here.
        # For models with multimodal support, we may want to get embeddings
        # for the valid tokens before padding.
        # This would require getting multimodal input embeddings here as well
        token_ids = align_and_pad(contents.token_ids, (target_bs, target_seq), itertools.repeat(-1))
        # Update query_lens and context_lens after padding
        query_lens.extend([0] * (target_bs - len(query_lens)))
        context_lens.extend([0] * (target_bs - len(context_lens)))

        # If the model uses M-RoPE, we need to fill
        # and pad the M-RoPE positions for the scheduled prefill tokens
        if self.uses_mrope:
            token_positions = self._align_and_pad_mrope_positions(
                contents.req_ids,
                context_lens,
                query_lens,
                (target_bs, target_seq),
                -1,
            )

        else:
            token_positions = align_and_pad(token_positions, (target_bs, target_seq), itertools.repeat(-1))
        token_slots = align_and_pad(token_slots, (target_bs, target_seq), itertools.repeat(-1))
        token_groups = align_and_pad(token_groups, (target_bs, target_seq), itertools.repeat(-1))
        context_blocks = align_and_pad(context_blocks, (target_bs, target_blocks), itertools.repeat(-1))
        context_groups = align_and_pad(context_groups, (target_bs, target_blocks), itertools.repeat(-1))

        # TODO: cycle through dummy slots and blocks
        # dummy_slots = itertools.cycle(
        #    range(self._PAD_SLOT_ID, self._PAD_SLOT_ID + self.block_size))

        cur_offset = 0
        logits_indices = []
        logits_requests = []
        for req_id, qlen, log_pos in zip(req_ids, query_lens, contents.logits_positions):
            source = [cur_offset + x for x in log_pos]
            dest = [req_id] * len(log_pos)
            logits_indices.extend(source)
            logits_requests.extend(dest)
            if self.use_merged_prefill:
                cur_offset += qlen
            else:
                cur_offset += len(token_ids[0])

        attn_bias = None
        if self.use_merged_prefill:
            attn_bias = self._make_attn_bias(context_groups, token_groups)
            attn_bias = attn_bias.to('hpu', non_blocking=True)
        else:
            attn_bias = None

        logits_indices = pad_list(logits_indices, round_up(len(logits_indices), self.logits_rounding),
                                  itertools.repeat(-1))

        query_lens = async_h2d_copy(query_lens, dtype=torch.int32)
        token_ids = async_h2d_copy(token_ids, dtype=torch.int32)
        token_positions = async_h2d_copy(token_positions, dtype=torch.int32)
        token_slots = async_h2d_copy(token_slots, dtype=torch.int64)
        logits_indices = async_h2d_copy(logits_indices, dtype=torch.int32)
        context_lens = async_h2d_copy(context_lens, dtype=torch.int32)
        context_blocks_t: Optional[torch.tensor]
        context_blocks_t = async_h2d_copy(context_blocks, dtype=torch.int32).flatten() if has_context else None

        attn_metadata = HPUAttentionMetadataV1.make_prefill_metadata(seq_lens_tensor=query_lens,
                                                                     context_lens_tensor=context_lens,
                                                                     slot_mapping=token_slots,
                                                                     block_list=context_blocks_t,
                                                                     attn_bias=attn_bias,
                                                                     block_size=self.block_size)
        return PrefillInputData(request_ids=[req_ids],
                                prompt_lens=[query_lens],
                                token_ids=[token_ids],
                                position_ids=[token_positions],
                                attn_metadata=[attn_metadata],
                                logits_indices=[logits_indices],
                                logits_requests=[logits_requests])

    def _form_unified_prefill_batch(self, contents):
        if len(contents.req_ids) == 0:
            return PrefillInputData()

        token_ids = contents.token_ids
        req_ids = contents.req_ids
        query_lens = [len(tids) for tids in contents.token_ids]
        prompt_lens = contents.prompt_lens
        if self.profiler.enabled:
            self.profiler_counter_helper.capture_prompt_seq_stats(query_lens)
        context_lens = contents.context_lens

        batch_data = create_unified_batch(
            token_ids=token_ids,
            block_size=self.block_size,
            block_table=contents.blocks,
            context_lengths=context_lens,
            query_lengths=query_lens,
            prompt_lengths=prompt_lens,
            dtype=self.dtype,
            contiguous_kv=self.use_contiguous_pa,
            bucketing_fn=self.unified_bucketing_fn,
            get_dp_padding_fn=self.get_dp_padding,
        )

        (token_ids_t, token_positions_t, logits_indices_t, logits_groups, attn_metadata) = batch_data
        logits_requests = [req_ids[lg] for lg in logits_groups]
        return PrefillInputData(request_ids=[req_ids],
                                prompt_lens=[None],
                                token_ids=[token_ids_t.unsqueeze(0)],
                                attn_metadata=[attn_metadata],
                                position_ids=[token_positions_t.unsqueeze(0)],
                                logits_indices=[logits_indices_t],
                                logits_requests=[logits_requests])

    def _create_dummy_prefill_batch_contents(self, num_prefills: int) -> list[PrefillInputData]:
        req_id = str(-1)
        context_len = 0
        query_len = 128
        prompt_tokens = 128
        token_ids = list(int(i) for i in range(prompt_tokens))
        num_blocks = round_up(context_len + query_len, self.block_size) // self.block_size
        blocks = [0] * num_blocks
        num_output_logits = context_len + query_len - prompt_tokens + 1
        logits_positions = list(range(query_len - num_output_logits, query_len))

        new_batch_contents = BatchContents(
            req_ids=[req_id],
            token_ids=[token_ids],
            context_lens=[context_len],
            blocks=[blocks],
            logits_positions=[logits_positions],
        )

        outputs = [self._form_prefill_batch(new_batch_contents.clone()) for _ in range(num_prefills)]
        return outputs

    def _prepare_prefill_inputs(self, num_prefills, num_decodes,
                                num_scheduled_tokens: list[int]) -> tuple[PrefillInputData, Optional[PrefillInputData]]:
        all_batch_contents, num_pad_across_dp = \
            self._extract_prefill_batch_contents(
                num_prefills, num_decodes, num_scheduled_tokens)
        all_batches = [self._form_prefill_batch(bc) for bc in all_batch_contents]
        merge_contents(all_batches[0], *all_batches[1:])

        dummy_prefill_input_batches = None
        if num_pad_across_dp > 0:
            dummy_prefill_input_batches = \
                self._create_dummy_prefill_batch_contents(num_pad_across_dp)
            merge_contents(dummy_prefill_input_batches[0], *dummy_prefill_input_batches[1:])
        return all_batches[0], dummy_prefill_input_batches[0] if dummy_prefill_input_batches else None

    def _prepare_unified_prefill_inputs(self,
                                        num_prefills,
                                        num_decodes,
                                        num_scheduled_tokens: list[int],
                                        warmup=False) -> tuple[PrefillInputData, None]:

        all_batch_contents, _ = self._extract_prefill_batch_contents(num_prefills, num_decodes, num_scheduled_tokens,
                                                                     warmup)
        all_batches = [self._form_unified_prefill_batch(bc) for bc in all_batch_contents]
        merge_contents(all_batches[0], *all_batches[1:])
        return all_batches[0], None

    def _create_decode_input_data(self,
                                  num_decodes,
                                  num_scheduled_tokens,
                                  context_lens,
                                  block_table_cpu_tensor,
                                  scheduler_output=None) -> DecodeInputData:
        # NOTE(kzawora): the +1 is what causes this entire thing to work,
        # as in the paged attention, we don't fetch just the context from cache,
        # but also kvs for the current token
        num_blocks = np.ceil((context_lens + 1) / self.block_size).astype(np.int32).tolist()

        # PAD FOR STATIC SHAPES.
        padded_batch_size: int
        padded_batch_size = self.bucketing_manager.find_decode_bucket(num_decodes, sum(num_blocks))[0]

        # dp aware padding
        padded_batch_size += self.get_dp_padding(padded_batch_size)

        num_tokens_per_req = num_scheduled_tokens[:num_decodes]
        num_tokens = max(num_tokens_per_req)
        total_num_scheduled_tokens = sum(num_tokens_per_req)
        num_tokens_per_req = num_tokens_per_req + [0] * (padded_batch_size - num_decodes)

        block_tables_list = []
        for i, n in enumerate(num_blocks):
            seq_block_table = block_table_cpu_tensor[i, :n].tolist()
            assert len(seq_block_table) == n
            block_tables_list.extend([seq_block_table] * num_tokens)

        ###################################
        # initialize positions with padding
        # POSITIONS. [batch, num_tokens]
        # NOTE(Chendi): Follow GPU_Model_Runner to use global
        # self.positions_cpu, which updated in prepare_inputs from
        # self.input_batch.num_computed_tokens_cpu[req_indices]
        positions = torch.zeros((padded_batch_size, num_tokens), dtype=torch.int32)
        if num_tokens == 1:
            positions[:num_decodes] = self.positions_cpu[:num_decodes].view(-1, 1)
        else:
            # per request using universal self.positions_cpu then pad
            position_split_tensors = torch.split(self.positions_cpu[:total_num_scheduled_tokens], num_tokens_per_req)
            positions[:num_decodes] = \
                pad_sequence(list(position_split_tensors),
                                batch_first=True,
                                padding_value=0)[:num_decodes]

        padded_index = torch.zeros((padded_batch_size, num_tokens), dtype=torch.int64)
        index = positions.to(torch.int64)[:num_decodes]
        padded_index[:num_decodes] = index

        input_mrope_positions_list: list[list[int]] = [[] for _ in range(3)]
        if self.uses_mrope:
            for idx, req_id in enumerate(self.input_batch.req_ids[:num_decodes]):
                seq_data = self.requests[req_id]
                context_len = context_lens[idx]
                position = context_len
                if seq_data.mrope_position_delta is not None:
                    seq_data.mrope_position_delta = int(seq_data.mrope_position_delta)
                    pos_for_mrope = MRotaryEmbedding \
                        .get_next_input_positions(
                            seq_data.mrope_position_delta,
                            context_len=context_len,
                            seq_len=context_len + 1)
                else:
                    pos_for_mrope = [[position]] * 3
                for idx in range(3):
                    input_mrope_positions_list[idx].extend(pos_for_mrope[idx])

            positions = torch.tensor(input_mrope_positions_list, dtype=torch.int32, device='cpu')

            # Pad the right side of input_mrope_positions by padded_batch_size
            pad_size = padded_batch_size - positions.size(1)
            if pad_size > 0:
                positions = F.pad(positions, (0, pad_size), value=-1, mode='constant')

        ###################################
        # initialize token_ids with padding
        # TOKEN_IDS. [batch, num_tokens]
        # NOTE(Chendi): Follow GPU_Model_Runner to use global
        # self.input_ids_cpu, which updated in prepare_inputs from
        # self.input_batch.token_ids_cpu[:total_num_scheduled_tokens]
        token_ids = torch.zeros((padded_batch_size, num_tokens), dtype=torch.int32)
        if num_tokens == 1:
            token_ids[:num_decodes] = self.input_ids_cpu[:num_decodes].view(-1, 1)
        else:
            token_ids_split_tensors = torch.split(self.input_ids_cpu[:total_num_scheduled_tokens], num_tokens_per_req)
            token_ids[:num_decodes] = \
                pad_sequence(list(token_ids_split_tensors),
                                batch_first=True,
                                padding_value=0)[:num_decodes]

        ###################################
        # SLOT_MAPPING [batch, 1]
        # The "slot" is the "physical index" of a token in the KV cache.
        # Look up the block_idx in the block table (logical<>physical map)
        # to compute this.
        block_number = torch.ones((padded_batch_size, num_tokens), dtype=torch.int32) * self._PAD_BLOCK_ID
        block_number[:num_decodes] = torch.gather(input=block_table_cpu_tensor, dim=1, index=(index // self.block_size))
        block_number.apply_(self.defragmenter.resolve)

        block_offsets = padded_index % self.block_size
        slot_mapping = block_number * self.block_size + block_offsets
        # set an out of range value for the padding tokens so that they
        # are ignored when inserting into the KV cache.
        slot_mapping = slot_mapping[:padded_batch_size]
        dummy_slots = itertools.cycle(range(self._PAD_SLOT_ID, self._PAD_SLOT_ID + self.block_size))
        slot_mapping[num_decodes:].apply_(lambda _, ds=dummy_slots: next(ds))

        #####################################
        # NOTE(Chendi): Since we can't actually do num_tokens = 2,
        # convert to [batch_size * num_tokens, 1]
        if num_tokens > 1:
            token_ids = token_ids.view(-1, 1)
            positions = padded_index.view(-1, 1)
            slot_mapping = slot_mapping.view(-1, 1)

        logits_indices = torch.zeros(padded_batch_size, dtype=torch.int32, device='cpu')

        # NOTE(Chendi): num_tokens might be > 1 in spec decode case,
        # example:
        # num_scheduled_tokens = [2, 1, 2, 1]
        # padded tokens_id = \
        #     [[tok_0, tok_1], [tok_2, pad], [tok_4, tok_4], [tok_6, pad]]
        # num_tokens = 2
        # query_start_loc_list = [2, 3, 6, 7]
        # query_start_loc_cpu = [0, 2, 3, 6, 7]
        # logits_indices = [1, 2, 5, 6] => the last token of each request
        query_start_loc_list = [i * num_tokens + n for i, n in enumerate(num_scheduled_tokens[:num_decodes])]
        query_start_loc_cpu = torch.empty((padded_batch_size + 1, ),
                                          dtype=torch.int32,
                                          device="cpu",
                                          pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc_cpu.numpy()
        query_start_loc_np[0] = 0
        query_start_loc_np[1:num_decodes + 1] = np.array(query_start_loc_list)

        logits_indices[:num_decodes] = query_start_loc_cpu[1:num_decodes + 1] - 1

        positions_device = async_h2d_copy(positions, device=self.device)
        block_tables_list = self.defragmenter.resolve_all(block_tables_list)

        # CONTEXT_LENS [batch_size]
        block_list, block_groups, block_usage = \
            self.get_habana_paged_attn_buffers(
                block_tables_list,
                slot_mapping.tolist(),
                padded_batch_size * num_tokens
            )

        if self.interleaved_sliding_window and self.sliding_window > 0:
            sliding_block_size = (self.sliding_window // self.block_size)
            window_block_tables = [block_table[-sliding_block_size:] for block_table in block_tables_list]
            window_block_list, window_block_groups, window_block_usage = \
                self.get_habana_paged_attn_buffers(
                    window_block_tables, slot_mapping.tolist(),
                    padded_batch_size * num_tokens)

        # CPU<>HPU sync *should not* happen here.
        block_list_device = async_h2d_copy(block_list, device=self.device)
        block_usage_device = async_h2d_copy(block_usage, device=self.device)
        block_groups_device = async_h2d_copy(block_groups, device=self.device)
        slot_mapping_device = async_h2d_copy(slot_mapping, device=self.device)
        window_block_list_device = async_h2d_copy(window_block_list,
                                                  device=self.device) if self.interleaved_sliding_window else None
        window_block_usage_device = async_h2d_copy(window_block_usage,
                                                   device=self.device) if self.interleaved_sliding_window else None
        window_block_groups_device = async_h2d_copy(window_block_groups,
                                                    device=self.device) if self.interleaved_sliding_window else None

        token_ids_device = async_h2d_copy(token_ids, device=self.device)
        # when DP also enabled, some DP ranks will exeucte dummy run with empty
        # SchedulerOutput, in this case we need skip the prepare_input_ids
        if self.use_async_scheduling and scheduler_output is not None:
            self._prepare_input_ids(scheduler_output)
            if num_tokens == 1:
                token_ids_device[:num_decodes] = self.input_ids_hpu[:num_decodes].view(-1, 1)
            else:
                token_ids_split_tensors = torch.split(self.input_ids_hpu[:total_num_scheduled_tokens],
                                                      num_tokens_per_req)
                token_ids_device[:num_decodes] = \
                    pad_sequence(list(token_ids_split_tensors),
                                    batch_first=True,
                                    padding_value=0)[:num_decodes]

            #####################################
            # NOTE(Chendi): Since we can't actually do num_tokens = 2,
            # convert to [batch_size * num_tokens, 1]
            if num_tokens > 1:
                token_ids_device = token_ids_device.view(-1, 1)

        # call prepare_spec_decode_inputs to get the logits indices and
        if scheduler_output is not None:
            logits_indices, spec_decode_metadata = self._prepare_spec_decode_inputs(scheduler_output, logits_indices,
                                                                                    token_ids_device, num_tokens)
        else:
            spec_decode_metadata = None
        logits_indices_device = async_h2d_copy(logits_indices, device=self.device)

        return DecodeInputData(num_decodes=num_decodes,
                               token_ids=token_ids_device,
                               position_ids=positions_device,
                               logits_indices=logits_indices_device,
                               attn_metadata=HPUAttentionMetadataV1.make_decode_metadata(
                                   block_list=block_list_device,
                                   block_usage=block_usage_device,
                                   block_groups=block_groups_device,
                                   input_positions=None,
                                   slot_mapping=slot_mapping_device,
                                   block_size=self.block_size,
                                   window_block_list=window_block_list_device,
                                   window_block_usage=window_block_usage_device,
                                   window_block_groups=window_block_groups_device,
                               ),
                               spec_decode_metadata=spec_decode_metadata)

    def _prepare_decode_inputs(self,
                               num_decodes,
                               num_scheduled_tokens,
                               scheduler_output=None) -> tuple[DecodeInputData, Optional[DecodeInputData]]:
        # Decodes run as one single padded batch with shape [batch, 1]
        #
        # We need to set _PAD_SLOT_ID for the padding tokens in the
        # slot_mapping, such that the attention KV cache insertion
        # logic knows to ignore those indicies. Otherwise, the
        # padding data can be dummy since we have a causal mask.

        num_pad_across_dp = self.get_dp_padding(num_decodes)
        if num_decodes == 0:
            if num_pad_across_dp > 0:
                dummy_decode_input_data = self._create_dummy_decode_input_data()
                return DecodeInputData(num_decodes=0), dummy_decode_input_data
            return DecodeInputData(num_decodes=0), None
        return self._create_decode_input_data(num_decodes, num_scheduled_tokens,
                                              self.input_batch.num_computed_tokens_cpu[:num_decodes],
                                              self.input_batch.block_table[0].get_cpu_tensor(), scheduler_output), None

    def _create_dummy_decode_input_data(self) -> DecodeInputData:
        # create dummy decode input data with batch size 1
        num_dummy_decodes = 1
        num_dummy_scheduled_tokens = [1]
        context_lens = np.array([128])
        block_table_cpu_tensor = torch.zeros([self._PAD_BLOCK_ID], dtype=torch.int32).reshape(1, -1)
        return self._create_decode_input_data(num_dummy_decodes, num_dummy_scheduled_tokens, context_lens,
                                              block_table_cpu_tensor)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_spec_decode_inputs(self, scheduler_output, logits_indices, token_ids_device, max_num_sampled_tokens):
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(logits_indices.numel(), dtype=np.int32)
            for req_id, draft_token_ids_in_req in (scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids_in_req)

            num_sampled_tokens = num_draft_tokens + 1

            cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(num_sampled_tokens, cumsum_dtype=np.int32)

            logits_indices = []
            bonus_logits_indices = []
            target_logits_indices = []
            for batch_id, n_tokens in enumerate(num_sampled_tokens):
                for i in range(n_tokens - 1):
                    logits_indices.append(batch_id * max_num_sampled_tokens + i)
                    target_logits_indices.append(batch_id * max_num_sampled_tokens + i)
                bonus_logits_indices.append(batch_id * max_num_sampled_tokens + n_tokens - 1)
                logits_indices.append(batch_id * max_num_sampled_tokens + n_tokens - 1)
                if n_tokens < max_num_sampled_tokens:
                    logits_indices.extend([-1] * (max_num_sampled_tokens - n_tokens))
                    target_logits_indices.extend([-1] * (max_num_sampled_tokens - n_tokens))
            logits_indices = np.array(logits_indices, dtype=np.int32)
            bonus_logits_indices = np.array(bonus_logits_indices, dtype=np.int32)
            target_logits_indices = np.array(target_logits_indices, dtype=np.int32)

            cu_num_draft_tokens, arange = self._get_cumsum_and_arange(num_draft_tokens, cumsum_dtype=np.int32)

            # TODO: Optimize the CPU -> GPU copy.
            cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(self.device, non_blocking=True)
            cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(self.device, non_blocking=True)

            ##################################################
            logits_indices = torch.from_numpy(logits_indices)
            target_logits_indices_device = \
                torch.from_numpy(target_logits_indices).to(
                self.device, non_blocking=True)
            bonus_logits_indices_device = \
                torch.from_numpy(bonus_logits_indices).to(
                self.device, non_blocking=True)
            draft_token_ids = token_ids_device[target_logits_indices_device + 1]

            spec_decode_metadata = SpecDecodeMetadata(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens.tolist(),
                cu_num_draft_tokens=cu_num_draft_tokens,
                cu_num_sampled_tokens=cu_num_sampled_tokens,
                target_logits_indices=target_logits_indices_device,
                bonus_logits_indices=bonus_logits_indices_device,
                logits_indices=logits_indices,
            )
        return logits_indices, spec_decode_metadata

    def _prepare_unified_decode_inputs(self, num_decodes, num_scheduled_tokens, warmup_mode=False):

        if num_decodes == 0:
            return DecodeInputData(num_decodes=0), None

        context_lens = self.input_batch.num_computed_tokens_cpu[:num_decodes]
        query_lengths = [1] * num_decodes
        prompt_lengths = self.input_batch.num_prompt_tokens[:num_decodes]
        token_ids_cpu = self.input_batch.token_ids_cpu
        block_table_cpu_tensor = self.input_batch.block_table[0].get_cpu_tensor()
        num_blocks = [
            math.ceil((ctx_len + q_len) / self.block_size) for ctx_len, q_len in zip(context_lens, query_lengths)
        ]
        block_table = [block_table_cpu_tensor[i, :nb].tolist() for i, nb in enumerate(num_blocks)]
        if not warmup_mode:
            block_table = self.defragmenter.resolve_all(block_table)
        token_ids = [[token_ids_cpu[i, ctx_len]] for i, ctx_len in enumerate(context_lens)]

        batch_data = create_unified_batch(
            token_ids=token_ids,
            block_size=self.block_size,
            block_table=block_table,
            context_lengths=context_lens,
            query_lengths=[1] * num_decodes,
            prompt_lengths=prompt_lengths,
            dtype=self.dtype,
            contiguous_kv=self.use_contiguous_pa,
            bucketing_fn=self.unified_bucketing_fn,
            get_dp_padding_fn=self.get_dp_padding,
        )
        (token_ids_t, token_positions_t, logits_indices_t, logits_groups, attn_metadata) = batch_data
        decode_input_data = DecodeInputData(
            num_decodes=num_decodes,
            token_ids=token_ids_t.unsqueeze(-1),
            position_ids=token_positions_t.unsqueeze(-1),
            logits_indices=logits_indices_t,
            attn_metadata=attn_metadata,
        )
        return decode_input_data, None

    def _prepare_input_ids(self, scheduler_output: "SchedulerOutput") -> None:
        """Prepare the input IDs for the current batch.
        
        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            return

        # Compute cu_num_tokens from scheduler_output
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        flattened_indices = []
        prev_common_req_indices = []
        indices_match = True
        max_flattened_index = -1
        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (self.input_batch.prev_sampled_token_ids_invalid_indices is not None
                    and req_id in self.input_batch.prev_sampled_token_ids_invalid_indices):
                # This request was in the previous batch but its
                # prev_sampled_token_ids is invalid
                continue
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                flattened_index = cu_num_tokens[cur_index].item() - 1
                flattened_indices.append(flattened_index)
                indices_match &= (prev_index == flattened_index)
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(flattened_indices)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids_cpu will have all the input ids.
            return

        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids

        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1
            self.input_ids_hpu[:len(flattened_indices)].copy_(prev_sampled_token_ids[:len(flattened_indices)])
            return

        # Upload the index tensors asynchronously
        # so the scatter can be non-blocking
        input_ids_index_tensor = torch.tensor(flattened_indices, dtype=torch.int64).to(self.device, non_blocking=True)
        if prev_sampled_token_ids.size(0) <= len(prev_common_req_indices):
            prev_common_req_indices = prev_common_req_indices[:prev_sampled_token_ids.size(0)]
        prev_common_req_indices_tensor = torch.tensor(prev_common_req_indices, dtype=torch.int64).to(self.device,
                                                                                                     non_blocking=True)
        self.input_ids_hpu.scatter_(dim=0,
                                    index=input_ids_index_tensor,
                                    src=prev_sampled_token_ids[prev_common_req_indices_tensor])

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_prefills,
        num_decodes,
        warmup=False,
    ):

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0

        num_reqs = num_prefills + num_decodes

        ###############################################
        # NOTE(Chendi): Follow GPU_Model_Runner to use set global
        # self.input_ids_cpu and self.positions_cpu
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices], arange, out=positions_np)
        token_indices = (positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])
        ###############################################

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        num_prompt_tokens = []
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            assert req_id is not None
            seq_num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            seq_num_prompt_tokens = self.input_batch.num_prompt_tokens[idx]
            num_scheduled_tokens.append(seq_num_scheduled_tokens)
            num_prompt_tokens.append(seq_num_prompt_tokens)
        return (self._prepare_prefill_inputs(num_prefills, num_decodes, num_scheduled_tokens),
                self._prepare_decode_inputs(num_decodes, num_scheduled_tokens, scheduler_output))

    def _seq_len(self, attn_metadata):
        return attn_metadata.seq_len()

    def _num_blocks(self, attn_metadata):
        return attn_metadata.num_blocks()

    def _check_config(self, batch_size, seq_len, num_blocks, attn_metadata, warmup_mode):
        cfg: tuple[Any, ...] | None = None
        if self.unified_attn:
            phase = "prompt" if attn_metadata.is_prompt else "decode"
            shared = attn_metadata.shared_blocks.size(0) if attn_metadata.shared_blocks is not None else 0
            unique = attn_metadata.unique_blocks if attn_metadata.unique_blocks else 0
            is_causal = 1 if attn_metadata.causal_bias is not None else 0
            cfg = (seq_len, shared, unique)
            seen = cfg in self.seen_configs
            self.seen_configs.add(cfg)
            if not seen and not warmup_mode:
                logger.warning("Configuration: (query, shared_blocks, unique_blocks) %s, (%s) was not warmed-up!", \
                               cfg, 'causal' if is_causal else 'not causal')
        else:
            phase = "prompt" if attn_metadata.is_prompt else "decode"
            cfg = (phase, batch_size, seq_len, num_blocks)
            if self.debug_fwd:
                self.debug_fwd(cfg)
            seen = cfg in self.seen_configs
            self.seen_configs.add(cfg)
            if not seen and not warmup_mode:
                logger.warning("Configuration: %s was not warmed-up!", cfg)

    def _get_unified_config(self, attn_metadata, logits_indices):
        has_causal = 'c' if attn_metadata.causal_bias is not None else '-'
        has_shared = 's' if attn_metadata.shared_bias is not None else '-'
        has_unique = 'u' if attn_metadata.unique_bias is not None else '-'
        phase = has_causal + has_shared + has_unique
        qlen = attn_metadata.slot_mapping.size(0)
        num_shared_blocks = attn_metadata.shared_blocks.size(0) if attn_metadata.shared_blocks is not None else 0
        num_unique_blocks = attn_metadata.unique_blocks
        num_logits = logits_indices.size(0)
        cfg = (phase, qlen, num_shared_blocks, num_unique_blocks, num_logits)
        return cfg

    def _check_unified_config(self, attn_metadata, logits_indices, warmup_mode):
        cfg = self._get_unified_config(attn_metadata, logits_indices)
        if self.debug_fwd:
            self.debug_fwd(cfg)
        seen = cfg in self.seen_configs
        self.seen_configs.add(cfg)
        if not seen and not warmup_mode:
            logger.warning("Configuration: %s was not warmed-up!", cfg)

    def _execute_model_generic(self,
                               token_ids,
                               position_ids,
                               attn_metadata,
                               logits_indices,
                               kv_caches,
                               lora_logits_mask,
                               lora_mask,
                               warmup_mode=False,
                               inputs_embeds=None,
                               model_mm_kwargs=None):
        # FORWARD.
        batch_size = token_ids.size(0)
        seq_len = self._seq_len(attn_metadata)
        num_blocks = self._num_blocks(attn_metadata)
        if not self.unified_attn:
            self._check_config(batch_size, seq_len, num_blocks, attn_metadata, warmup_mode)
        else:
            self._check_unified_config(attn_metadata, logits_indices, warmup_mode)
        additional_kwargs = {}
        if htorch.utils.internal.is_lazy():
            use_graphs = self._use_graphs()
            additional_kwargs.update({"bypass_hpu_graphs": not use_graphs})
        else:
            # no hpu graphs for t.compile?
            use_graphs = False
        trimmed_attn_metadata = attn_metadata if self.unified_attn else trim_attn_metadata(attn_metadata)
        if self.is_driver_worker:
            model_event_name = ("model_forward_"
                                f"bs{batch_size}_"
                                f"seq{seq_len}_"
                                f"ctx{num_blocks}_"
                                f"graphs{'T' if use_graphs else 'F'}")
        else:
            model_event_name = 'model_executable'
        with self.profiler.record_event('internal', model_event_name):
            hidden_states = self.model.forward(input_ids=token_ids,
                                               positions=position_ids,
                                               attn_metadata=trimmed_attn_metadata,
                                               kv_caches=kv_caches,
                                               inputs_embeds=inputs_embeds,
                                               model_mm_kwargs=model_mm_kwargs,
                                               lora_mask=lora_mask,
                                               **additional_kwargs)
        # NOTE(kzawora): returning hidden_states is required in prompt logprobs
        # scenarios, as they will do logit processing on their own
        if self.use_aux_hidden_state_outputs:
            non_flattened_hidden_states, aux_hidden_states = hidden_states
            hidden_states = non_flattened_hidden_states
        else:
            non_flattened_hidden_states = hidden_states
            aux_hidden_states = None

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = hidden_states[logits_indices]
        LoraMask.setLoraMask(lora_logits_mask)
        with self.profiler.record_event('internal', ('compute_logits'
                                                     f'{batch_size}_'
                                                     f'seq{seq_len}_ctx'
                                                     f'{num_blocks}')):
            logits = self.model.compute_logits(hidden_states)
        return non_flattened_hidden_states, aux_hidden_states, \
            hidden_states, logits

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for i, (req_id, num_prompt_logprobs) in enumerate(num_prompt_logprobs_dict.items()):

            num_tokens = scheduler_output.num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(self.device, non_blocking=True)

            # Determine number of logits to retrieve.
            start_tok = request.num_computed_tokens + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens < num_remaining_tokens:
                # This is a chunk, more tokens remain.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            prompt_hidden_states = hidden_states[i, :num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            prompt_logprobs_dict[req_id] = LogprobsTensors(
                token_ids.to("cpu", non_blocking=True),
                logprobs.to("cpu", non_blocking=True),
                ranks.to("cpu", non_blocking=True),
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        torch.hpu.synchronize()

        return prompt_logprobs_dict

    def _is_quant_with_inc(self):
        quant_config = os.getenv("QUANT_CONFIG", None) is not None
        return (self.model_config.quantization == "inc" or quant_config)

    # Copied from vllm/v1/worker/gpu_model_runner.py
    def apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
    ):
        grammar_bitmask = scheduler_output.grammar_bitmask
        if grammar_bitmask is None:
            return

        # We receive the structured output bitmask from the scheduler,
        # compacted to contain bitmasks only for structured output requests.
        # The order of the requests in the bitmask is not guaranteed to be the
        # same as the order of the requests in the gpu runner's batch. We need
        # to sort the bitmask to match the order of the requests used here.

        # Get the batch indices of the structured output requests.
        # Keep track of the number of speculative tokens scheduled for every
        # request in the batch, as the logit indices are offset by this amount.
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        seq = sorted(self.input_batch.req_id_to_index.items(), key=lambda x: x[1])
        for req_id, batch_index in seq:
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            if req_id in scheduler_output.structured_output_request_ids:
                struct_out_req_batch_indices[req_id] = logit_index

        out_indices = []

        # Reorder the bitmask to match the order of the requests in the batch.
        sorted_bitmask = np.zeros_like(grammar_bitmask, shape=(logits.shape[0], grammar_bitmask.shape[1]))
        cumulative_index = 0

        for req_id in scheduler_output.structured_output_request_ids:
            logit_index = struct_out_req_batch_indices[req_id]
            num_spec_tokens = len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, []))
            for i in range(1 + num_spec_tokens):
                sorted_bitmask[logit_index + i] = \
                    grammar_bitmask[cumulative_index + i]
                out_indices.append(logit_index + i)
            cumulative_index += 1 + num_spec_tokens
        grammar_bitmask = sorted_bitmask

        # If the grammar bitmask and the logits have the same shape
        # we don't need to pass indices to the kernel,
        # since the bitmask is already aligned with the logits.
        skip_out_indices = grammar_bitmask.shape[0] == logits.shape[0]

        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = torch.from_numpy(grammar_bitmask).contiguous()

        # Force use of the torch.compile implementation from xgrammar to work
        # around issues with the Triton kernel in concurrent structured output
        # scenarios. See PR #19565 and issues #19493, #18376 for details.

        # xgr_torch_compile.apply_token_bitmask_inplace_torch_compile(
        #     logits,
        #     grammar_bitmask.to(self.device, non_blocking=True),
        #     indices=out_indices if not skip_out_indices else None,
        # )

        # NOTE(tianmu-li): xgr_torch_compile uses torch.inductor by default.
        # Have to use the CPU backend, which has its overhead.
        logits_cpu = logits.cpu().to(torch.float32)
        xgr_cpu.apply_token_bitmask_inplace_cpu(
            logits_cpu,
            grammar_bitmask.to("cpu"),
            indices=out_indices if not skip_out_indices else None,
        )
        logits.copy_(logits_cpu.to(self.device, non_blocking=True).to(logits.dtype))

    def _configure_lora(self, input, requests, req_ids, is_prompt):
        lora_mask = None
        lora_logits_mask = None
        if self.lora_config:
            if is_prompt:
                lora_requests = [] if req_ids else requests
                lora_ids = []
                lora_index_mapping = []
                lora_prompt_mapping = []
                for i, r_id in enumerate(req_ids):
                    lora_requests.append(requests[r_id].lora_request)
                for lora_req in lora_requests:
                    lora_id = lora_req.lora_int_id if lora_req else 0
                    lora_index_mapping += [lora_id] * (input.shape[1])
                    #TODO: This may need to change when logprobs
                    # sampling is enabled
                    lora_prompt_mapping += [lora_id]
                    lora_ids.append(lora_id)
            else:
                lora_requests = []
                # lora_ids, lora_index_mapping, lora_prompt_mapping
                # filled with 0 (indicating no lora) to account for
                # any padding
                lora_ids = [0] * input.shape[0]
                lora_index_mapping = [0] * input.shape[0]
                lora_prompt_mapping = [0] * input.shape[0]
                for i, r_id in enumerate(req_ids):
                    lora_requests.append(requests[r_id].lora_request)

                for i, lora_req in enumerate(lora_requests):
                    lora_id = lora_req.lora_int_id if lora_req else 0
                    lora_index_mapping[i] = lora_id
                    lora_prompt_mapping[i] = lora_id
                    lora_ids[i] = lora_id

            # is_prefill should always be "False" for HPU
            lora_mapping = LoRAMapping(lora_index_mapping, lora_prompt_mapping, is_prefill=False)
            self.set_active_loras(lora_requests, lora_mapping)
            lora_mask, lora_logits_mask = self.create_lora_mask(input, lora_ids, is_prompt)

        return lora_mask, lora_logits_mask

    def _run_sampling(self,
                      batch_changed: bool,
                      logits_device: torch.Tensor,
                      request_ids: Optional[list[str]] = None,
                      pad_to: Optional[int] = None,
                      logits_requests=None) -> tuple[torch.Tensor, SamplingMetadata]:
        htorch.core.mark_step()
        sampling_metadata = self._prepare_sampling(batch_changed, request_ids, pad_to, logits_requests)
        sampler_output = self.sampler(logits=logits_device, sampling_metadata=sampling_metadata)
        htorch.core.mark_step()
        return sampler_output, sampling_metadata

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
    ) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs ==\
            len(self.input_batch.pooling_params), \
        "Either all or none of the requests in" \
        " a batch must be pooling request"
        hidden_states = hidden_states[:num_scheduled_tokens]

        pooling_metadata = self.input_batch.pooling_metadata
        pooling_metadata.build_pooling_cursor(num_scheduled_tokens_np.tolist(), device=hidden_states.device)

        num_reqs = self.input_batch.num_reqs

        seq_lens = (
            torch.tensor(self.input_batch.num_prompt_tokens[:num_reqs], dtype=torch.int32, device=self.device) +
            torch.tensor(self.input_batch.num_computed_tokens_cpu[:num_reqs], dtype=torch.int32, device=self.device))
        raw_pooler_output = self.model.pooler(hidden_states=hidden_states, pooling_metadata=pooling_metadata)
        raw_pooler_output = json_map_leaves(
            lambda x: x.to("cpu", non_blocking=True),
            raw_pooler_output,
        )

        pooler_output: list[Optional[torch.Tensor]] = []
        for raw_output, seq_len, prompt_len in zip(raw_pooler_output, seq_lens, pooling_metadata.prompt_lens):

            if seq_len == prompt_len:
                pooler_output.append(raw_output)
            else:
                pooler_output.append(None)

        return ModelRunnerOutput(
            req_ids=[self.input_batch.req_ids],
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=None,
        )

    def _prepare_inputs_for_pooling(self, scheduler_output):
        """Gather inputs, positions, slot mapping, and build attn_metadata"""
        num_scheduled_tokens = []
        input_ids_list = []
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu
        num_reqs = self.input_batch.num_reqs

        # Collect token ids and scheduled lengths
        for idx, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            seq_num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(seq_num_scheduled)

            scheduled_req = scheduler_output.scheduled_new_reqs[idx]
            token_ids = torch.as_tensor(scheduled_req.prompt_token_ids, dtype=torch.long).flatten()
            input_ids_list.append(token_ids)

        input_ids = torch.cat(input_ids_list, dim=0).to(self.device)

        # Absolute positions
        absolute_positions = []
        for i, n in enumerate(num_scheduled_tokens):
            prefix = num_computed_tokens_cpu[i]
            absolute_positions.append(prefix + np.arange(n, dtype=np.int64))
        position_ids = torch.from_numpy(np.concatenate(absolute_positions)).to(self.device)

        # Slot mapping + metadata
        total_scheduled_tokens = sum(num_scheduled_tokens)
        slot_mapping = torch.arange(total_scheduled_tokens, dtype=torch.long, device="hpu:0")
        seq_lens_tensor = torch.tensor([total_scheduled_tokens], device='hpu:0', dtype=torch.int32)
        context_lens_tensor = torch.tensor([0], device='hpu:0', dtype=torch.int32)

        attn_metadata = HPUAttentionMetadataV1.make_prefill_metadata(
            seq_lens_tensor=seq_lens_tensor,
            context_lens_tensor=context_lens_tensor,
            slot_mapping=slot_mapping,
            block_list=None,
            attn_bias=None,
            block_size=self.block_size,
        )

        return input_ids, position_ids, num_scheduled_tokens, attn_metadata, \
            total_scheduled_tokens

    @torch.inference_mode()
    def run_defragmenter(self, scheduler_output: "SchedulerOutput", warmup_mode: bool = False):
        if self.defragmenter.enabled and self.kv_caches and not warmup_mode:
            new = {req.req_id: flatten(req.block_ids) for req in scheduler_output.scheduled_new_reqs if req.block_ids}
            #TODO: Add support for preempted blocks
            cached = {
                req_id: flatten(new_block_ids)
                for req_id, new_block_ids in zip(scheduler_output.scheduled_cached_reqs.req_ids,
                                                 scheduler_output.scheduled_cached_reqs.new_block_ids) if new_block_ids
            }
            self.defragmenter.update_state(new | cached, scheduler_output.finished_req_ids)
            self.defragmenter.defragment()

    def prepare_unified_batch(self, scheduler_output):
        num_reqs = len(self.input_batch.req_ids)
        num_computed_tokens = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
        num_prompt_tokens = torch.from_numpy(self.input_batch.num_prompt_tokens[:num_reqs])
        num_scheduled_tokens = torch.tensor(
            [scheduler_output.num_scheduled_tokens[req_id] for req_id in self.input_batch._req_ids],
            dtype=torch.int32,
            device='cpu')
        max_seq = (num_computed_tokens + num_scheduled_tokens).max()
        max_blocks = (max_seq + self.block_size - 1) // self.block_size
        all_token_ids = self.input_batch.token_ids_cpu_tensor[:num_reqs, :max_seq]
        # TODO: check if it's safe to always slice on first dim
        block_table = self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs, :max_blocks].clone()
        if self.defragmenter.enabled:
            block_table.apply_(self.defragmenter.resolve)

        return create_unified_batch(self.input_batch.req_ids, all_token_ids, num_computed_tokens, num_scheduled_tokens,
                                    num_prompt_tokens, block_table, self.block_size, self.dtype,
                                    self.unified_bucketing_fn, self.get_dp_padding)

    @torch.inference_mode()
    def unified_execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        warmup_mode: bool = False,
    ) -> ModelRunnerOutput:
        self.run_defragmenter(scheduler_output, warmup_mode)
        batch_changed = self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT
        htorch.core.mark_step()
        with self.profiler.record_event('internal', 'prepare_unified_batch'):
            batch = self.prepare_unified_batch(scheduler_output)
        htorch.core.mark_step()
        if self.is_driver_worker:
            unified_attn_cfg = self._get_unified_config(batch.attn_metadata, batch.logits_indices)
            (phase, qlen, num_shared_blocks, num_unique_blocks, num_logits) = unified_attn_cfg
            model_event_name = (
                f"model_forward_{phase}_qlen{qlen}_nsb{num_shared_blocks}_nub{num_unique_blocks}_nlog{num_logits}")
        else:
            model_event_name = 'model_executable'
        with self.profiler.record_event('internal', model_event_name):
            non_flattened_hidden_states, aux_hidden_states, hidden_states, logits_device = \
                self._execute_model_generic(
                    token_ids=batch.token_ids.unsqueeze(-1),
                    position_ids=batch.token_positions.unsqueeze(-1),
                    attn_metadata=batch.attn_metadata,
                    logits_indices=batch.logits_indices,
                    kv_caches=self.kv_caches,
                    lora_logits_mask=None,
                    lora_mask=None,
                    warmup_mode=warmup_mode)
        selected_req_ids = [batch.req_ids_cpu[idx] for idx in batch.logits_groups_cpu.tolist()]
        htorch.core.mark_step()
        with self.profiler.record_event('internal', 'unified_sampler'):
            sampling_metadata = self._prepare_sampling(batch_changed, selected_req_ids, pad_to=logits_device.shape[0])
            sampler_output = self.sampler(logits=logits_device, sampling_metadata=sampling_metadata)

        with self.profiler.record_event('internal', 'unified_postprocess'):
            sampled_token_ids_cpu = sampler_output.sampled_token_ids.cpu()
            sampled_token_ids: list[list[int]] = [[] for _ in batch.req_ids_cpu]
            for req_id, tokens in zip(selected_req_ids, sampled_token_ids_cpu.tolist()):
                sampled_token_ids[self.input_batch.req_id_to_index[req_id]].extend(tokens)

            #TODO: add support for multi-token output
            assert sampled_token_ids_cpu.size(1) == 1, 'Currently only single token output is supported!'
            sampled_token_ids_cpu = sampled_token_ids_cpu.flatten()
            htorch.core.mark_step()

            sampled_token_ids_cpu = sampled_token_ids_cpu.index_select(0, batch.logits_groups_cpu)
            self.input_batch.token_ids_cpu_tensor.index_put_((batch.logits_groups_cpu, batch.new_token_positions_cpu),
                                                             sampled_token_ids_cpu)

            ######### UPDATE REQUEST STATE WITH GENERATED TOKENS #########
            num_reqs = len(selected_req_ids)
            for req_id in self.input_batch.req_ids[:num_reqs]:
                req_state = self.requests[req_id]
                i = self.input_batch.req_id_to_index[req_id]
                seq_len = (req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id])
                token_ids = sampled_token_ids[i]
                num_tokens = len(token_ids)
                self.input_batch.token_ids_cpu[i, seq_len:seq_len + num_tokens] = token_ids
                self.input_batch.num_tokens[i] += len(token_ids)
                req_state.output_token_ids.extend(token_ids)

        model_runner_output = ModelRunnerOutput(
            req_ids=batch.req_ids_cpu,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        return model_runner_output

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        warmup_mode: bool = False,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput]:
        if self.unified_attn:
            return self.unified_execute_model(scheduler_output, warmup_mode)
        # NOTE(kzawora): Since scheduler doesn't differentiate between prefills
        # and decodes, we must handle mixed batches. In _update_states we make
        # sure that first self.input_batch.num_decodes requests are decodes,
        # and remaining ones until the end are prefills. _update_states also
        # handles changes in request cache based on scheduler outputs and
        # previous iterations (e.g. keeping block tables and context lengths up
        # to date, creating, pruning and updating request caches,
        # and some more stuff)

        # If num_decodes == self.input_batch.num_reqs, then batch is all decode, and only a single decode forward pass will be executed in this method. # noqa
        # If num_decodes == 0, then batch is all prefill, and only prefill forward passes will be executed  in this method. # noqa
        # If neither apply, then batch is mixed, and both prefill and decode forward passes will be executed in this method. # noqa

        # First, we will execute all decodes (if any) in a single batch,
        # then we'll execute prefills in batches of up to max_prefill_batch_size elements. # noqa
        # All shapes used in forward passes are bucketed appropriately to mitigate risk of graph recompilations. # noqa

        # We perform sampling directly after executing each forward pass
        # Everything is done asynchronously - the only sync point is the place
        # where we copy the generated tokens back to the host.

        # Example: If a batch has 6 requests, 3 prefills and 3 decodes, the unprocessed sequences in batch will be laid as follows: # noqa
        # [D0, D1, D2, P0, P1, P2]
        # If we assume max_prefill_batch_size=2, the flow of this method will look as follows: # noqa
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1], [P2] (BS=2 + BS=1 bucket, no seqs padding) # noqa
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS2 [P0, P1]
        # prefill[iter 0] compute_logits BS2 [P0, P1]
        # prefill[iter 0] sampler BS2 [P0, P1] -> [tokP0, tokP1]
        # prefill[iter 1] forward pass BS1 [P0, P1]
        # prefill[iter 1] compute_logits BS1 [P0, P1]
        # prefill[iter 1] sampler BS1 [P0, P1] -> [tokP2]
        # prefill concat sampler results [tokP0, tokP1], [tokP2] -> [tokP0, tokP1, tokP2] # noqa
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] # noqa
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2] # noqa
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        # Example2: Same thing, but with max_prefill_batch_size=4:
        # prepare_inputs: bucket [D0, D1, D2] -> [D0, D1, D2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # prepare_inputs: bucket [P0, P1, P2] -> [P0, P1, P2, 0] (BS=4 bucket, 1 seq padding) # noqa
        # decode forward pass BS4 [D0, D1, D2, 0]
        # decode compute_logits BS4 [D0, D1, D2, 0]
        # decode sampler BS4 [D0, D1, D2, 0] -> [tokD0, tokD1, tokD2, 0]
        # prefill[iter 0] forward pass BS4 [P0, P1, P2, 0]
        # prefill[iter 0] compute_logits BS4 [P0, P1, P2, 0]
        # prefill[iter 0] sampler BS4 [P0, P1, P2, 0] -> [tokP0, tokP1, tokP2, 0] # noqa
        # Join the prefill and decode on device into [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] # noqa
        # Transfer [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] to CPU
        # On CPU, sanitize [tokD0, tokD1, tokD2, 0, tokP0, tokP1, tokP2, 0] -> [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2] # noqa
        # Return [tokD0, tokD1, tokD2, tokP0, tokP1, tokP2]

        self.run_defragmenter(scheduler_output, warmup_mode)

        batch_changed = self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group() or warmup_mode:
                # Return empty ModelRunnerOuptut if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT
            # For D case, wait until kv finish load here
            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
        if self.input_batch.pooling_params:
            (input_ids, position_ids, num_scheduled_tokens, attn_metadata,
             total_scheduled_tokens) = self._prepare_inputs_for_pooling(scheduler_output)

            with set_forward_context(attn_metadata, self.vllm_config):
                hidden_states = self.model.forward(
                    input_ids=input_ids,
                    positions=position_ids,
                )

            flattened = hidden_states.view(-1, hidden_states.shape[-1])
            pooled_output = self._pool(
                flattened,
                total_scheduled_tokens,
                np.array(num_scheduled_tokens, dtype=np.int32),
            )
            return pooled_output
        # If necessary, swap decodes/prompts to have all decodes on the start

        ensure_decodes_first(self.input_batch)
        # Prepare prompts/decodes info
        pd_info = self._get_prompts_and_decodes(scheduler_output)
        num_decodes = len(pd_info.decode_req_ids)
        num_prefills = len(pd_info.prompt_req_ids)
        num_reqs = num_decodes + num_prefills
        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            prefill_input_data, decode_input_data = self._prepare_inputs(scheduler_output, num_prefills, num_decodes,
                                                                         warmup_mode)
        prefill_data, \
            dummy_prefill_input_data_batches_across_dp = prefill_input_data
        num_pad_prefill_batch_across_dp = \
            0 if dummy_prefill_input_data_batches_across_dp is None \
            else len(dummy_prefill_input_data_batches_across_dp.request_ids)
        decode_data, dummy_decode_input_data_across_dp = decode_input_data
        #FIXME(kzawora): Currently there's no handling of logprobs. Fix that
        # later.
        prefill_sampled_token_ids = []
        prefill_sampled_requests = []
        decode_sampled_token_ids = []
        decode_sampled_requests = []
        #if not has_kv_transfer_group():
        #    assert not (num_prefills > 0 and num_decodes > 0)
        # skip kv_connector if dummy run
        if not warmup_mode:
            with set_forward_context(None, self.vllm_config):
                self.maybe_setup_kv_connector(scheduler_output)
        finished_sending, finished_recving = set(), set()

        # NOTE(Chendi): used by spec decode draft model, since we are doing
        # prefill one by one, so save hidden states as list
        non_flattened_hidden_states_prefills = []
        aux_hidden_states_prefills = []
        sample_hidden_states_prefills = []
        decode_sampled_token_ids_device = None
        # NOTE(tianmu-li): For structured output, combine logits before
        # postprocessing. Should it be done for all requests?
        structured_output = False
        spec_decode_num_tokens = None
        if scheduler_output.grammar_bitmask is not None:
            logits_prompt = []
            logits_decode = []
            structured_output = True
        if self.use_async_scheduling:
            invalid_req_indices = []
        ######################### PREFILLS #########################
        if num_prefills > 0:
            htorch.core.mark_step()
            for idx, (req_id, prompt_len, token_ids, position_ids, attn_metadata, logits_indices,
                      logits_requests) in enumerate(zip(*shallow_tuple(prefill_data))):

                inputs_embeds = None
                model_mm_kwargs = None
                if self.supports_mm_inputs:
                    # Run the multimodal encoder if any.
                    with self.profiler.record_event('internal', 'prepare_input_encoders'):
                        self._execute_mm_encoder(scheduler_output, req_id)
                    htorch.core.mark_step()

                    mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output,
                                                                        req_id,
                                                                        total_num_scheduled_tokens=token_ids.shape[-1])
                    htorch.core.mark_step()

                    # TODO: Only get embeddings for valid token_ids. Ignore token_ids[<pad_idxs>] # noqa E501
                    # This may require moving multimodal input preps into _prepare_inputs,        # noqa E501
                    # to avoid padding issues.
                    inputs_embeds = self.model.get_input_embeddings(
                        token_ids,
                        multimodal_embeddings=mm_embeds,
                        is_multimodal=is_mm_embed,
                    )

                    model_mm_kwargs = self._extract_mm_kwargs(scheduler_output)
                    model_mm_kwargs = MultiModalKwargs.as_kwargs(
                        model_mm_kwargs,
                        device=self.device,
                    )

                lora_mask, lora_logits_mask = self._configure_lora(token_ids, self.requests, req_id, True)

                self.event_start = self.profiler.get_timestamp_us()
                self.profiler.start("internal", "prefill")
                # NOTE(tianmu-li): Align behavior of incomplete prompt with gpu_model_runner
                # If logits_indices is smaller than req_id, the last request is a chunked prompt request that
                # hasn't finished in this step. We add the last token position to logits_indices to ensure
                # the last token of the chunk is sampled. This sampled token will be discarded later
                if logits_indices.shape[0] < len(req_id):
                    if structured_output or self.use_async_scheduling:
                        # When there are multiple requests in the batch (e.g. self.use_merged_prefill=True),
                        # the last token position is the sum of all prompt lengths - 1
                        # This logic also holds when there is only one request in the batch
                        logits_indices_append = torch.tensor([torch.sum(prompt_len) - 1],
                                                             device=token_ids.device,
                                                             dtype=torch.int32)
                        logits_indices = torch.cat([logits_indices, logits_indices_append])
                    if self.use_async_scheduling:
                        # Discard partial prefill logit for async scheduling
                        # Depends on 1 decode token/batch
                        prefill_start_idx = num_decodes
                        invalid_req_indices.append(prefill_start_idx + idx)
                htorch.core.mark_step()
                non_flattened_hidden_states, aux_hidden_states, \
                    sample_hidden_states, logits_device = \
                    self._execute_model_generic(
                        token_ids, position_ids, attn_metadata, logits_indices,
                        self.kv_caches,
                        lora_logits_mask,
                        lora_mask,
                        inputs_embeds=inputs_embeds,
                        model_mm_kwargs=model_mm_kwargs,
                        warmup_mode=warmup_mode,)
                htorch.core.mark_step()
                non_flattened_hidden_states_prefills.append(non_flattened_hidden_states)
                if self.use_aux_hidden_state_outputs:
                    aux_hidden_states_prefills.append(aux_hidden_states)
                sample_hidden_states_prefills.append(sample_hidden_states)
                # Skip separate sampling for structured output
                if structured_output:
                    logits_prompt.append(logits_device)
                    prefill_sampled_requests.extend(logits_requests)
                else:
                    # If there are no logits, there is nothing to sample.
                    # This can happen with chunked prefill when a chunk does
                    # not complete the prompt and no logits are generated.
                    if logits_device.numel() > 0:
                        with self.profiler.record_event('internal', "sampler"):
                            sampler_output, sampling_metadata = self._run_sampling(batch_changed, logits_device, req_id,
                                                                                   logits_device.shape[0],
                                                                                   logits_requests)
                            prefill_sampled_token_ids.append(sampler_output.sampled_token_ids.flatten())
                            prefill_sampled_requests.extend(logits_requests)
                if self.is_driver_worker and self.profiler.enabled:
                    # Stop recording 'execute_model_generic' event
                    self.profiler.end()
                    event_end = self.profiler.get_timestamp_us()
                    counters = self.profiler_counter_helper.get_counter_dict(cache_config=self.cache_config,
                                                                             duration=event_end - self.event_start,
                                                                             seq_len=self._seq_len(attn_metadata),
                                                                             batch_size_padded=token_ids.size(0),
                                                                             real_batch_size=len(req_id),
                                                                             prompt_batch_idx=idx,
                                                                             is_prompt=True)
                    self.profiler.record_counter(self.event_start, counters)
            if not warmup_mode:
                self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (self.get_finished_kv_transfers(scheduler_output))

            if self.is_driver_worker and self.profiler.enabled:
                self.profiler_counter_helper.reset_prompt_seq_stats()

        if num_pad_prefill_batch_across_dp > 0:
            for idx, (req_id, prompt_len, token_ids, position_ids, attn_metadata, logits_indices,
                      logits_requests) in enumerate(zip(*shallow_tuple(dummy_prefill_input_data_batches_across_dp))):
                htorch.core.mark_step()
                _, _, _, dummy_logits_device = \
                self._execute_model_generic(
                    token_ids,
                    position_ids,
                    attn_metadata,
                    logits_indices,
                    self.kv_caches,
                    None,
                    None,
                    warmup_mode=warmup_mode)
                htorch.core.mark_step()

        ######################### DECODES #########################
        # Decodes run as one single batch with [padded_decode_bs, 1]
        if num_decodes > 0:
            assert decode_data is not None
            lora_mask, lora_logits_mask = self._configure_lora(decode_data.token_ids, self.requests,
                                                               pd_info.decode_req_ids, False)
            self.event_start = self.profiler.get_timestamp_us()
            self.profiler.start("internal", "decode")
            htorch.core.mark_step()
            non_flattened_hidden_states, aux_hidden_states, \
                sample_hidden_states, logits_device = \
                    self._execute_model_generic(
                decode_data.token_ids,
                decode_data.position_ids,
                decode_data.attn_metadata,
                decode_data.logits_indices,
                self.kv_caches,
                lora_logits_mask,
                lora_mask,
                warmup_mode=warmup_mode)
            htorch.core.mark_step()

            if structured_output:
                logits_decode.append(logits_device[:num_decodes])
                decode_sampled_requests.extend(self.input_batch.req_ids[:num_decodes])
            else:
                with self.profiler.record_event('internal', "sampler"):
                    ##### Sampling Start #####
                    spec_decode_metadata = decode_data.spec_decode_metadata
                    sampler_output, sampling_metadata = self._run_sampling(
                        batch_changed, logits_device
                        if spec_decode_metadata is None else logits_device[spec_decode_metadata.bonus_logits_indices],
                        pd_info.decode_req_ids, logits_device.shape[0])

                    if spec_decode_metadata is None:
                        decode_sampled_token_ids.append(sampler_output.sampled_token_ids.flatten())
                    else:
                        # Handling spec decode sampling.
                        sampler_output = self.rejection_sampler(
                            spec_decode_metadata,
                            None,  # draft_probs
                            logits_device,
                            sampling_metadata,
                        )
                        sampled_token_ids = sampler_output.sampled_token_ids
                        decode_sampled_token_ids = \
                            self.rejection_sampler.parse_output(
                                sampled_token_ids,
                                self.input_batch.vocab_size,
                        )
                        # convert decode_sampled_token_ids as list of tensor
                        spec_decode_num_tokens = [len(v) for v in decode_sampled_token_ids]
                        decode_sampled_token_ids = [
                            torch.tensor(v, device="cpu").int() for v in decode_sampled_token_ids
                        ]
                        decode_sampled_token_ids_device = \
                            sampled_token_ids.to("hpu", non_blocking=True)
                    decode_sampled_requests.extend(self.input_batch.req_ids[:num_decodes])
                    ##### Sampling End #####

            if self.is_driver_worker and self.profiler.enabled:
                # Stop recording 'execute_model' event
                self.profiler.end()
                event_end = self.profiler.get_timestamp_us()
                counters = self.profiler_counter_helper.get_counter_dict(
                    cache_config=self.cache_config,
                    duration=event_end - self.event_start,
                    seq_len=self._seq_len(decode_data.attn_metadata),
                    batch_size_padded= \
                        decode_data.token_ids.size(0), # type: ignore
                    real_batch_size=decode_data.num_decodes,
                    prompt_batch_idx=None,
                    is_prompt=False)
                self.profiler.record_counter(self.event_start, counters)

        elif dummy_decode_input_data_across_dp is not None:
            htorch.core.mark_step()
            _, _, _, dummy_logits_device = self._execute_model_generic(dummy_decode_input_data_across_dp.token_ids,
                                                                       dummy_decode_input_data_across_dp.position_ids,
                                                                       dummy_decode_input_data_across_dp.attn_metadata,
                                                                       dummy_decode_input_data_across_dp.logits_indices,
                                                                       self.kv_caches,
                                                                       None,
                                                                       None,
                                                                       warmup_mode=warmup_mode)
            htorch.core.mark_step()

        if structured_output:
            # Scheduler places cached before prompt
            logits_combined = logits_decode + logits_prompt
            logits = torch.cat(logits_combined, dim=0)
            # Apply structured output bitmasks if present
            if scheduler_output.structured_output_request_ids:
                self.apply_grammar_bitmask(scheduler_output, logits)
            sampler_output, _sampling_metadata = self._run_sampling(batch_changed, logits,
                                                                    pd_info.prompt_req_ids + pd_info.decode_req_ids,
                                                                    logits.shape[0])
            # Deal with the case of incomplete prompt
            for i in range(logits.shape[0] - num_decodes):
                prefill_sampled_token_ids.append(sampler_output.sampled_token_ids[num_decodes + i].flatten())
            decode_sampled_token_ids.append(sampler_output.sampled_token_ids[:num_decodes].flatten())
        elif self.use_async_scheduling:
            # For async scheduling: keep tokens on HPU and avoid CPU sync
            # Concatenate decode and prefill tokens on HPU
            if decode_sampled_token_ids or prefill_sampled_token_ids:
                decode_sampled_token_ids = [tensor[:num_decodes] for tensor in decode_sampled_token_ids]
                # Note: this will cause an issue with the current spec decode impl, as they are on different devices
                sampled_token_ids = torch.cat(decode_sampled_token_ids + prefill_sampled_token_ids).view(-1, 1)
            else:
                sampled_token_ids = torch.empty((0, 1), dtype=torch.int32, device=self.device)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = \
            self.input_batch.req_id_to_index.copy()

        max_req_index = max(self.input_batch.req_id_to_index.values())
        postprocessed_sampled_token_ids: list[list[int]] = [[] for _ in range(max_req_index + 1)]
        if self.use_async_scheduling:
            self.input_batch.prev_sampled_token_ids = sampled_token_ids.flatten()
            # self.input_batch.prev_sampled_token_ids_invalid_indices
            invalid_req_indices_set = set(invalid_req_indices)
            self.input_batch.prev_sampled_token_ids_invalid_indices = \
                invalid_req_indices_set
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids) if i not in invalid_req_indices_set
            }
            # For the output, postprocessed_sampled_token_ids will be filled during serialization
        else:
            prefill_sampled_token_ids_device = prefill_sampled_token_ids
            # From this point onward, all operations are done on CPU.
            # We already have tokens. Let's copy the data to
            # CPU as is, and then discard padded tokens.
            with self.profiler.record_event('internal', "sampler_postprocessing"):
                prefill_sampled_token_ids = [tensor.cpu() for tensor in prefill_sampled_token_ids]
                if spec_decode_num_tokens is not None:
                    decode_sampled_token_ids = [tensor.cpu() for tensor in decode_sampled_token_ids]
                else:
                    decode_sampled_token_ids = [tensor.cpu()[:num_decodes] for tensor in decode_sampled_token_ids]
                if decode_sampled_token_ids + prefill_sampled_token_ids:
                    sampled_token_ids_list = torch.cat(decode_sampled_token_ids + prefill_sampled_token_ids).tolist()
                else:
                    sampled_token_ids_list = []
                sampled_token_requests = \
                    decode_sampled_requests + prefill_sampled_requests
                max_req_index = max(self.input_batch.req_id_to_index.values())
                # NOTE(Chendi): in post-processing, spec_decode might
                # return more than 1 token during decode.
                start_idx = 0
                for i, req_id in enumerate(sampled_token_requests):
                    num_tokens = spec_decode_num_tokens[
                        i] if spec_decode_num_tokens is not None and i < num_decodes else 1
                    postprocessed_sampled_token_ids[
                        self.input_batch.req_id_to_index[req_id]] += sampled_token_ids_list[start_idx:start_idx +
                                                                                            num_tokens]
                    start_idx += num_tokens

        ################## RETURN ##################
        # NOTE(kzawora): idk what happens if part of batch doesn't have logprobs

        ######### UPDATE REQUEST STATE WITH GENERATED TOKENS #########
        for req_id in self.input_batch.req_ids[:num_reqs]:
            req_state = self.requests[req_id]
            i = self.input_batch.req_id_to_index[req_id]
            seq_len = (req_state.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id])
            token_ids = postprocessed_sampled_token_ids[i]
            num_tokens = len(token_ids)
            self.input_batch.token_ids_cpu[i, seq_len:seq_len + num_tokens] = token_ids
            self.input_batch.num_tokens[i] += len(token_ids)
            req_state.output_token_ids.extend(token_ids)
        # NOTE(chendi): enable cache based on PR(#20291)
        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        for req_idx, sampled_ids in enumerate(postprocessed_sampled_token_ids[:num_reqs]):
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            # NOTE(adobrzyn): assert for full max prompt length including
            # max_model_len and one token that's going to be generated
            # especially needed for biggest prompt in warm-up phase
            full_max_prompt = self.max_model_len + 1
            assert end_idx <= full_max_prompt, ("Sampled token IDs exceed the max model length. "
                                                f"Total number of tokens: {end_idx} > max_model_len: "
                                                f"{full_max_prompt}")

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx
            req_id = self.input_batch.req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        ################## Spec Decode ##################
        # Now, we will call drafter to propose draft token ids
        if self.speculative_config:
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output, postprocessed_sampled_token_ids, prefill_sampled_token_ids_device,
                decode_sampled_token_ids_device, sampling_metadata, non_flattened_hidden_states, sample_hidden_states,
                aux_hidden_states, non_flattened_hidden_states_prefills, sample_hidden_states_prefills,
                aux_hidden_states_prefills, num_decodes, prefill_data if num_prefills > 0 else None,
                decode_data if num_decodes > 0 else None)
        ################## Spec Decode end ##################

        # Create output.
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids
        # prompt_logprobs_dict: dict[
        #    str, Optional[LogprobsTensors]] = self._get_prompt_logprobs_dict(
        #        prefill_hidden_states_device, scheduler_output)
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}
        all_req_ids = pd_info.decode_req_ids + pd_info.prompt_req_ids
        logprobs = None

        if self.use_async_scheduling:
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,  # CHECK
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=postprocessed_sampled_token_ids,
                logprobs=logprobs,
                prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
                pooler_output=[],
                kv_connector_output=KVConnectorOutput(
                    finished_sending=finished_sending,
                    finished_recving=finished_recving,
                ))
            return AsyncHPUModelRunnerOutput(
                model_runner_output=model_runner_output,
                sampled_token_ids=sampled_token_ids,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
            )
        model_runner_output = ModelRunnerOutput(
            req_ids=all_req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=postprocessed_sampled_token_ids,
            logprobs=logprobs,
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            pooler_output=[],
            kv_connector_output=KVConnectorOutput(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
            ))
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()

        return model_runner_output

    def load_model(self) -> None:
        import habana_frameworks.torch.core as htcore
        if self.model_config.quantization == 'inc' or \
                self.model_config.quantization == 'fp8':
            htcore.hpu_set_env()
        logger.info("Starting to load model %s...", self.model_config.model)
        with HabanaMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config, self.device)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Loading model weights took %.4f GB", self.model_memory_usage / float(2**30))

        if self._is_quant_with_inc():
            logger.info("Preparing model with INC..")
            with HabanaMemoryProfiler() as m_inc:
                from neural_compressor.torch.quantization import (FP8Config, convert, prepare)
                config = FP8Config.from_json_file(os.getenv("QUANT_CONFIG", ""))
                self._inc_preprocess()
                if config.measure:
                    self.model = prepare(self.model, config)
                elif config.quantize:
                    self.model = convert(self.model, config)
                else:
                    raise ValueError("Unknown quantization config mode,"
                                     "please validate quantization config file")
                htcore.hpu_initialize(self.model, mark_only_scales_as_const=True)
            self.inc_initialized_successfully = True
            self.model_memory_usage = m_inc.consumed_device_memory
            logger.info("Preparing model with INC took %.4f GB", self.model_memory_usage / float(2**30))
        elif not is_fake_hpu():
            self.model = self.model.to("hpu")
            htcore.mark_step()

        hidden_layer_markstep_interval = int(os.getenv('VLLM_CONFIG_HIDDEN_LAYERS', '1'))
        model_config = getattr(self.model, "config", None)
        modify_model_layers(self.model,
                            get_target_layer_suffix_list(model_config.model_type if model_config is not None else None),
                            hidden_layer_markstep_interval)
        torch.hpu.synchronize()

        if not self.is_pooling_model:
            with HabanaMemoryProfiler() as m:
                self.model = _maybe_wrap_in_hpu_graph(
                    self.model,
                    vllm_config=self.vllm_config,
                )
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Wrapping in HPUGraph took %.4f GB", self.model_memory_usage / float(2**30))

        ########### Spec Decode model ############
        if hasattr(self, "drafter"):
            with HabanaMemoryProfiler() as m:  # noqa: SIM117
                logger.info("Loading drafter model %s...", self.vllm_config.speculative_config.draft_model_config)
                self.drafter.load_model(self.model.model)
                if self.use_aux_hidden_state_outputs:
                    if supports_eagle3(self.model.model):
                        self.model.model.set_aux_hidden_state_layers(
                            self.model.model.get_eagle3_aux_hidden_state_layers())
                    else:
                        raise RuntimeError("Model does not support EAGLE3 interface but "
                                           "aux_hidden_state_outputs was requested")
            self.model_memory_usage = m.consumed_device_memory
            logger.info("Loading drafter model weights took %.4f GB", self.model_memory_usage / float(2**30))
            if hasattr(self.drafter, "model"):
                self.drafter.model = self.drafter.model.to("hpu")
                torch.hpu.synchronize()
                with HabanaMemoryProfiler() as m:  # noqa: SIM117
                    self.drafter.model = _maybe_wrap_in_hpu_graph(self.drafter.model, vllm_config=self.vllm_config)
                self.model_memory_usage = m.consumed_device_memory
                logger.info("Wrapping in HPUGraph took %.4f GB", self.model_memory_usage / float(2**30))
        #############################################

        with HabanaMemoryProfiler() as m:
            self._maybe_compile(self.model)
        self.model_memory_usage = m.consumed_device_memory
        logger.info("Compilation took %.4f GB", self.model_memory_usage / float(2**30))

    def _maybe_compile(self, *args, **kwargs):
        """Entrypoint for a torch.compilation of the model"""
        if (not is_fake_hpu() and not htorch.utils.internal.is_lazy()
                and not self.vllm_config.model_config.enforce_eager):
            # force_parameter_static_shapes = False  alows to use dynamic
            # shapes on tensors added to module via register_buffer()
            torch._dynamo.config.force_parameter_static_shapes = False
            self.compile_config = HPUCompileConfig()
            if self.compile_config.regional_compilation:
                self._compile_methods()
                self.regional_compilation_layers_list = [RMSNorm, VocabParallelEmbedding]
                self._regional_compilation(self.model)
                self.sampler = self._compile(self.sampler)
            else:
                self.model = self._compile(self.model)

    def _compile_methods(self):
        """
        Compile methods which are not part of the compiled model i.e. those
        which will not be compiled during model's compilation.
        """
        compiled_methods = ['_update_metadata', '_rotary_prepare_cos_sin']
        for method_name in compiled_methods:
            method = getattr(self.model, method_name)
            if method is not None:
                self._compile_region(self.model, method_name, method)

    def _regional_compilation(self, module, parent_module=None, module_name=None):
        """
        Recursively traverses a PyTorch module and compiles its regions, which
        can be one of two:
        1. Children of the nn.ModuleList
        2. Member of regional_compilation_layers_list
        """
        if isinstance(module, torch.nn.ModuleList):
            for children_name, children_module in module.named_children():
                self._compile_region(module, children_name, children_module)
        elif any(isinstance(module, layer) for layer in self.regional_compilation_layers_list):
            self._compile_region(
                parent_module,
                module_name,
                module,
            )
        else:
            for children_name, children_module in module.named_children():
                self._regional_compilation(children_module, module, children_name)

    def _compile_region(self, model, name, module):
        module = self._compile(module)
        setattr(model, name, module)

    def _compile(self, module):
        return torch.compile(module, **self.compile_config.get_compile_args())

    def _use_graphs(self):
        return not self.model_config.enforce_eager

    def _remove_duplicate_submodules(self):
        model = self.get_model()
        if hasattr(model, "model"):
            for layer in self.get_model().model.layers:
                self_attn = layer.self_attn
                # delete attr kv_b_proj in self_attn,
                # as they have been transferred to the MLAImpl.
                if hasattr(self_attn, "mla_attn"):
                    mla_attn = self_attn.mla_attn
                    duplicate_mods = [
                        "kv_a_proj_with_mqa",
                        "q_proj",
                        "kv_b_proj",
                        "o_proj",
                        "fused_qkv_a_proj",
                        "q_b_proj",
                    ]
                    for m in duplicate_mods:
                        if hasattr(self_attn, m) and hasattr(mla_attn, m):
                            delattr(self_attn, m)
                    if hasattr(mla_attn, "mla_attn") and hasattr(mla_attn.mla_attn, "impl"):
                        mla_impl = mla_attn.mla_attn.impl
                        duplicate_mods = ["kv_b_proj"]
                        for m in duplicate_mods:
                            if hasattr(mla_attn, m) and hasattr(mla_impl, m):
                                delattr(mla_attn, m)

    def _inc_preprocess(self):
        self._remove_duplicate_submodules()

    def log_graph_warmup_summary(self, buckets, is_prompt, total_mem):
        phase = f'Graph/{"Prompt" if is_prompt else "Decode"}'
        msg = (f'{phase} captured:{len(buckets)} '
               f'used_mem:{format_bytes(total_mem)}')
        logger.info(msg)

    def log_warmup(self, phase, i, max_i, first_dim, second_dim, third_dim, causal=False):
        free_mem = format_bytes(HabanaMemoryProfiler.current_free_device_memory())
        if self.unified_attn:
            msg = (f"[Warmup][{phase}][{i + 1}/{max_i}] "
                   f"query_len:{first_dim} "
                   f"shared_blocks:{second_dim} "
                   f"unique_blocks:{third_dim} "
                   f"({'causal' if causal else 'non causal'}) "
                   f"free_mem:{free_mem}")
        else:
            msg = (f"[Warmup][{phase}][{i + 1}/{max_i}] "
                   f"batch_size:{first_dim} "
                   f"query_len:{second_dim} "
                   f"num_blocks:{third_dim} "
                   f"free_mem:{free_mem}")
        logger.info(msg)

    def log_warmup_multimodal(self, phase, i, max_i, batch_size, seq_len, img_args):
        free_mem = format_bytes(HabanaMemoryProfiler.current_free_device_memory())
        msg = (f"[Warmup][{phase}][{i+1}/{max_i}] "
               f"batch_size:{batch_size} "
               f"seq_len:{seq_len} "
               f"img_args:{img_args} "
               f"free_mem:{free_mem}")
        logger.info(msg)

    def warmup_sampler(self):
        """
        Warmup the sampler with different temperature, top-p, and top-k values.
        """
        # Choose batch sizes for warmup based on bucketing
        test_batch_sizes = list(dict.fromkeys([0, 1] + [bucket[0] for bucket in self.bucketing_manager.decode_buckets]))

        # Test different sampling configurations
        sampling_configs = [
            # (temperature, top_p, top_k, batch_changed)
            (0.0, 1.0, 0, True),  # Greedy sampling
            (1.0, 1.0, 0, True),  # Random sampling with temp=1.0
            (0.7, 0.9, 50, True),  # Common creative settings
            (0.3, 0.95, 20, True),  # Conservative settings
            (1.2, 0.8, 100, True),  # High temperature settings
            (0.8, 0.85, 0, True),  # Different top-p sampling
            (0.0, 1.0, 0, False),  # Greedy sampling
            (1.0, 1.0, 0, False),  # Random sampling with temp=1.0
            (0.7, 0.9, 50, False),  # Common creative settings
            (0.3, 0.95, 20, False),  # Conservative settings
            (1.2, 0.8, 100, False),  # High temperature settings
            (0.8, 0.85, 0, False),  # Different top-p sampling
        ]

        logger.info("Warming up sampler with batch sizes: %s and following configs:", test_batch_sizes)
        for temp, top_p, top_k, batch_changed in sampling_configs:
            logger.info("temp=%s, top_p=%s, top_k=%s, batch_changed=%s", temp, top_p, top_k, batch_changed)
        logger.info("Starting sampler warmup...")

        for batch_size in test_batch_sizes:
            dummy_hidden_states = torch.randn(batch_size, self.hidden_size, dtype=self.dtype, device=self.device)
            if self.lora_config:
                lora_logits_mask = torch.zeros(batch_size,
                                               (self.lora_config.max_loras) * self.lora_config.max_lora_rank,
                                               dtype=self.lora_config.lora_dtype).to('hpu')
                LoraMask.setLoraMask(lora_logits_mask)
            dummy_logits = self.model.compute_logits(dummy_hidden_states)

            # Create dummy requests for this specific configuration
            dummy_req_ids = [f"warmup_req_{batch_size}_{i}" for i in range(max(1, batch_size))]

            for i, req_id in enumerate(dummy_req_ids):
                self.requests[req_id] = CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=list(range(10)),  # Dummy prompt
                    mm_features=[],
                    sampling_params=SamplingParams(),
                    pooling_params=None,
                    generator=None,
                    block_ids=[[0]],
                    num_computed_tokens=10,
                    output_token_ids=[],
                )
                self.input_batch.req_id_to_index[req_id] = i

            for temp, top_p, top_k, batch_changed in sampling_configs:
                # Add dummy requests to cache with consistent sampling params
                for i, req_id in enumerate(dummy_req_ids):
                    self.requests[req_id].sampling_params = SamplingParams(
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                    )

                    if temp == 0.0:  # Greedy sampling
                        self.input_batch.greedy_reqs.add(req_id)
                    else:  # Random sampling
                        self.input_batch.random_reqs.add(req_id)

                self.input_batch.req_output_token_ids = [
                    item[1] for item in self._generate_req_id_output_token_ids_lst(dummy_req_ids, pad_to=batch_size)
                ]
                self.input_batch.refresh_sampling_metadata()

                _sampler_output, _sampling_metadata = self._run_sampling(batch_changed=batch_changed,
                                                                         logits_device=dummy_logits,
                                                                         request_ids=dummy_req_ids,
                                                                         pad_to=dummy_logits.shape[0])

                # Cleanup after sampling
                self.input_batch.greedy_reqs = set()
                self.input_batch.random_reqs = set()
                self.input_batch.req_output_token_ids = []

            # Cleanup after batch has been warmed up
            self.input_batch.req_id_to_index = {}
            self.requests = {}

        # Final synchronization to ensure all operations are completed
        torch.hpu.synchronize()

        logger.info("Sampler warmup completed successfully")

    def warmup_defragmenter(self):
        """Warm up defragmentation swap graphs for different thresholds.

        We execute a minimal swap (1 pair) which will be padded internally to the
        requested threshold size. Thresholds chosen to mirror potential production
        values: 8, 16, 32, 64, 128, 256, 512.
        """
        # If defragmenter is disabled or cache utils not prepared, skip.
        if not getattr(self.defragmenter, 'enabled', False):
            return
        if self.defragmenter.cache_utils is None:
            return

        thresholds = self.defragmenter.to_swap_pad_thresholds

        logger.info("Warming up defragmenter with thresholds: %s", thresholds)

        # Use simple valid block ids present in caches (assume at least 2 blocks allocated when kv caches created)
        # We only need distinct ids for a swap. They will be scaled by block_size inside swap.
        # If for some reason only 1 block exists, skip warmup gracefully.
        try:
            k_cache = self.defragmenter.cache_utils.kv_caches[0][0]
            num_blocks_available = k_cache.shape[0] // self.block_size
        except Exception:
            num_blocks_available = 0
        if num_blocks_available < 2:
            logger.warning("Skipping defragmenter warmup, insufficient blocks (%s)", num_blocks_available)
            return

        # Minimal pair to trigger a swap path
        to_swap = [(1, 0)]

        for th in thresholds:
            self.defragmenter.cache_utils.swap(to_swap, th)

        # If the number of swaps was odd, do one more to make it even and return to original state.
        if len(thresholds) % 2 == 1:
            self.defragmenter.cache_utils.swap(to_swap, thresholds[0])

        logger.info("Defragmenter warmup completed successfully")

    def warmup_graphs(self, buckets, is_prompt, kv_caches, starting_mem=0, total_batch_seq=0.001):
        total_mem = starting_mem
        idx = 0
        num_candidates = len(buckets)
        captured_all = True
        for idx, (batch_size, seq_len, num_blocks) in enumerate(reversed(buckets)):
            if seq_len > self.max_num_tokens:
                continue
            # Graph memory usage is proportional to seq dimension in a batch
            phase = f"Graph/{'prompt' if is_prompt else 'decode'}"
            if is_prompt:
                batch_seq = batch_size * seq_len * num_blocks if num_blocks else batch_size * seq_len
            else:
                batch_seq = batch_size

            graphed_bucket = (batch_size, seq_len, num_blocks, is_prompt)
            if graphed_bucket in self.graphed_buckets:
                continue
            self.graphed_buckets.add(graphed_bucket)
            self.log_warmup(phase, idx, num_candidates, batch_size, seq_len, num_blocks)
            prompt_cfg, decode_cfg = None, None
            with HabanaMemoryProfiler() as mem_prof:
                if is_prompt:
                    prompt_cfg = (batch_size, seq_len, num_blocks)
                else:
                    decode_cfg = (batch_size, 1, num_blocks)
                self._prepare_dummy_scenario(prompt_cfg, decode_cfg)
            # TODO(kzawora): align_workers
            used_mem = mem_prof.consumed_device_memory
            total_mem += used_mem
            total_batch_seq += batch_seq

        return total_mem, total_batch_seq, captured_all

    def warmup_unified_graphs(self, buckets, kv_cache):
        idx = 0
        num_candidates = len(buckets)
        for idx, (query, shared_ctx, unique_ctx, is_causal) in enumerate(reversed(buckets)):
            unified_cfg = (query, shared_ctx, unique_ctx, is_causal)
            if unified_cfg in self.graphed_buckets:
                continue
            self.graphed_buckets.add(unified_cfg)
            self.log_warmup("Unified CFG", idx, num_candidates, query, shared_ctx, unique_ctx, is_causal)
            self._prepare_dummy_unified_scenario(unified_cfg)

    def _add_dummy_request(self,
                           requests,
                           num_scheduled_tokens,
                           num_computed_tokens,
                           total_tokens,
                           scheduled_tokens,
                           is_prompt,
                           block_id=0):
        num_blocks = round_up(total_tokens, self.block_size) // self.block_size
        prompt_token_ids = list(range(total_tokens))

        req_id = f'{len(requests)}'
        block_ids = [block_id] * num_blocks
        sampling_params = SamplingParams(temperature=0.0)

        req = NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=[block_ids],
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
        )
        requests.append(req)
        if is_prompt:
            num_scheduled_tokens[req_id] = len(prompt_token_ids) - num_computed_tokens
        else:
            num_scheduled_tokens[req_id] = scheduled_tokens

    def _add_dummy_unified_request(self, requests, is_prompt, is_unique, block_num, num_computed_tokens,
                                   num_scheduled_tokens, scheduled_tokens):
        from vllm.v1.core.sched.output import NewRequestData

        req_id = f'{len(requests)}'
        sampling_params = SamplingParams(temperature=0.0)
        num_computed_tokens = max(len(block_num) * self.block_size - 1, 0)

        if is_prompt:
            # num_computed_tokens + num_scheduled_tokens <= prompt token ids
            prompt_token_ids = num_computed_tokens + num_scheduled_tokens + 1
        else:
            # num_computed_tokens + num_scheduled_tokens > prompt_token_ids
            prompt_token_ids = num_computed_tokens + num_scheduled_tokens - 1
        prompt_token_ids = list(range(prompt_token_ids))

        req = NewRequestData(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=[block_num],
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
        )

        requests.append(req)
        scheduled_tokens[req_id] = num_scheduled_tokens

    @staticmethod
    def _generate_seq_lengths(num_samples, num_blocks, block_size):
        assert num_samples <= num_blocks
        blocks = [num_blocks // num_samples] * num_samples
        missing_blocks = num_blocks - sum(blocks)
        for i in range(missing_blocks):
            blocks[i] += 1
        seq_lengths = [b * block_size - 1 for b in blocks]
        return seq_lengths

    def distribute_sum_evenly(self, total_sum, max_length):
        '''
        Return a balanced list of ints that sums up to total_sum.
        List cannot be longer than max_length.
        '''
        base, remain = divmod(total_sum, max_length)
        result = [base] * max_length

        for i in range(remain):
            result[i] += 1

        return result

    def get_merged_prefill_seq_lens(self, query_len, ctx_blocks):
        '''
        Get seperate sequence lengths from merged layout to individual 
        samples.
        Returns list of sequence length (including query and context) and
        context lengths.
        '''
        ctx_list = self.distribute_sum_evenly(ctx_blocks, self.max_num_seqs)
        query_list = self.distribute_sum_evenly(query_len, self.max_num_seqs)
        prompt_list = [q + c * self.block_size for q, c in zip(query_list, ctx_list)]
        ctx_list = ctx_list if len(ctx_list) > 0 else [0] * len(prompt_list)
        return prompt_list, ctx_list

    def _prepare_dummy_unified_scenario(self, unified_cfg):
        requests: list[NewRequestData] = []
        scheduled_tokens: dict[str, int] = {}

        query_len, shared_ctx_len, unique_ctx_len, is_causal = unified_cfg
        num_computed_tokens = (shared_ctx_len + unique_ctx_len) * self.block_size

        if is_causal:
            decode_reqs_query = []
            decode_reqs_blocks = []
            prompt_reqs_query = []
            prompt_reqs_blocks: list = []

            all_shared_blocks_ids = [block for block in range(shared_ctx_len)]
            unique_block = unique_ctx_len - 1
            # do not use unique block id
            if unique_block in all_shared_blocks_ids:
                all_shared_blocks_ids.remove(unique_ctx_len - 1)
                all_shared_blocks_ids.append(shared_ctx_len + 1)

            #add unique
            if unique_ctx_len > 0:
                decode_reqs_query.append(1)
                decode_reqs_blocks.append([unique_ctx_len - 1])
            prompts_number = self.max_num_seqs - len(decode_reqs_query)
            remaining_query = query_len - sum(decode_reqs_query)

            q, r = divmod(remaining_query, prompts_number)
            prompt_reqs_query = [q + (1 if i < r else 0) for i in range(prompts_number)]
            prompt_reqs_blocks = [[] for _ in range(len(prompt_reqs_query))]
            for idx, query in enumerate(prompt_reqs_query):
                available_space_for_ctx = math.floor((self.max_model_len - query) // self.block_size)
                if len(all_shared_blocks_ids) >= available_space_for_ctx:
                    prompt_reqs_blocks[idx] = all_shared_blocks_ids[:available_space_for_ctx]
                    del all_shared_blocks_ids[:available_space_for_ctx]
                else:
                    prompt_reqs_blocks[idx] = all_shared_blocks_ids
                    break
            if unique_ctx_len > 0:
                self._add_dummy_unified_request(requests, False, True, [unique_ctx_len - 1], num_computed_tokens, 1,
                                                scheduled_tokens)

            for query, blocks in zip(prompt_reqs_query, prompt_reqs_blocks):
                self._add_dummy_unified_request(requests, True, False, blocks, num_computed_tokens, query,
                                                scheduled_tokens)
        else:
            remaining_samples = query_len
            base = shared_ctx_len // remaining_samples
            remain = shared_ctx_len % remaining_samples
            all_shared_blocks_ids = [block for block in range(shared_ctx_len)]
            unique_block = unique_ctx_len - 1
            # do not use unique block id
            if unique_block in all_shared_blocks_ids:
                all_shared_blocks_ids.remove(unique_ctx_len - 1)
                all_shared_blocks_ids.append(shared_ctx_len + 1)

            # distribute evenly across sublists
            split_shared_blocks_ids: list[list[int]] = [[] for _ in range(remaining_samples)]
            idx = 0
            for i in range(remaining_samples):
                size = base + (1 if i < remain else 0)
                for _ in range(size):
                    split_shared_blocks_ids[i].append(all_shared_blocks_ids[idx])
                    idx += 1

            # make sure that all blocks are shared = in at least two decodes
            for i, block in enumerate(all_shared_blocks_ids):
                target = (i + 1) % remaining_samples
                if block not in split_shared_blocks_ids[target]:
                    split_shared_blocks_ids[target].append(block)

            # add unique id
            if unique_ctx_len > 0:
                min_idx = min(range(remaining_samples), key=lambda j: len(split_shared_blocks_ids[j]))
                split_shared_blocks_ids[min_idx].append(unique_block)

            for i in range(len(split_shared_blocks_ids)):
                if not split_shared_blocks_ids[i]:
                    if unique_block - i >= 0:
                        split_shared_blocks_ids[i] = [unique_block - i]
                    else:
                        split_shared_blocks_ids[i] = [all_shared_blocks_ids[0]]

            for request_blocks in split_shared_blocks_ids:
                self._add_dummy_unified_request(requests, False, False, request_blocks, num_computed_tokens, 1,
                                                scheduled_tokens)

        self._execute_dummy_scenario(requests, scheduled_tokens)

    def _prepare_dummy_scenario(self, prompt_cfg, decode_cfg):
        requests: list[NewRequestData] = []
        scheduled_tokens: dict[str, int] = {}

        if prompt_cfg:
            prompt_bs, prompt_query_len, prompt_num_blocks = prompt_cfg
            prompt_ctx_len = prompt_num_blocks * self.block_size
            prompt_total_tokens = [prompt_query_len + prompt_ctx_len]
            prompt_num_context_blocks = [prompt_num_blocks]
            if self.max_model_len < sum(prompt_total_tokens) \
                and self.use_merged_prefill:
                # split query and ctx in merged prefill case
                prompt_total_tokens, prompt_num_context_blocks = \
                     self.get_merged_prefill_seq_lens(prompt_query_len,
                                                 prompt_num_blocks)
            for _ in range(prompt_bs):
                for tokens, context_len in zip(prompt_total_tokens, prompt_num_context_blocks):
                    self._add_dummy_request(requests,
                                            scheduled_tokens,
                                            num_computed_tokens=(context_len * self.block_size),
                                            total_tokens=tokens,
                                            scheduled_tokens=prompt_query_len,
                                            is_prompt=True)
        if decode_cfg:
            decode_bs, decode_query_len, decode_num_blocks = decode_cfg
            if self.use_contiguous_pa:
                decode_seq_lengths = [self.block_size] * decode_bs
                block_id = decode_num_blocks - 1
            else:
                decode_seq_lengths = self._generate_seq_lengths(decode_bs, decode_num_blocks, self.block_size)
                block_id = 0
            for dsl in decode_seq_lengths:
                self._add_dummy_request(requests,
                                        scheduled_tokens,
                                        num_computed_tokens=dsl,
                                        total_tokens=dsl,
                                        scheduled_tokens=1,
                                        is_prompt=False,
                                        block_id=block_id)
        self._execute_dummy_scenario(requests, scheduled_tokens)

    def _execute_dummy_scenario(self, requests, scheduled_tokens):
        from vllm.v1.core.sched.output import (SchedulerOutput, CachedRequestData)

        sched_output = SchedulerOutput(
            scheduled_new_reqs=requests,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=scheduled_tokens,
            total_num_scheduled_tokens=sum(scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=0,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        cleanup = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=0,
            finished_req_ids=set(req.req_id for req in requests),
            free_encoder_mm_hashes=[],
            structured_output_request_ids={},
            grammar_bitmask=None,
        )
        self.execute_model(sched_output, warmup_mode=True)
        self.execute_model(cleanup, warmup_mode=True)

    def _generate_profiling(self, prompt_cfg, decode_cfg):
        steps = 3
        profiler = setup_profiler(warmup=steps - 1, active=1)
        if prompt_cfg and prompt_cfg not in self.bucketing_manager.prompt_buckets:
            self.bucketing_manager.prompt_buckets.insert(0, prompt_cfg)
        elif decode_cfg and decode_cfg not in self.bucketing_manager.decode_buckets:
            self.bucketing_manager.decode_buckets.insert(0, decode_cfg)
        torch.hpu.synchronize()
        profiler.start()
        for _ in range(steps):
            self._prepare_dummy_scenario(prompt_cfg, decode_cfg)
            torch.hpu.synchronize()
            profiler.step()
        profiler.stop()

    @staticmethod
    def _parse_profile_cfg(profile_cfg):
        if profile_cfg:
            return tuple(map(int, profile_cfg.split(',')))
        return None

    @staticmethod
    def _parse_legacy_profile_cfg(profile_cfg):
        if profile_cfg:
            cfg = profile_cfg.split('_')
            assert cfg[0] in ['prompt', 'decode']
            return (cfg[0], int(cfg[1]), int(cfg[2]), cfg[3] == 't')
        return None

    def _read_profiling_cfg(self):
        prompt_cfg = self._parse_profile_cfg(os.environ.get('VLLM_PROFILE_PROMPT', None))
        decode_cfg = self._parse_profile_cfg(os.environ.get('VLLM_PROFILE_DECODE', None))
        legacy_cfg = self._parse_legacy_profile_cfg(os.environ.get('VLLM_PT_PROFILE', None))
        if legacy_cfg and not (prompt_cfg or decode_cfg):
            phase, bs, seq_or_blocks, use_graphs = legacy_cfg
            assert use_graphs != self.model_config.enforce_eager, \
                "'use_graphs' is out of sync with model config. " \
                "Either change the flag or change vllm engine parameters"
            if phase == 'prompt':
                prompt_cfg = (bs, seq_or_blocks, 0)
            else:
                decode_cfg = (bs, seq_or_blocks)
        # align with current bucketing
        if decode_cfg:
            decode_cfg = (decode_cfg[0], 1, decode_cfg[1])
        return prompt_cfg, decode_cfg

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        dummy_decoder_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_model_len,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_data = dummy_decoder_data.multi_modal_data

        # Result in the maximum GPU consumption of the model
        dummy_mm_item = dummy_mm_data[modality][0]
        dummy_mm_items = [dummy_mm_item] * max_items_per_batch

        self.model.model = cast(SupportsMultiModal, self.model.model)
        return next(mm_kwargs_group for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
            dummy_mm_items,
            device=self.device,
            pin_memory=self.pin_memory,
            merge_by_field_config=self.model.model.merge_by_field_config,
        ))

    def warmup_multimodal_graphs(self, buckets):

        phase = 'Graph/Multimodal'
        from vllm.v1.worker.utils import MultiModalBudget
        self.mm_budget = MultiModalBudget(
            self.model_config,
            self.scheduler_config,
            self.mm_registry,
        ) if self.supports_mm_inputs else None

        #self.mm_budget.mm_limits : {'image': 2}
        for modality, max_items in self.mm_budget.mm_limits.items():
            phase = f'Graph/Multimodal({modality})'
            num_candidates = len(buckets)

            for idx, img_arg in enumerate(buckets):
                # Create dummy batch of multimodal inputs.
                batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                    modality,
                    img_arg,
                )
                #htorch.core.mark_step()
                # Run multimodal encoder.
                dummy_encoder_outputs = \
                     self.model.get_multimodal_embeddings(
                     **batched_dummy_mm_inputs)
                #htorch.core.mark_step()

                sanity_check_mm_encoder_outputs(
                    dummy_encoder_outputs,
                    expected_num_items=img_arg,
                )

                self.graphed_buckets.add(img_arg)
                self.log_warmup_multimodal(phase, idx, num_candidates, 1, 0, img_arg)

    @torch.inference_mode()
    def warmup_model(self) -> None:
        if not self.enable_bucketing:
            return

        if self.unified_attn:
            self.bucketing_manager.generate_unified_buckets()
        else:
            self.bucketing_manager.generate_prompt_buckets()
            self.bucketing_manager.generate_decode_buckets()

            if self.supports_mm_inputs:
                # Delayed multimodal buckets during warmup until model is loaded.
                from vllm_gaudi.extension.bucketing.vision import HPUVisionBucketManager
                self.get_model().vision_bucket_manager = HPUVisionBucketManager(self.model_config.model)
                msg = (f"Multimodal bucket : {self.get_model().vision_bucket_manager.multimodal_buckets}")
                logger.info(msg)

            max_bucket = max(self.bucketing_manager.decode_buckets[-1][0], self.bucketing_manager.prompt_buckets[-1][0])
            if max_bucket > self.input_batch.max_num_reqs:
                input_batch_bkp = self.input_batch
                self.input_batch = InputBatch(
                    max_num_reqs=self.bucketing_manager.decode_buckets[-1][0],
                    max_model_len=self.max_model_len,
                    max_num_batched_tokens=self.max_num_tokens,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    vocab_size=self.model_config.get_vocab_size(),
                    block_sizes=[self.block_size],
                    kernel_block_sizes=[self.block_size],
                    logitsprocs=build_logitsprocs(self.vllm_config, self.device, self.pin_memory, self.is_pooling_model,
                                                  self.vllm_config.model_config.logits_processors),
                )

        self.defragmenter.initialize(self.kv_caches, self.block_size)

        prompt_profile_cfg, decode_profile_cfg = self._read_profiling_cfg()
        if prompt_profile_cfg or decode_profile_cfg:
            self._generate_profiling(prompt_profile_cfg, decode_profile_cfg)
            raise AssertionError("Finished profiling")
        kv_caches = self.kv_caches

        if not htorch.utils.internal.is_lazy() and not self.model_config.enforce_eager:
            multiplier = 5 if self.compile_config.regional_compilation else 1
            cache_size_limit = 1 + multiplier * (len(self.bucketing_manager.prompt_buckets) +
                                                 len(self.bucketing_manager.decode_buckets))
            torch._dynamo.config.cache_size_limit = max(cache_size_limit, torch._dynamo.config.cache_size_limit)
            # Multiply by 8 to follow the original default ratio between
            # the cache_size_limit and accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = max(cache_size_limit * 8,
                                                                    torch._dynamo.config.accumulated_cache_size_limit)

        if self.skip_warmup:
            logger.info("Skipping warmup...")
            return

        self.profiler.start('internal', 'warmup')
        start_mem = HabanaMemoryProfiler.current_device_memory_usage()
        start_time = time.perf_counter()

        # Most model's multimodal embedding has to be run without COMPILE ONLY mode.
        if self.supports_mm_inputs:
            self.warmup_multimodal_graphs(self.get_model().vision_bucket_manager.multimodal_buckets)

        compile_only_mode_context = functools.partial(bc.env_setting, "PT_COMPILE_ONLY_MODE", True)
        can_use_compile_only_mode = True
        try:
            with compile_only_mode_context():
                pass
            logger.debug("Using PT_COMPILE_ONLY_MODE.")
        except KeyError:
            can_use_compile_only_mode = False
            logger.warning('Cannot use PT_COMPILE_ONLY_MODE. '
                           'Warmup time will be negatively impacted. '
                           'Please update Gaudi Software Suite.')
        with compile_only_mode_context() if can_use_compile_only_mode else contextlib.nullcontext():
            if not self.model_config.enforce_eager:
                assert self.mem_margin is not None, \
                    ("HabanaWorker.determine_num_available_blocks needs "
                     "to be called before warming up the model.")

                self.warmup_sampler()
                self.warmup_defragmenter()

                # TODO(kzawora): align_workers
                if self.unified_attn:
                    self.warmup_unified_graphs(self.bucketing_manager.unified_buckets, kv_caches)
                else:
                    mem_post_prompt, prompt_batch_seq, prompt_captured_all = \
                        self.warmup_graphs(
                            self.bucketing_manager.prompt_buckets, True, kv_caches)
                    mem_post_decode, decode_batch_seq, decode_captured_all = \
                        self.warmup_graphs(
                            self.bucketing_manager.decode_buckets, False, kv_caches)

                    self.log_graph_warmup_summary(self.bucketing_manager.prompt_buckets, True, mem_post_prompt)
                    self.log_graph_warmup_summary(self.bucketing_manager.decode_buckets, False, mem_post_decode)

        end_time = time.perf_counter()
        end_mem = HabanaMemoryProfiler.current_device_memory_usage()
        if os.getenv('VLLM_FULL_WARMUP', 'false').strip().lower() in ("1", "true"):
            # Since the model is warmed up for all possible tensor sizes,
            # Dynamo can skip checking the guards
            torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        elapsed_time = end_time - start_time
        msg = (f"Warmup finished in {elapsed_time:.0f} secs, "
               f"allocated {format_bytes(end_mem - start_mem)} of device memory")
        logger.info(msg)
        self.profiler.end()

        if not self.unified_attn and max_bucket > self.input_batch.max_num_reqs:
            self.input_batch = input_batch_bkp
        # NOTE(kzawora): This is a nasty workaround - for whatever cache_utils-related reason,
        # reusing defragmenter used in warmup causes accuracy drops, which is why we re-create
        # and re-initialize it.
        self.defragmenter = OnlineDefragmenter()
        self.defragmenter.initialize(self.kv_caches, self.block_size)

    def shutdown_inc(self):
        can_finalize_inc = self._is_quant_with_inc() and \
            (self.model.model is not None) and \
            self.inc_initialized_successfully and \
            not self._is_inc_finalized
        if can_finalize_inc:
            from neural_compressor.torch.quantization import (finalize_calibration)
            finalize_calibration(self.model.model)
            self._is_inc_finalized = True

    def __del__(self):
        self.shutdown_inc()

    @torch.inference_mode()
    def profile_run(self) -> None:
        return

    def _dummy_run(self, max_num_batched_tokens: int) -> None:
        assert max_num_batched_tokens == 1
        # when P/D disagg used, add dummy prefill run for prefiller instance
        if has_kv_transfer_group() and self.vllm_config.kv_transfer_config.is_kv_producer:
            prompt_cfg = 1, 1, 1
            decode_cfg = None
        else:
            prompt_cfg = None
            decode_cfg = 1, 1, 1
        # add dummy run
        self._prepare_dummy_scenario(prompt_cfg, decode_cfg)
        return

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        if len(kv_cache_config.kv_cache_groups) > 1:
            raise NotImplementedError("Hybrid models with more than one KV cache type are not "
                                      "supported yet.")

        # build a map from layer_name -> KVCacheTensor
        tensor_map: dict[str, KVCacheTensor] = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            for lname in tensor.shared_by:
                tensor_map[lname] = tensor

        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_sizes = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1
            kv_cache_sizes[kv_cache_tensor.shared_by[0]] = kv_cache_tensor.size

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = \
                    kv_cache_tensor.size // kv_cache_spec.page_size_bytes
                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(num_blocks + 1, kv_cache_spec.block_size,
                                                                          kv_cache_spec.num_kv_heads,
                                                                          kv_cache_spec.head_size)
                    v_cache_shape = None if self.model_config.use_mla \
                        else kv_cache_shape
                    dtype = kv_cache_spec.dtype
                    key_cache = torch.zeros(kv_cache_shape, dtype=dtype, device=self.device)
                    if v_cache_shape is not None:
                        value_cache = torch.zeros(v_cache_shape, dtype=dtype, device=self.device)
                    else:
                        value_cache = None
                    for layer_name in kv_cache_tensor.shared_by:
                        kv_caches[layer_name] = (key_cache, value_cache)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")
            layer_names = set()
            for group in kv_cache_config.kv_cache_groups:
                layer_names.update(group.layer_names)
            assert layer_names == set(kv_caches.keys()), "Some layers are not correctly initialized"
        bind_kv_cache(kv_caches, self.vllm_config.compilation_config.static_forward_context, self.kv_caches)

        if self.enable_bucketing:
            self.bucketing_manager.num_hpu_blocks = num_blocks
        self._PAD_BLOCK_ID = num_blocks
        self._PAD_SLOT_ID = num_blocks * self.block_size

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(self.get_kv_caches_4D(kv_caches))
            if self.vllm_config.kv_transfer_config.kv_buffer_device == "cpu":
                get_kv_transfer_group().set_host_xfer_buffer_ops(copy_kv_blocks)
            global hpu_buffer
        htorch.hpu.synchronize()

    def get_kv_caches_4D(self, kv_caches) -> dict[str, torch.Tensor]:
        kv_caches_4D: dict[str, torch.Tensor] = {}
        for layer_name, cache_or_cachelist in kv_caches.items():
            kv_cache_per_layer = []
            for cache in cache_or_cachelist:
                if cache is None:
                    continue
                kv_cache_per_layer.append(cache.view(-1, self.block_size, *cache.shape[1:]))
                #NOTE(Chendi): Do not remove, call torch data_ptr to record physical address
                cache.data_ptr()
            kv_caches_4D[layer_name] = TensorTuple(tuple(kv_cache_per_layer)) \
                if len(kv_cache_per_layer) == 2 else kv_cache_per_layer[0]
        return kv_caches_4D

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def _get_nans_in_logits(
        self,
        logits: Optional[torch.Tensor],
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (int(num_nans_for_index[req_index])
                                              if num_nans_for_index is not None and req_index < logits.shape[0] else 0)
            return num_nans_in_logits
        except IndexError:
            return {}

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, \
                f"Config `{config_name}` not supported. " \
                f"Allowed configs: {allowed_config_names}"
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def reload_weights(self) -> None:
        assert getattr(self, "model", None) is not None, \
            "Cannot reload weights before model is loaded."
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model_loader.load_weights(self.model, model_config=self.model_config)
        torch.hpu.synchronize()

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        if isinstance(self._draft_token_ids, torch.Tensor):
            draft_token_ids = self._draft_token_ids.tolist()
        else:
            draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
        prefill_sampled_token_ids_tensor: torch.Tensor,
        decode_sampled_token_ids_tensor: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[torch.Tensor],
        hidden_states_prefills: list[torch.Tensor],
        sample_hidden_states_prefills: list[torch.Tensor],
        aux_hidden_states_prefills: list[Optional[torch.Tensor]],
        num_decodes: int,
        prefill_data: Optional[PrefillInputData] = None,
        decode_data: Optional[DecodeInputData] = None,
    ) -> Union[list[list[int]], torch.Tensor]:
        if self.speculative_config.method == "ngram":
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.propose_ngram_draft_token_ids(sampled_token_ids)
        elif self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)

            def execute_drafter_model(target_token_ids, target_positions, target_hidden_states, last_token_indices,
                                      common_attn_metadata):
                if self.drafter.method == "eagle3":
                    assert isinstance(self.drafter.model.model, Eagle3LlamaForCausalLM)
                    target_hidden_states = \
                        self.drafter.model.model.combine_hidden_states(
                        target_hidden_states)
                    assert target_hidden_states.shape[-1] == self.hidden_size
                #htorch.core.mark_step()
                ret_hidden_states = self.drafter.model(
                    input_ids=target_token_ids,
                    positions=target_positions,
                    hidden_states=target_hidden_states,
                    inputs_embeds=None,
                    attn_metadata=common_attn_metadata,
                )
                #htorch.core.mark_step()
                if self.drafter.method in ("deepseek_mtp", "ernie_mtp"):
                    last_hidden_states = ret_hidden_states
                    hidden_states = last_hidden_states
                else:
                    last_hidden_states, hidden_states = ret_hidden_states
                last_hidden_states = last_hidden_states.view(-1, last_hidden_states.shape[-1])
                sample_hidden_states = last_hidden_states[last_token_indices]
                logits = self.drafter.model.compute_logits(sample_hidden_states)
                draft_token_ids = logits.argmax(dim=-1)
                return draft_token_ids, hidden_states

            draft_token_ids = None
            if decode_data is not None:
                assert decode_data.spec_decode_metadata is not None
                assert decode_data.position_ids is not None
                num_draft_tokens = \
                    decode_data.spec_decode_metadata.num_draft_tokens
                max_num_draft_tokens = max(num_draft_tokens)
                common_attn_metadata = decode_data.attn_metadata

                num_picked_token_indices = []
                last_token_indices = []
                starting_index = 0
                num_rejected_tokens = [
                    n + 1 - len(sampled_token_ids[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)
                ]
                for i, n in enumerate(num_draft_tokens):
                    r = num_rejected_tokens[i]
                    step = max_num_draft_tokens + 1
                    for j in range(step):
                        if j == n - r:
                            last_token_indices.append(starting_index + j)
                        if j < n + 1 - r:
                            num_picked_token_indices.append(starting_index + j)
                        else:
                            num_picked_token_indices.append(-1)
                    starting_index += step
                hidden_states_indices = torch.tensor(num_picked_token_indices, device=self.device)
                last_token_indices = torch.tensor(last_token_indices, device=self.device)

                target_token_ids = decode_sampled_token_ids_tensor.reshape(-1, 1)[hidden_states_indices]
                target_positions = decode_data.position_ids[hidden_states_indices]

                if self.use_aux_hidden_state_outputs and \
                    aux_hidden_states is not None:
                    target_hidden_states = torch.cat([h[hidden_states_indices] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[hidden_states_indices]

                if target_hidden_states.dim() == 2:
                    target_hidden_states = target_hidden_states.unsqueeze(1)
                draft_token_ids, hidden_states = execute_drafter_model(target_token_ids, target_positions,
                                                                       target_hidden_states, last_token_indices,
                                                                       common_attn_metadata)

                draft_token_ids = draft_token_ids[:num_decodes]
            # handle prefill
            if prefill_data is not None:
                # Currently, prefill is done one by one
                draft_token_ids_prefill = []
                hidden_states_prefill = []

                for idx, (req_id, prompt_len, token_ids, position_ids, attn_metadata, logits_indices,
                          logits_requests) in enumerate(zip(*shallow_tuple(prefill_data))):
                    hidden_states = hidden_states_prefills[idx]
                    if self.use_aux_hidden_state_outputs:
                        aux_hidden_states = aux_hidden_states_prefills[idx]
                        target_hidden_states = torch.cat(aux_hidden_states, dim=-1)
                    else:
                        target_hidden_states = hidden_states
                    next_token_ids = prefill_sampled_token_ids_tensor[idx]
                    # Follow GPU to shift input_tokens by one to the left
                    # to match hidden_states
                    token_ids = token_ids.squeeze()
                    target_token_ids = token_ids.clone()
                    target_token_ids[:-1].copy_(token_ids[1:])
                    target_token_ids[logits_indices] = next_token_ids
                    target_token_ids = target_token_ids.unsqueeze(0)
                    if target_hidden_states.dim() == 2:
                        target_hidden_states = target_hidden_states.unsqueeze(0)
                    _draft_token_ids, _hidden_states = execute_drafter_model(target_token_ids, position_ids,
                                                                             target_hidden_states, logits_indices,
                                                                             attn_metadata)
                    draft_token_ids_prefill.append(_draft_token_ids)
                    hidden_states_prefill.append(_hidden_states)
                if draft_token_ids is None:
                    draft_token_ids = torch.cat(draft_token_ids_prefill, dim=0)
                    hidden_states = torch.cat(hidden_states_prefill, dim=0)
                else:
                    draft_token_ids = torch.cat([draft_token_ids] + draft_token_ids_prefill, dim=0)
                    hidden_states = torch.cat([hidden_states] + hidden_states_prefill, dim=0)

            # Early exit if there is only one draft token to be generated.
            # [batch_size, 1]

            if self.speculative_config.num_speculative_tokens == 1:
                return draft_token_ids.view(-1, 1)  # type: ignore

        return draft_token_ids

    def propose_ngram_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
    ) -> list[list[int]]:
        draft_token_ids = self.drafter.propose(sampled_token_ids, self.input_batch.req_ids,
                                               self.input_batch.num_tokens_no_spec, self.input_batch.token_ids_cpu,
                                               self.input_batch.spec_decode_unsupported_reqs)
        # swipe draft_token_ids_native replacing [] to [-1]
        for i in range(len(draft_token_ids)):
            if len(draft_token_ids[i]) == 0:
                draft_token_ids[i] = [-1]
        return draft_token_ids


# --- Helper Functions ---
def get_shape(data):
    """Recursively finds the shape of a nested tuple or list."""
    if isinstance(data, torch.Tensor):
        return data.shape

    if not isinstance(data, (list, tuple)):
        return ()  # End of a non-tensor branch

    if not data:
        return (0, )

    first_dim = len(data)
    sub_shape = get_shape(data[0])

    for item in data[1:]:
        if get_shape(item) != sub_shape:
            raise ValueError("Inconsistent dimensions: The structure is ragged.")

    return (first_dim, ) + sub_shape


def _find_tensors_and_validate(data, attr_name):
    """
    A generic helper to find all tensors and validate a specific attribute
    (like 'device' or 'dtype') ensuring they are all the same.
    """
    found_attr = None

    def find_tensors(nested_data):
        if isinstance(nested_data, torch.Tensor):
            yield nested_data
        elif isinstance(nested_data, (list, tuple)):
            for item in nested_data:
                yield from find_tensors(item)

    tensor_iterator = find_tensors(data)

    try:
        first_tensor = next(tensor_iterator)
        found_attr = getattr(first_tensor, attr_name)
    except StopIteration:
        return None  # No tensors found

    for tensor in tensor_iterator:
        current_attr = getattr(tensor, attr_name)
        if current_attr != found_attr:
            raise ValueError(f"Inconsistent {attr_name}: Found tensors with both '{found_attr}' and '{current_attr}'.")

    return found_attr


class TensorTuple(tuple):
    """
    A tuple subclass designed to hold nested torch.Tensors, providing
    .shape and .device properties.
    
    It ensures that the nested structure is not ragged and that all
    contained tensors reside on the same device.
    """

    _shape: tuple[int, ...]
    _device: Optional[torch.device]
    _dtype: Optional[torch.dtype]

    def __new__(cls, iterable):
        # First, we create the actual tuple object instance
        instance = super().__new__(cls, iterable)

        # Now, compute and attach the custom properties.
        # This is done here because tuples are immutable.
        # We store them with a leading underscore.
        instance._shape = get_shape(instance)
        instance._device = _find_tensors_and_validate(instance, 'device')
        instance._dtype = _find_tensors_and_validate(instance, 'dtype')

        return instance

    @property
    def shape(self):
        """Returns the shape of the nested tuple structure."""
        return self._shape

    @property
    def device(self):
        """
        Returns the torch.device of the tensors within the tuple.
        Returns None if no tensors are present.
        """
        return self._device

    @property
    def dtype(self):
        """Returns the torch.dtype of the tensors within the tuple."""
        return self._dtype
