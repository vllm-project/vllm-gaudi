import torch
from vllm.model_executor.models.ernie45_vl import (
    Ernie4_5VLMultiModalProcessor,
    Ernie4_5_VLProcessingInfo,
    Ernie4_5_VLDummyInputsBuilder,
    Ernie4_5_VLMoeForConditionalGeneration,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.model_executor.models.ernie45_vl_moe import Ernie4_5_VLMoeDecoderLayer, Ernie4_5_VLMoeMoE
from vllm.config import VllmConfig
from einops import rearrange
import types


@MULTIMODAL_REGISTRY.register_processor(
    Ernie4_5VLMultiModalProcessor,
    info=Ernie4_5_VLProcessingInfo,
    dummy_inputs=Ernie4_5_VLDummyInputsBuilder,
)
class HpuErnie4_5_VLMoeForConditionalGeneration(Ernie4_5_VLMoeForConditionalGeneration):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Ernie4_5_VisionTransformer -> Ernie4_5_VisionBlock -> Ernie4_5_VisionAttention
        # modify Ernie4_5_VisionAttention's forward
        for vision_block in self.vision_model.blocks:
            vision_block.attn.forward = types.MethodType(ernie4_5_visionattention_forward_hpu, vision_block.attn)

        # Ernie4_5_VLMoeForCausalLM -> Ernie4_5_VLMoeModel -> Ernie4_5_VLMoeDecoderLayer -> Ernie4_5_VLMoeMoE
        # modify Ernie4_5_VLMoeMoE's forward
        for decode_layer in self.language_model.model.layers:
            if isinstance(decode_layer, Ernie4_5_VLMoeDecoderLayer) and isinstance(decode_layer.mlp, Ernie4_5_VLMoeMoE):
                decode_layer.mlp.forward = types.MethodType(ernie4_5_vlmoemoe_forward_hpu, decode_layer.mlp)


def ernie4_5_vlmoemoe_forward_hpu(
    self,
    hidden_states: torch.Tensor,
    visual_token_mask: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_dim = hidden_states.shape[-1]
    hidden_states = hidden_states.view(-1, hidden_dim)

    if visual_token_mask is not None and visual_token_mask.cpu().all():  # WA for HPU: fallback to CPU
        # only vision modal input
        router_logits, _ = self.vision_experts_gate(hidden_states.to(dtype=torch.float32))
        final_hidden_states = self.vision_experts(hidden_states=hidden_states, router_logits=router_logits)
    elif visual_token_mask is not None and visual_token_mask.cpu().any():  # WA for HPU: fallback to CPU
        text_token_mask = ~visual_token_mask
        text_router_logits, _ = self.text_experts_gate(hidden_states.to(dtype=torch.float32))
        text_shared_output, text_experts_output = self.text_experts(hidden_states=hidden_states,
                                                                    router_logits=text_router_logits)
        vision_router_logits, _ = self.vision_experts_gate(hidden_states.to(dtype=torch.float32))
        vision_shared_output, vision_experts_output = self.vision_experts(hidden_states=hidden_states,
                                                                          router_logits=vision_router_logits)
        final_hidden_states = (text_shared_output * text_token_mask +
                               vision_shared_output * visual_token_mask if self.has_shared_experts else None,
                               text_experts_output * text_token_mask + vision_experts_output * visual_token_mask)
    else:
        # only text modal input
        text_router_logits, _ = self.text_experts_gate(hidden_states.to(dtype=torch.float32))

        final_hidden_states = self.text_experts(hidden_states=hidden_states, router_logits=text_router_logits)

    if self.has_shared_experts:
        # for shared_experts model
        final_hidden_states = final_hidden_states[0] + final_hidden_states[1]
    else:
        # for not shared_experts model
        final_hidden_states = final_hidden_states[1]

    if self.tp_size > 1:
        final_hidden_states = (self.text_experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states))

    return final_hidden_states.view(orig_shape)


def ernie4_5_visionattention_forward_hpu(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
) -> torch.Tensor:
    # [s, b, c] --> [s, b, head * 3 * head_dim]
    x, _ = self.qkv(x)

    # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
    q, k, v = self.split_qkv(x)

    q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v))
    if rotary_pos_emb is not None:
        qk_concat = torch.cat([q, k], dim=0)
        qk_rotated = self.apply_rotary_emb(
            qk_concat,
            rotary_pos_emb.cos(),
            rotary_pos_emb.sin(),
        )
        q, k = torch.chunk(qk_rotated, 2, dim=0)

    output = self.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    if len(output.shape) == 3:
        context_layer = rearrange(output, "b s ... -> s b ...").contiguous()
    else:
        context_layer = rearrange(output, "b s h d -> s b (h d)").contiguous()

    output, _ = self.proj(context_layer)
    return output
