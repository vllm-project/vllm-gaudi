from collections.abc import Callable
from enum import Enum
from functools import partial
import os
from typing import Union

from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner as MoERunnerBase, )
import torch
import vllm
import vllm.envs as envs
from vllm.config import get_current_vllm_config, get_current_vllm_config_or_none
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter, )
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    FusedTopKBiasRouter, )
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter, )
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter, )
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter, )
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import (
    RoutingSimulatorRouter, )
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter, )
from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOp
from vllm_gaudi.extension.runtime import get_config
from vllm_gaudi.utils import has_quant_config
from vllm_gaudi.v1.worker.hpu_dp_utils import dispatch_hidden_states, dispatch_tensor, get_hpu_dp_metadata

# Map model activation names to the activation strings the Gaudi fused-MoE op's
# MoeActivationMode_t enum accepts. The synapse MoE kernel enumerates "gelu" (and
# silu/selu) but NOT "gelu_tanh"; the Gaudi gelu TPC kernel is itself the tanh
# approximation (tpc_kernels .../gelu_f16.c uses tanh_f16), so HF's
# gelu_pytorch_tanh / gelu_tanh map to the op's "gelu" (the ~1e-3 exact-vs-tanh
# difference does not affect inference). Without this, Gemma4's MoE aborts at
# warmup_graphs with: Activation "gelu_tanh" not found among MoeActivationMode_t.
_MOE_ACTIVATION_ALIASES = {
    "gelu_tanh": "gelu",
    "gelu_pytorch_tanh": "gelu",
    "gelu_new": "gelu",
    "quick_gelu": "gelu",
}


def _normalize_moe_activation(activation):
    act = activation.value if isinstance(activation, Enum) else activation
    return _MOE_ACTIVATION_ALIASES.get(act, act)


# Activation string used by MiniMax-M3 experts. The Habana fused-MoE kernel
# (torch.ops.hpu.mixture_of_experts) only supports silu/gelu, so this
# activation is computed unfused in PyTorch instead (see _unfused_swigluoai_moe).
_SWIGLU_OAI_ACTIVATION = "swigluoai_uninterleave"

# The unfused SwiGLU-OAI dense expert pass materializes [E_local, T, 2*I]
# intermediates. For large prompt buckets (T up to the max batched-token bucket)
# a single such graph exceeds the Synapse compiler's limits and fails to compile
# with ``synStatus 26 [Generic failure]``. Tiling the token dimension bounds the
# size of every fused graph/op while keeping the math and per-tile shapes static,
# which the HPU graph compiler requires. Tunable via env for large-prompt tuning.
_SWIGLU_OAI_TOKEN_TILE = int(os.environ.get("VLLM_MINIMAX_M3_MOE_TOKEN_TILE", "512"))

# Non-gated activations (``is_act_and_mul=False``). Their w13 is a single
# up-projection of size I (not a 2*I gate+up split), and the elementwise
# activation is applied directly: y = w2(act(w1 x)). The Habana fused-MoE kernel
# is gated-only, so these are computed unfused (see _unfused_no_mul_moe). The
# key is the MoEActivation ``.value`` (e.g. "relu2_no_mul" for Nemotron-H).
_NO_MUL_ACTIVATIONS = {
    "silu_no_mul": lambda h: torch.nn.functional.silu(h),
    "gelu_no_mul": lambda h: torch.nn.functional.gelu(h),
    "gelu_tanh_no_mul": lambda h: torch.nn.functional.gelu(h, approximate="tanh"),
    "relu2_no_mul": lambda h: torch.square(torch.nn.functional.relu(h)),
}


def _unfused_swigluoai_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    limit: float | None,
) -> torch.Tensor:
    """Unfused expert MLP with SwiGLU-OAI activation for MiniMax-M3 on HPU.

    Replaces a single ``moe_op(x, topk_ids, topk_weights, activation=...)`` fused
    call for the ``swigluoai_uninterleave`` activation (unsupported by the Habana
    MoE kernel). Returns the same per-rank (pre-all-reduce) partial as the fused
    op, so the surrounding MoERunner reduction/combine pipeline is unchanged.

    A dense pass is used (every token through every local expert, masked by the
    routing weight) to keep static shapes, which the HPU graph compiler requires
    -- data-dependent gather/indexing would produce dynamic shapes. It is
    vectorized as a sync-free expert-chunked ``bmm`` over the stacked expert
    weights rather than a per-expert Python loop: the loop issues one matmul per
    expert and serializes the HPU graph (~0.3-1.4 tok/s), while the chunked
    ``bmm`` batches experts with static shapes (~11 tok/s). The SwiGLU-OAI
    activation is computed in fp32 for accuracy (matches
    ``minimax_m3_sparse.swiglu_oai``).

    Args:
        layer: the ``FusedMoE``/``RoutedExperts`` layer. Reads the stacked
            per-rank expert weights ``layer.w13_weight`` [E_local, 2*I, H] and
            ``layer.w2_weight`` [E_local, H, I] (the same tensors that seed
            ``moe_op.w13_list``) and ``layer.moe_config.ep_rank`` /
            ``layer.local_num_experts`` for this rank's expert-id offset.
        x: [T, H] flattened activations.
        topk_ids: [T, K] selected (global) expert ids.
        topk_weights: [T, K] routing weights.
    """
    w13 = layer.w13_weight  # [E_local, 2*I, H]
    w2 = layer.w2_weight  # [E_local, H, I]
    # First global expert id owned by this (EP) rank; matches the experts_min
    # that process_weights_after_loading passes to VllmMixtureOfExpertsOp.
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)

    e_local = w13.shape[0]
    d = w13.shape[1] // 2
    hidden = x.shape[-1]
    tokens = x.shape[0]

    # Per-(token, local-expert) combine weight, computed sync-free via scatter
    # (no host sync, unlike an ``== expert_id`` comparison inside a Python loop).
    local_ids = topk_ids - experts_min  # [T, K]
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    # Dense experts via expert-chunked bmm (static shapes, no host sync). Only
    # this rank's local experts contribute; cross-rank combine happens outside.
    # The token dimension is tiled so no single fused graph materializes the full
    # [E_local, T, 2*I] intermediate (which overflows the Synapse compiler and
    # fails with synStatus 26 for large prompt buckets); per-tile shapes stay
    # static and tiles are concatenated back into the original [T, H] output.
    chunk = 32
    token_tile = _SWIGLU_OAI_TOKEN_TILE
    if token_tile <= 0 or token_tile >= tokens:
        token_tile = tokens
    out_tiles = []
    for tstart in range(0, tokens, token_tile):
        tend = min(tstart + token_tile, tokens)
        xt = x[tstart:tend]  # [t, H]
        t = xt.shape[0]
        acc = torch.zeros_like(xt)  # [t, H]
        for start in range(0, e_local, chunk):
            end = min(start + chunk, e_local)
            c = end - start
            w13c = w13[start:end].transpose(1, 2)  # [c, H, 2I]
            w2c = w2[start:end].transpose(1, 2)  # [c, I, H]
            xe = xt.unsqueeze(0).expand(c, t, hidden)  # [c, t, H]
            h = torch.bmm(xe, w13c)  # [c, t, 2I]
            # SwiGLU-OAI in fp32 for accuracy: gate=first half, up=second half.
            g = h[..., :d].float()
            u = h[..., d:].float()
            if limit is not None:
                g = g.clamp(max=limit)
                u = u.clamp(min=-limit, max=limit)
            act = (g * torch.sigmoid(alpha * g) * (u + beta)).to(x.dtype)  # [c, t, I]
            y = torch.bmm(act, w2c)  # [c, t, H] per-rank partial
            gate_wc = gate_w[tstart:tend, start:end].t().unsqueeze(-1)  # [c, t, 1]
            acc = acc + (y.float() * gate_wc).sum(0).to(x.dtype)
        out_tiles.append(acc)
    return out_tiles[0] if len(out_tiles) == 1 else torch.cat(out_tiles, dim=0)


def _unfused_no_mul_moe(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    act_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Unfused expert MLP for non-gated (``is_act_and_mul=False``) MoE on HPU.

    The Habana fused-MoE kernel (``torch.ops.hpu.mixture_of_experts``) only
    implements the gated pattern ``act(gate) * up`` over a ``2*I`` w13
    projection. Non-gated models (e.g. Nemotron-H, which uses squared-ReLU) have
    a single ``I``-wide w13 up-projection with the activation applied directly:
    ``y = w2(act(w1 x))``. This computes that unfused, mirroring
    ``_unfused_swigluoai_moe``: a dense, sync-free, expert-chunked + token-tiled
    ``bmm`` pass with static shapes for the HPU graph compiler, returning the
    same per-rank (pre-all-reduce) partial the fused op would.

    Args:
        layer: the ``FusedMoE``/``RoutedExperts`` layer. Reads the stacked
            per-rank expert weights ``layer.w13_weight`` [E_local, I, H] and
            ``layer.w2_weight`` [E_local, H, I] and ``layer.moe_config.ep_rank`` /
            ``layer.local_num_experts`` for this rank's expert-id offset.
        x: [T, H] flattened activations.
        topk_ids: [T, K] selected (global) expert ids.
        topk_weights: [T, K] routing weights.
        act_fn: elementwise activation applied to the w13 projection (fp32).
    """
    w13 = layer.w13_weight  # [E_local, I, H]
    w2 = layer.w2_weight  # [E_local, H, I]
    # First global expert id owned by this (EP) rank; matches experts_min in
    # process_weights_after_loading.
    experts_min = int(layer.moe_config.ep_rank * layer.local_num_experts)

    e_local = w13.shape[0]
    hidden = x.shape[-1]
    tokens = x.shape[0]

    # Per-(token, local-expert) combine weight, computed sync-free via scatter
    # (no host sync). Mirrors _unfused_swigluoai_moe.
    local_ids = topk_ids - experts_min  # [T, K]
    in_range = (local_ids >= 0) & (local_ids < e_local)
    safe_ids = torch.where(in_range, local_ids, torch.zeros_like(local_ids))
    gate_w = x.new_zeros(tokens, e_local, dtype=torch.float32)
    gate_w.scatter_add_(
        1,
        safe_ids,
        torch.where(in_range, topk_weights, torch.zeros_like(topk_weights)).to(torch.float32),
    )

    # Dense experts via expert-chunked bmm (static shapes, no host sync); the
    # token dimension is tiled so no single fused graph materializes the full
    # [E_local, T, I] intermediate (which overflows the Synapse compiler for
    # large prompt buckets). Per-tile shapes stay static.
    chunk = 32
    token_tile = _SWIGLU_OAI_TOKEN_TILE
    if token_tile <= 0 or token_tile >= tokens:
        token_tile = tokens
    out_tiles = []
    for tstart in range(0, tokens, token_tile):
        tend = min(tstart + token_tile, tokens)
        xt = x[tstart:tend]  # [t, H]
        t = xt.shape[0]
        acc = torch.zeros_like(xt)  # [t, H]
        for start in range(0, e_local, chunk):
            end = min(start + chunk, e_local)
            c = end - start
            w13c = w13[start:end].transpose(1, 2)  # [c, H, I]
            w2c = w2[start:end].transpose(1, 2)  # [c, I, H]
            xe = xt.unsqueeze(0).expand(c, t, hidden)  # [c, t, H]
            h = torch.bmm(xe, w13c)  # [c, t, I]
            # Non-gated activation in fp32 for accuracy (matches the fp32
            # combine below), then cast back before the down-projection.
            act = act_fn(h.float()).to(x.dtype)  # [c, t, I]
            y = torch.bmm(act, w2c)  # [c, t, H] per-rank partial
            gate_wc = gate_w[tstart:tend, start:end].t().unsqueeze(-1)  # [c, t, 1]
            acc = acc + (y.float() * gate_wc).sum(0).to(x.dtype)
        out_tiles.append(acc)
    return out_tiles[0] if len(out_tiles) == 1 else torch.cat(out_tiles, dim=0)


def model_has_quant_config() -> bool:
    """Whether the active model runs with a MoE quantization config.

    After upstream PR #41184 the ``layer`` reaching ``apply_monolithic`` is a
    ``RoutedExperts`` instance, which no longer carries ``vllm_config`` (that
    field belonged to the old top-level ``FusedMoE``). The model config must
    therefore be resolved from the global vLLM config instead of off ``layer``.

    This MUST be called at build time (e.g. in ``__init__`` /
    ``process_weights_after_loading``), where the vLLM config context is set:
    the result is static per run, so callers cache it and read the cached flag
    on the forward hot path. ``get_current_vllm_config_or_none`` is used so the
    function degrades to ``False`` instead of raising if no context is active.

    Returns:
        ``True`` when the active model config declares a MoE quant config.
    """
    vllm_config = get_current_vllm_config_or_none()
    model_config = vllm_config.model_config if vllm_config is not None else None
    return model_config is not None and has_quant_config(model_config)


def select_experts_from_routed(layer, hidden_states: torch.Tensor,
                               router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Route tokens to experts for the grouped/custom-routing monolithic path.

    After upstream PR #41184 the ``layer`` passed to ``apply_monolithic`` is a
    ``RoutedExperts`` instance, which no longer owns a ``.router`` object (the
    router moved onto ``MoERunner``). ``RoutedExperts`` does, however, carry all
    the routing parameters, so we reproduce upstream's behaviour via the
    standalone ``select_experts`` helper. It is imported lazily because
    ``cpu_fused_moe`` registers a CPU custom op at module import time.

    Args:
        layer: The ``RoutedExperts`` instance holding the routing parameters.
        hidden_states: Flattened input activations.
        router_logits: Gate logits for the current tokens.

    Returns:
        A ``(topk_weights, topk_ids)`` tuple.
    """
    from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts

    return select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=layer.top_k,
        use_grouped_topk=layer.use_grouped_topk,
        renormalize=layer.renormalize,
        topk_group=layer.topk_group,
        num_expert_group=layer.num_expert_group,
        custom_routing_function=layer.custom_routing_function,
        scoring_func=layer.scoring_func,
        routed_scaling_factor=layer.routed_scaling_factor,
        e_score_correction_bias=layer.e_score_correction_bias,
    )


@UnquantizedFusedMoEMethod.register_oot
class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dispatch_fn = get_config().use_dispatch_fn
        # Snapshot the (static) quant-config flag while the vLLM config context
        # is set; the forward hot path reads this cached value instead.
        self.has_moe_quant_config = model_has_quant_config()
        torch.hpu.synchronize()
        vllm_config = get_current_vllm_config()
        self.model_type = None
        if (vllm_config is not None and vllm_config.model_config is not None
                and vllm_config.model_config.hf_config is not None):
            self.model_type = vllm_config.model_config.hf_config.model_type

        # SwiGLU-OAI activation params (MiniMax-M3). Cached here (config context
        # active) and read on the forward hot path when the fused kernel cannot
        # provide the activation. Falls back to the MiniMax-M3 defaults.
        tc = None
        if vllm_config is not None and vllm_config.model_config is not None:
            tc = getattr(vllm_config.model_config, "hf_text_config", None)
            if tc is None:
                tc = vllm_config.model_config.hf_config
        self.swiglu_alpha = float(getattr(tc, "swiglu_alpha", 1.702))
        self.swiglu_beta = float(getattr(tc, "swiglu_beta", 1.0))
        _limit = getattr(tc, "swiglu_limit", 7.0)
        self.swiglu_limit = None if _limit is None else float(_limit)

    def _select_monolithic(self) -> Callable:
        """Overriding base method"""
        return self.apply_monolithic

    @property
    def is_monolithic(self) -> bool:
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.moe_config.ep_rank * num_experts
        has_bias = hasattr(layer, "w13_bias") and hasattr(layer, "w2_bias")

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1

        if layer.moe_config.dp_size > 1 and self.use_dispatch_fn:
            dispatch_fn = partial(dispatch_hidden_states, is_sequence_parallel=layer.moe_config.is_sequence_parallel)
        else:
            dispatch_fn = None

        bias = has_bias if has_bias is True else None

        is_bf16 = getattr(layer, "w13_weight", None) is not None and layer.w13_weight.dtype == torch.bfloat16

        is_unquantized = not self.has_moe_quant_config

        cache_weight_lists = bool(is_bf16 and is_unquantized)

        # Pass cache flag into moe_op (requires ops.py __init__ signature update)
        layer.moe_op = VllmMixtureOfExpertsOp(layer.global_num_experts, num_experts, experts_min, experts_max, bias,
                                              dispatch_fn)

        for expert_id in range(layer.local_num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(layer.w13_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(layer.w2_weight.data[expert_id])
            if has_bias:
                layer.moe_op.w13_list[expert_id].set_bias(layer.w13_bias.data[expert_id])
                layer.moe_op.w2_list[expert_id].set_bias(layer.w2_bias.data[expert_id])

        # Build cache once AFTER weights/bias are set (BF16 + unquantized only)
        if cache_weight_lists and hasattr(layer.moe_op, "_cache_weight_lists"):
            layer.moe_op._cache_weight_lists()

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F

            if self.model_type == "gpt_oss":
                topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
            else:
                topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
                topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # The HPU mixture_of_experts kernel compiles for bf16 (x.dtype) router
        # weights and int64 routing tables. The grouped-topk / custom-routing
        # helper returns float32 weights and int32 ids; the regular-topk path
        # above already normalized them, but the grouped path previously left
        # them unconverted -> the bf16 MoE kernel graph received a float32
        # router_weights tensor and failed to compile (synStatus 26). Normalize
        # for every routing path so the kernel inputs are dtype-consistent.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)
        elif layer.moe_config.is_sequence_parallel:
            # Sequence-parallel MoE without data parallelism (TP>1 + EP with the
            # allgather_reducescatter backend => use_sequence_parallel_moe). The
            # model block chunks tokens by tp_size before the experts and
            # all-gathers afterwards, expecting `self.experts` to be
            # token-neutral. Upstream MoERunner._maybe_combine (mirrored in
            # patched_fused_moe_forward) reduce-scatters the expert output over
            # the EP group whenever is_sequence_parallel is set, regardless of
            # dp_size. The paired dispatch all-gather is set up via dispatch_fn
            # only for dp_size > 1, so at dp_size == 1 the combine is unpaired
            # and halves the token count -> the block's final reshape fails.
            # All-gather x / topk over the EP group here to restore the
            # dispatch/combine symmetry (dp_metadata is None at dp_size == 1, so
            # dispatch_tensor allocates its own EP-sized output).
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        activation = _normalize_moe_activation(layer.activation)
        if activation == _SWIGLU_OAI_ACTIVATION:
            # Habana fused MoE lacks SwiGLU-OAI; compute the experts unfused.
            output = _unfused_swigluoai_moe(
                layer,
                x,
                topk_ids,
                topk_weights,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
        elif activation in _NO_MUL_ACTIVATIONS:
            # Non-gated (is_act_and_mul=False) experts; the fused kernel is
            # gated-only, so compute y = w2(act(w1 x)) unfused.
            output = _unfused_no_mul_moe(
                layer,
                x,
                topk_ids,
                topk_weights,
                _NO_MUL_ACTIVATIONS[activation],
            )
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)

    def forward_oot(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        **kwargs,
    ):
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if layer.use_grouped_topk or getattr(layer, "custom_routing_function", None) is not None:
            topk_weights, topk_ids = select_experts_from_routed(layer, x, router_logits)
        else:
            import torch.nn.functional as F

            if self.model_type is not None and self.model_type in ["gpt_oss"]:
                topk_weights, topk_ids = torch.topk(router_logits, layer.top_k, dim=-1)
                topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
            else:
                topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
                topk_weights, topk_ids = torch.topk(topk_weights, layer.top_k, dim=-1)
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)

        # See apply_monolithic: the bf16 HPU MoE kernel needs int64 routing
        # tables and x.dtype router weights. Normalize for every routing path
        # (grouped-topk / custom routing returns int32 + float32) so the kernel
        # graph receives dtype-consistent inputs and compiles.
        topk_ids = topk_ids.to(torch.int64)
        topk_weights = topk_weights.to(x.dtype)

        if layer.moe_config.dp_size > 1:
            dp_metadata = get_hpu_dp_metadata()
            if not (self.has_moe_quant_config and self.use_dispatch_fn):
                hidden_states_across_dp = dp_metadata.hidden_states_across_dp if dp_metadata is not None else None
                x = dispatch_tensor(x, hidden_states_across_dp, layer.moe_config.is_sequence_parallel)

            topk_ids_across_dp = dp_metadata.topk_ids_across_dp if dp_metadata is not None else None
            topk_ids = dispatch_tensor(topk_ids, topk_ids_across_dp, layer.moe_config.is_sequence_parallel)

            topk_weights_across_dp = dp_metadata.topk_weights_across_dp if dp_metadata is not None else None
            topk_weights = dispatch_tensor(topk_weights, topk_weights_across_dp, layer.moe_config.is_sequence_parallel)
        elif layer.moe_config.is_sequence_parallel:
            # See apply_monolithic: at dp_size == 1 with sequence-parallel MoE
            # (TP>1 + EP), MoERunner._maybe_combine reduce-scatters the expert
            # output over the EP group but no paired dispatch all-gather runs
            # (dispatch_fn is wired only for dp_size > 1). Restore symmetry by
            # all-gathering the inputs over the EP group so the combine leaves
            # the token count unchanged for the block's post-experts reshape.
            x = dispatch_tensor(x, None, is_sequence_parallel=True)
            topk_ids = dispatch_tensor(topk_ids, None, is_sequence_parallel=True)
            topk_weights = dispatch_tensor(topk_weights, None, is_sequence_parallel=True)

        topk_ids = topk_ids.view(-1, topk_ids.shape[-1])
        topk_weights = topk_weights.view(-1, topk_weights.shape[-1])

        if self.model_type in ["gpt_oss"]:
            gpt_oss_out = layer.moe_op(
                x,
                topk_ids.to(torch.int64),
                topk_weights.to(x.dtype),
                permuted_weights=True,
                activation=_normalize_moe_activation(layer.activation),
            )
            if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
                return gpt_oss_out.view(*(gpt_oss_out.size(0), *input_shape[1:]))
            return gpt_oss_out.view(*input_shape)

        activation = _normalize_moe_activation(layer.activation)
        if activation == _SWIGLU_OAI_ACTIVATION:
            # Habana fused MoE lacks SwiGLU-OAI; compute the experts unfused.
            output = _unfused_swigluoai_moe(
                layer,
                x,
                topk_ids,
                topk_weights,
                alpha=self.swiglu_alpha,
                beta=self.swiglu_beta,
                limit=self.swiglu_limit,
            )
        elif activation in _NO_MUL_ACTIVATIONS:
            # Non-gated (is_act_and_mul=False) experts; the fused kernel is
            # gated-only, so compute y = w2(act(w1 x)) unfused.
            output = _unfused_no_mul_moe(
                layer,
                x,
                topk_ids,
                topk_weights,
                _NO_MUL_ACTIVATIONS[activation],
            )
        else:
            output = layer.moe_op(
                x,
                topk_ids,
                topk_weights,
                permuted_weights=True,
                activation=activation,
            )
        if layer.moe_config.dp_size > 1 or layer.moe_config.is_sequence_parallel:
            return output.view(*(output.size(0), *input_shape[1:]))
        else:
            return output.view(*input_shape)


def patched_fused_moe_forward(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Patched forward that avoids graph breaks from ForwardContext lookups
    and dynamo per-layer string guards.

    Instead of calling _forward_impl (which uses _sequence_parallel_context
    and _maybe_dispatch — both of which access ForwardContext and cause
    torch.compile graph breaks), for dp_size==1 we inline the quant-config
    init, gate application, _apply_quant_method and _maybe_combine directly.
    This also bypasses self.layer_name (a per-layer string) so dynamo no
    longer emits per-layer string guards that trigger recompilation.

    After upstream PR #41184 (FusedMoE/MoERunner inversion), `self` IS the
    MoERunner: the expert weights and quant_method live on
    self.routed_experts, quant init moved to
    routed_experts._ensure_moe_quant_config_init(), and _apply_quant_method
    no longer takes a `layer` argument. The post-forward reduction sequence
    mirrors upstream MoERunner.forward so we stay in sync with the shared/
    fused output combination logic.
    """
    hidden_states, shared_experts_input = self.apply_routed_input_transform(hidden_states)
    # Upstream _maybe_pad_hidden_states returns a 3-tuple: the (possibly padded)
    # hidden_states plus separate pre-transform and post-transform truncation
    # sizes (either may be None). Mirror MoERunner.forward, which keeps them
    # distinct — pre_xform strips fused_output before the routed-output
    # transform, post_xform strips the final output after all-reduce.
    hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (self._maybe_pad_hidden_states(
        shared_experts_input, hidden_states))

    if self.moe_config.dp_size == 1:
        # Bypass _forward_impl entirely for dp_size==1 to eliminate
        # graph breaks from _sequence_parallel_context() (which calls
        # get_forward_context()), skip the no-op _maybe_dispatch(), and
        # avoid double gate / stream-sync calls that _forward_impl
        # would redundantly repeat.
        if self.moe_config.pcp_size > 1:
            raise RuntimeError("dp_size==1 fast path does not support pcp_size > 1")
        # Mirror MoERunner._forward_impl's pre-dispatch setup: the quant config
        # is initialized on routed_experts (which the runner holds directly), so
        # unlike the old FusedMoE-layer-based init we do NOT need the layer here.
        self.routed_experts._ensure_moe_quant_config_init()
        self._maybe_sync_shared_experts_stream(shared_experts_input)
        # Apply the gate if the runner holds it (mirrors _forward_impl).
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = torch.nn.functional.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)
        # Core MoERunner._apply_quant_method takes no layer argument — it reads
        # everything it needs off the runner (self.routed_experts / self.router).
        # Call it exactly as upstream _forward_impl does.
        shared_output, fused_hidden = self._apply_quant_method(
            hidden_states=hidden_states,
            router_logits=router_logits,
            shared_experts_input=shared_experts_input,
            input_ids=input_ids,
        )
        result = self._maybe_combine(shared_output, fused_hidden)
    else:
        # Upstream PR #41184 dropped MoERunner._trtllm_mxfp4_unpadded_dim(); the
        # TRT-LLM MXFP4 unpadded hidden dim now comes from moe_config and is only
        # non-zero when the quant method produces unpadded output. Mirror
        # MoERunner.forward exactly so we stay in sync with the custom-op signature.
        hidden_dim_unpadded = (self.moe_config.hidden_dim_unpadded if self._quant_method.has_unpadded_output else 0)
        result = self._forward_entry(hidden_states, router_logits, shared_experts_input, input_ids,
                                     self._encode_layer_name(), hidden_dim_unpadded)

    # Mirror upstream MoERunner.forward post-_forward_entry pipeline.
    if isinstance(result, tuple):
        shared_output, fused_output = result
    else:
        shared_output, fused_output = None, result

    # Trim padding from fused_output before the routed-output transform, matching
    # upstream MoERunner.forward (latent MoE with shared experts).
    if og_hidden_dim_pre_xform is not None:
        fused_output = fused_output[..., :og_hidden_dim_pre_xform]

    shared_output = self._maybe_reduce_shared_expert_output(shared_output)
    shared_output, fused_output = self._maybe_apply_routed_scale_to_output(shared_output, fused_output)
    fused_output = self.apply_routed_output_transform(fused_output)

    combined = (shared_output + fused_output) if shared_output is not None else fused_output

    combined = self._maybe_reduce_final_output(combined, og_hidden_dim_post_xform)
    return self._maybe_add_zero_expert_output(combined)


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    This implementation uses a standard Python loop, which is compatible with
    graph compilation modes that do not support dynamic shapes resulting from
    operations like `torch.where`.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping a global expert index to its local index. Contains -1 for
            experts that are not assigned to the current rank.

    Returns:
        str: A string mapping from local to global index,
        ordered by global index.
            (e.g., "0->5, 1->12, 2->23")
    """
    mappings = []
    # A standard loop over a tensor with a known shape is statically analyzable.
    # `enumerate` provides the global_index (the position in the tensor) and
    # `local_index_tensor` (the value at that position).
    for global_index, local_index_tensor in enumerate(expert_map):
        local_index = local_index_tensor.item()
        # We only build strings for valid experts (those not marked as -1).
        if local_index != -1:
            mappings.append(f"{local_index}->{global_index}")

    return ", ".join(mappings)


def create_fused_moe_router(
    # common parameters
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    # grouped topk parameters
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    shared_expert_weight: float = 1.0,
    # grouped topk + fused topk bias parameters
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    # custom routing parameters
    custom_routing_function: Callable | None = None,
    # eplb parameters
    eplb_state: EplbLayerState | None = None,
    # zero expert parameters
    zero_expert_type: str | None = None,
    num_logical_experts: int | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> FusedMoERouter:
    """
    Factory function to create the appropriate FusedMoERouter subclass based on
    the provided parameters.

    The selection logic follows this priority order:
    1. RoutingSimulatorRouter - if VLLM_MOE_ROUTING_SIMULATION_STRATEGY env var is set
    2. ZeroExpertRouter - if zero_expert_type is not None
    3. GroupedTopKRouter - if use_grouped_topk is True
    4. CustomRoutingRouter - if custom_routing_function is not None
    5. FusedTopKBiasRouter - if e_score_correction_bias is not None
    6. FusedTopKRouter - default fallback

    Common arguments:
        top_k: Number of experts to select per token
        global_num_experts: Total number of experts in the model
        renormalize: Whether to renormalize the routing weights

    Grouped topk arguments:
        use_grouped_topk: Whether to use grouped top-k routing
        num_expert_group: Number of expert groups (for grouped routing)
        topk_group: Top-k within each group (for grouped routing)
        scoring_func: Scoring function to use ("softmax" or "sigmoid")
        num_fused_shared_experts: Number of fused shared experts (for ROCm AITER)
        shared_expert_weight: Weight of the fused shared-expert slot (upstream
            vLLM d039c171+). Accepted for signature parity; unused on HPU since
            the fused-shared-expert path is not taken (default 1.0 no-op).

    Grouped topk and fused topk bias arguments:
        routed_scaling_factor: Scaling factor for routed weights
        e_score_correction_bias: Optional bias correction for expert scores

    Custom routing arguments:
        custom_routing_function: Optional custom routing function

    EPLB arguments:
        eplb_state: EPLB (Expert Parallelism Load Balancing) state

    Zero expert arguments:
        zero_expert_type: Type of zero expert (e.g. identity). If not None,
            creates a ZeroExpertRouter.
        num_logical_experts: Number of real (non-zero) experts. Required when
            zero_expert_type is not None.

    Hash Indices Table:
        hash_indices_table: Used to map input_ids to experts, needed for
            Deepseek V4

    Returns:
        An instance of the appropriate FusedMoERouter subclass
    """

    routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
    if routing_strategy != "":
        return RoutingSimulatorRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )

    if zero_expert_type is not None:
        assert num_logical_experts is not None, "num_logical_experts is required when zero_expert_type is set"
        assert e_score_correction_bias is not None, "e_score_correction_bias is required when zero_expert_type is set"
        return ZeroExpertRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            num_logical_experts=num_logical_experts,
            zero_expert_type=zero_expert_type,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )

    if use_grouped_topk:
        assert custom_routing_function is None
        if num_expert_group is None or topk_group is None:
            raise ValueError("num_expert_group and topk_group must be provided when use_grouped_topk is True")
        grouped_topk_router = GroupedTopKRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            renormalize=renormalize,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=num_fused_shared_experts,
        )
        return grouped_topk_router

    if custom_routing_function is not None:
        return CustomRoutingRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            custom_routing_function=custom_routing_function,
            renormalize=renormalize,
        )

    assert scoring_func in ["sigmoid", "softmax", "sqrtsoftplus"]

    if e_score_correction_bias is not None or hash_indices_table is not None:
        return FusedTopKBiasRouter(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            e_score_correction_bias=e_score_correction_bias,
            scoring_func=scoring_func,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            hash_indices_table=hash_indices_table,
            num_fused_shared_experts=num_fused_shared_experts,
            shared_expert_weight=shared_expert_weight,
        )

    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        eplb_state=eplb_state,
        renormalize=renormalize,
        scoring_func=scoring_func,
    )


# Apply patches
_orig_default_moe_runner_forward = MoERunnerBase.forward

# When enabled, bypasses the opaque torch.ops.vllm.moe_forward_shared custom
# op wrapper so that torch.ops.hpu.mixture_of_experts is captured directly in
# compiled Synapse graphs instead of running eagerly.
# Set HPU_FUSED_MOE=0 to disable and fall back to the original path.
_MOE_COMPILE = os.getenv("HPU_FUSED_MOE", "1") == "1"


def _patched_default_moe_runner_forward(self, *args, **kwargs):
    if _MOE_COMPILE:
        return patched_fused_moe_forward(self, *args, **kwargs)
    return _orig_default_moe_runner_forward(self, *args, **kwargs)


MoERunnerBase.forward = _patched_default_moe_runner_forward

# Note: after upstream PR #41184, FusedMoE is a factory function (not an
# nn.Module subclass), so there is no FusedMoE.__init__ to patch. The
# dp_size==1 fast path in patched_fused_moe_forward operates directly on the
# MoERunner (self.routed_experts / self.gate), so no stashed layer reference
# is required.

vllm.model_executor.layers.fused_moe.layer.get_compressed_expert_map = get_compressed_expert_map
vllm.model_executor.layers.fused_moe.router.router_factory.create_fused_moe_router = create_fused_moe_router
vllm.model_executor.layers.fused_moe.layer.create_fused_moe_router = create_fused_moe_router

# Enable non-gated (is_act_and_mul=False) MoE on HPU.
#
# vLLM's FusedMoEConfig.__post_init__ raises NotImplementedError for non-gated
# activations on any platform that is not CUDA/XPU/ROCm. HPU now serves them via
# the unfused expert path in HPUUnquantizedFusedMoEMethod (see
# _unfused_no_mul_moe), so relax that guard: run the original __post_init__ and
# swallow only that specific error. The guard is the final statement of
# __post_init__, so all other configuration has already completed by the time it
# fires -- catching it leaves a fully-initialized config.
from vllm.model_executor.layers.fused_moe import config as _vllm_moe_config  # noqa: E402

_orig_moe_config_post_init = _vllm_moe_config.FusedMoEConfig.__post_init__


def _patched_moe_config_post_init(self):
    try:
        _orig_moe_config_post_init(self)
    except NotImplementedError as exc:
        if "is_act_and_mul=False" not in str(exc):
            raise


_vllm_moe_config.FusedMoEConfig.__post_init__ = _patched_moe_config_post_init
