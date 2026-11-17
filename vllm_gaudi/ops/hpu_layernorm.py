from typing import Optional, Union
import torch
from vllm.model_executor.layers.layernorm import \
    RMSNorm, GemmaRMSNorm


@RMSNorm.register_oot
class HPURMSNorm(RMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_gaudi.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        if residual is not None:
            orig_shape = x.shape
            residual = residual + x.reshape(residual.shape)
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, self.weight, self.variance_epsilon)
            return x.reshape(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, self.weight, self.variance_epsilon)
        return x


@GemmaRMSNorm.register_oot
class HPUGemmaRMSNorm(GemmaRMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_gaudi.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        # Match GemmaRMSNorm.forward_native: add the offset in FP32.
        # BF16 addition rounds trained weights and can be folded differently
        # by the HPU compiler when this norm feeds a projection.
        gemma_weight = self.weight.float() + 1.0
        orig_dtype = x.dtype
        if residual is not None:
            orig_shape = x.shape
            # Normalize the FP32 sum before rounding the returned residual,
            # as in the upstream fused_add_rms_norm implementation.
            residual_float = residual.float() + x.reshape(residual.shape).float()
            x = HPUFusedRMSNorm.apply(residual_float, gemma_weight, self.variance_epsilon)
            return x.to(orig_dtype).reshape(orig_shape), residual_float.to(residual.dtype)

        x = HPUFusedRMSNorm.apply(x.float(), gemma_weight, self.variance_epsilon)
        return x.to(orig_dtype)
