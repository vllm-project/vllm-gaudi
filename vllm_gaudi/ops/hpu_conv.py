import torch
import torch.nn.functional as F
from vllm.model_executor.layers.conv import Conv2dLayer


@Conv2dLayer.register_oot
class HPUConv2dLayer(Conv2dLayer):

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        B, C, H, W = x.shape
        K1, K2 = self.kernel_size
        H, W = H // K1, W // K2

        # TODO: HPU doesn't support unfold, implement with view,reshape.
        #x = x.unfold(2, K1, K1).unfold(3, K2, K2)
        #x = x.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.input_size)
        x = x.view(B, C, H, K1, W, K2)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, self.input_size)  # [B*H*W, C*K1*K2]

        x = F.linear(
            x,
            self.weight.view(self.out_channels, self.input_size),
            self.bias,
        )
        x = x.view(B, H, W, self.out_channels).permute(0, 3, 1, 2)
        return x

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """Expected input shape: (batch_size, in_channels, height, width)"""
        assert x.dim() == 4
        if self.enable_linear:
            return self._forward_mulmat(x)
        else:
            return self._forward_conv(x)
