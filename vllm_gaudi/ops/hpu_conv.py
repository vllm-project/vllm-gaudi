import torch
import torch.nn.functional as F
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer
from vllm.utils.torch_utils import is_torch_equal


@Conv3dLayer.register_oot
class HPUConv3dLayer(Conv3dLayer):
    """Conv layer with Conv3d."""

    num_dim = 3

    def _forward_mulmat(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        B, C, T, H, W = x.shape
        K1, K2, K3 = self.kernel_size
        T, H, W = T // K1, H // K2, W // K3
        x = x.view(B, C, T, K1, H, K2, W, K3)
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(-1, self.input_size)
        x = F.linear(
            x,
            self.weight.view(self.out_channels, self.input_size),
            self.bias,
        )
        x = x.view(B, T, H, W, self.out_channels).permute(0, 4, 1, 2, 3)
        return x

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        x = F.conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """Expected input shape: (batch_size, in_channels, time, height, width)"""
        if self.enable_linear:
            return self._forward_mulmat(x)
        else:
            return self._forward_conv(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch2.9.0 disabled CUDNN's Conv3D, which caused a
        # significant performance regression.
        # See: https://github.com/vllm-project/vllm/issues/27406
        # and https://github.com/pytorch/pytorch/issues/166122
        # By default, we use CUDNN's convolution ops with optimization.
        if self.enable_linear and (is_torch_equal("2.9.0") or is_torch_equal("2.9.1")):
            return self._forward_mulmat(x)
        return self._forward_conv(x)


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
