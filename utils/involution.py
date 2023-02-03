import types
import torch.nn as nn


# implementation of algorithm 1 from v2 (11 Apr 2021) of https://arxiv.org/abs/2103.06255
class Involution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, r=1):
        super().__init__()
        
        if in_channels != out_channels:
            raise RuntimeError(f"in_channels ({in_channels}) must be equal to out_channels ({out_channels})")
        if kernel_size % 2 != 1:
            raise RuntimeError(f"kernel_size ({kernel_size}) must be odd")
        if in_channels % groups != 0:
            raise RuntimeError(f"cannot group {in_channels} channels into {groups} groups")
        if in_channels % r != 0:
            raise RuntimeError(f"cannot reduce {in_channels} channels by ratio {r}")
            
        self.inv_params = types.SimpleNamespace(
            C = in_channels,
            CG = in_channels // groups,
            M = dilation * (kernel_size - 1),
            M2 = dilation * ((kernel_size-1) // 2),
            K2 = kernel_size**2,
            S = stride,
            G = groups
        )
        
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.reduce = nn.Conv2d(in_channels, in_channels//r, 1, bias=bias)
        self.get_kernels = nn.Conv2d(in_channels//r, groups * kernel_size**2, 1, bias=bias)
        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
    
    def forward(self, x):
        H = x.shape[-2] - self.inv_params.M
        W = x.shape[-1] - self.inv_params.M
        
        if (H%self.inv_params.S!=0) or (W%self.inv_params.S!=0):
            raise RuntimeError(f"input (size={H}x{W}) not divisible by stride ({self.inv_params.S})")
            
        H //= self.inv_params.S
        W //= self.inv_params.S
        
        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.view(-1, self.inv_params.G, self.inv_params.CG, self.inv_params.K2, H, W)
        
        x = self.pool(x[
            :,
            :,
            self.inv_params.M2:-self.inv_params.M2,
            self.inv_params.M2:-self.inv_params.M2
        ])
        kernels = self.get_kernels(self.reduce(x))
        kernels = kernels.view(-1, self.inv_params.G, self.inv_params.K2, H, W).unsqueeze(2)
        
        x = kernels * x_unfolded
        return x.sum(dim=3).view(-1, self.inv_params.C, H, W)
