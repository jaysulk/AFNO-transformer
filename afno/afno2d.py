# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

def dht2d(x: torch.Tensor):
    X = torch.fft.fft2(x, dim=(-2, -1))
    X = X.real - X.imag
    return X

def idht2d(X: torch.Tensor):
    n = X.shape[-2] * X.shape[-1]
    X = dht2d(X)
    x = X / n
    return x

def convolution_multiply2d(x, y):
    X = dht2d(x)
    Y = dht2d(y)
    Xflip = torch.roll(torch.flip(x, [-2, -1]), shifts=(1, 1), dims=(-2, -1))
    Yflip = torch.roll(torch.flip(y, [-2, -1]), shifts=(1, 1), dims=(-2, -1))
    Yplus = Y + Yflip
    Yminus = Y - Yflip
    Z = 0.5 * (torch.einsum("aefgh,ijk->aefij", X, Yplus) + torch.einsum("aefgh,ijk->aefij", Xflip, Yminus))
    z = idht2d(Z)
    return z

class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        x = x.reshape(B, H, W, C)
        x = dht2d(x)
        x = x.reshape(B, H, W, self.num_blocks, self.block_size)

        o1 = torch.zeros([B, H, W, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2 = torch.zeros(x.shape, device=x.device)

        total_modes = H * W
        kept_modes = int(math.sqrt(total_modes) * self.hard_thresholding_fraction)

        o1[:, :kept_modes, :kept_modes] = F.relu(
            convolution_multiply2d(x[:, :kept_modes, :kept_modes], self.w1[0]) + \
            convolution_multiply2d(x[:, :kept_modes, :kept_modes], self.w1[1]) + \
            self.b1[0]
        )

        o2[:, :kept_modes, :kept_modes] = (
            convolution_multiply2d(o1[:, :kept_modes, :kept_modes], self.w2[0]) + \
            convolution_multiply2d(o1[:, :kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias
