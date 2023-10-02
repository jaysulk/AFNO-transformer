# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

def dht(x: torch.Tensor):
    X = torch.fft.fft(x)
    X = X.real - X.imag
    return X

def idht(X: torch.Tensor):
    n = X.numel()
    X = dht(X)
    x = X / n
    return x

def convolution_multiply(x, y):
    X = dht(x)
    Y = dht(y)
    Xflip = torch.roll(torch.flip(x, [0]), 1, dims=0)
    Yflip = torch.roll(torch.flip(y, [0]), 1, dims=0)
    Yplus = Y + Yflip
    Yminus = Y - Yflip
    Z = 0.5 * (torch.einsum("..bi,bio->...bo", X, Yplus) + torch.einsum("..bi,bio->...bo", Xflip, Yminus))
    z = idht(Z)
    return z

class AFNO1D(nn.Module):
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

        x = dht(x)
        x = x.reshape(B, N, self.num_blocks, self.block_size)

        o1 = torch.zeros([B, N, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2 = torch.zeros(x.shape, device=x.device)

        total_modes = N
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1[:, :kept_modes] = F.relu(
            convolution_multiply(x[:, :kept_modes], self.w1[0]) + \
            convolution_multiply(x[:, :kept_modes], self.w1[1]) + \
            self.b1[0]
        )

        o2[:, :kept_modes] = (
            convolution_multiply(o1[:, :kept_modes], self.w2[0]) + \
            convolution_multiply(o1[:, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = idht(x)
        x = x.type(dtype)
        return x + bias
