import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

def dht2d(x: torch.Tensor):
    # Perform real FFT
    X = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

    # Calculate DHT (Discrete Hartley Transform)
    X = X.real - X.imag

    # Padding to match the original last dimension size
    # The last dimension of the output of rfft2 is (input_size // 2 + 1)
    # We need to calculate how much padding is needed to restore it to the original size
    last_dim_padding = x.shape[-1] - X.shape[-1]

    # Apply padding to the last dimension
    X_padded = F.pad(X, (0, last_dim_padding))

    return X_padded

def idht2d(X: torch.Tensor):
    dims = X.size()
    n = torch.prod(torch.tensor(dims)).item()
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
    Z1 = torch.mul(X, Yplus)
    Z2 = torch.mul(Xflip, Yminus)
    Z = 0.5 * (Z1 + Z2)
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

        #if spatial_size == None:
        H = W = int(math.sqrt(N))
        #else:
        #    H, W = spatial_size

        x = x.reshape(B, H, W, C)
        x = dht2d(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1 = torch.zeros([B, H, W, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2 = torch.zeros(x.shape, device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

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

        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = idht2d(x)
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias
