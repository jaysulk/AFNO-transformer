import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

def hartley_kernel(n):
    """ Returns the Hartley kernel for a given size n using PyTorch """
    theta = 2 * torch.pi / n
    return torch.cos(torch.arange(n) * theta) + torch.sin(torch.arange(n) * theta)

def radix2_fht_1d(x):
    """ Recursive implementation of the Radix-2 Fast Hartley Transform for 1D data using PyTorch """
    N = x.shape[0]

    # Base case
    if N <= 1:
        return x

    # Split the sequence into even and odd parts
    even = radix2_fht_1d(x[::2])
    odd = radix2_fht_1d(x[1::2])

    # Combine
    combined = torch.zeros(N)
    H = hartley_kernel(N)
    for k in range(N // 2):
        combined[k] = even[k] + H[k] * odd[k]
        combined[k + N // 2] = even[k] - H[k] * odd[k]

    return combined

def apply_fht_to_rows(x):
    """ Apply the FHT to each row of a 2D tensor """
    rows, cols = x.shape
    for r in range(rows):
        x[r, :] = radix2_fht_1d(x[r, :])
    return x

def apply_fht_to_cols(x):
    """ Apply the FHT to each column of a 2D tensor """
    rows, cols = x.shape
    for c in range(cols):
        x[:, c] = radix2_fht_1d(x[:, c])
    return x

def dht2d(x):
    """ Compute the 2D Hartley Transform for each channel of each image """
    if x.dim() != 4:
        raise ValueError("Input must be a 4D tensor representing a batch of images")

    batch_size, channels, rows, cols = x.shape

    # Applying 2D FHT to each channel of each image
    for i in range(batch_size):
        for c in range(channels):
            x[i, c, :, :] = apply_fht_to_rows(x[i, c, :, :])
            x[i, c, :, :] = apply_fht_to_cols(x[i, c, :, :])

    return x

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

        #if spatial_size == None:
        H = W = int(math.sqrt(N))
        #else:
        #    H, W = spatial_size

        x = x.reshape(B, H, W, C)
        x = dht2d(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1 = torch.zeros([B, H, W, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2 = torch.zeros(x.shape, device=x.device)
        #o2_real = torch.zeros(x.shape, device=x.device)
        #o2_imag = torch.zeros(x.shape, device=x.device)

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
