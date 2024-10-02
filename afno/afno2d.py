import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

def dht2d(x: torch.Tensor) -> torch.Tensor:
    N = x.size(-1)
    
    # Use float dtype for creating the Hartley kernel
    n = torch.arange(N, device=x.device, dtype=torch.float32)
    k = n.view(-1, 1)
    
    # Calculate the Hartley kernel (cas function)
    cas = torch.cos(2 * torch.pi * k * n / N) + torch.sin(2 * torch.pi * k * n / N)
    
    # If x is complex, separate real and imaginary parts and process them separately
    if x.is_complex():
        real_part = torch.matmul(x.real, cas)
        imag_part = torch.matmul(x.imag, cas)
        X = real_part - imag_part
    else:
        # Perform the matrix multiplication between input and the Hartley kernel
        X = torch.matmul(x, cas)
    
    return X

def idht2d(x):
    # Assume that dht2d is already defined for NumPy
    # Get the dimensions of X
    dims = x.size()
    n = torch.prod(torch.tensor(dims)).item()
    dht = dht2d(x)
    H = dht / n
    return H

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

    def forward(self, x, spatial_size=None):
        bias = x
    
        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape
    
        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size
    
        # Reshape x to (B, H, W, C)
        x = x.reshape(B, H, W, C)
    
        # Replace FFT with DHT (assume x is already in DHT domain)
        X_H_k = dht2d(x)  # DHT of x
        X_H_neg_k = torch.roll(torch.flip(x, dims=[1, 2]), shifts=(1, 1), dims=[1, 2])
    
        block_size = self.block_size
        hidden_size_factor = self.hidden_size_factor
    
        # Ensure o1 and o2 dimensions match the expected sizes
        o1_H_k = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_H_neg_k = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
    
        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)
    
        # Reshape and align the dimensions of X_H_k and X_H_neg_k for broadcasting
        X_H_k = X_H_k.reshape(B, H, W, self.num_blocks, block_size)
        X_H_neg_k = X_H_neg_k.reshape(B, H, W, self.num_blocks, block_size)
    
        o1_H_k[:, :, :kept_modes] = F.relu(
            0.5 * (
                torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[0]) -
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[0])
            ) + self.b1[0]
        )
    
        o1_H_neg_k[:, :, :kept_modes] = F.relu(
            0.5 * (
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[0]) -
                torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[1]) +
                torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[0])
            ) + self.b1[1]
        )
    
        # Perform second multiplication similar to the first
        o2_H_k = torch.zeros(X_H_k.shape, device=x.device)
        o2_H_neg_k = torch.zeros(X_H_k.shape, device=x.device)
    
        o2_H_k[:, :, :kept_modes] = (
            0.5 * (
                torch.einsum('...bi,bio->...bo', o1_H_k[:, :, :kept_modes], self.w2[0]) -
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, :, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_k[:, :, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, :, :kept_modes], self.w2[0])
            ) + self.b2[0]
        )
    
        o2_H_neg_k[:, :, :kept_modes] = (
            0.5 * (
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, :, :kept_modes], self.w2[0]) -
                torch.einsum('...bi,bio->...bo', o2_H_k[:, :, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o1_H_neg_k[:, :, :kept_modes], self.w2[1]) +
                torch.einsum('...bi,bio->...bo', o2_H_k[:, :, :kept_modes], self.w2[0])
            ) + self.b2[1]
        )
    
        # Combine positive and negative frequency components back
        x = o2_H_k + o2_H_neg_k
        x = F.softshrink(x, lambd=self.sparsity_threshold)
    
        # Transform back to spatial domain (assuming DHT-based iDHT here)
        x = idht2d(x)
        x = x.reshape(B, H, W, C)
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x.real + bias.real
