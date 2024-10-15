import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

def dht2d(x: torch.Tensor, dim=None) -> torch.Tensor:
    # Compute the N-dimensional FFT of the input tensor
    result = torch.fft.fftn(x, dim=dim)
    
    # Combine real and imaginary parts to compute the DHT
    return result.real + result.imag  # Use subtraction to match DHT definition

def idht2d(x: torch.Tensor, dim=None) -> torch.Tensor:
    # Compute the DHT of the input tensor
    transformed = dht2d(x, dim=dim)
    
    # Determine normalization factor based on the specified dimensions
    if dim is None:
        # If dim is None, use the total number of elements
        normalization_factor = x.numel()
    else:
        # Ensure dim is a list of dimensions
        if isinstance(dim, int):
            dim = [dim]
        normalization_factor = 1
        for d in dim:
            normalization_factor *= x.size(d)
    
    # Return the normalized inverse DHT
    return transformed / normalization_factor

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but fewer parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 fewer FLOPs)
    """
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

        # Parameters for two linear layers
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

        # Reshape input tensor while preserving the batch size
        x = x.reshape(B, H, W, C)

        # Perform the DHT-based convolution using the circular convolution theorem
        X_H_k = x
        X_H_neg_k = torch.roll(torch.flip(x, dims=[1, 2]), shifts=(1, 1), dims=[1, 2])

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real = torch.zeros([B, H, W, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros([B, H, W, self.num_blocks, self.block_size], device=x.device)
        o2_imag = torch.zeros([B, H, W, self.num_blocks, self.block_size], device=x.device)

        # Apply DHT-based convolution theorem
        o1_real[:, :, :kept_modes] = F.relu(
            0.5 * (torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[0]) - \
                   torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[1]) + \
                   torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[1]) + \
                   torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[0]) + \
                   self.b1[0])
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            0.5 * (torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[0]) + \
                   torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[1]) - \
                   torch.einsum('...bi,bio->...bo', X_H_k[:, :, :kept_modes], self.w1[1]) - \
                   torch.einsum('...bi,bio->...bo', X_H_neg_k[:, :, :kept_modes], self.w1[0]) + \
                   self.b1[1])
        )

        o2_real[:, :, :kept_modes] = (
            0.5 * (torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
                   torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
                   torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
                   torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
                   self.b2[0])
        )

        o2_imag[:, :, :kept_modes] = (
            0.5 * (torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
                   torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) - \
                   torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) - \
                   torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) + \
                   self.b2[1])
        )

        # Reconstruct the final output, ensuring the tensor shape is preserved
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        #x = torch.view_as_real(x)  # View as real for DHT convolution output

        # Make sure to preserve the shape by reshaping back to the original
        x = x.reshape(B, H * W, C)
        x = x.type(dtype)

        # Add the residual bias back to the output
        return x.real + bias.real
