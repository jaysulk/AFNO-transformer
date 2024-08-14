import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

def reverse(x: torch.Tensor) -> torch.Tensor:
    N = x.size(-1)
    return torch.roll(x.flip(-1), shifts=N+1, dims=-1)

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


def idht2d(X: torch.Tensor) -> torch.Tensor:
    N = X.size(-1)
    n = torch.prod(torch.tensor(X.size())).item()
    
    # Perform the forward DHT on the input
    x_reconstructed = dht2d(X)
    
    # Scale the result by the number of elements
    x_reconstructed /= n
    
    return x_reconstructed

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
        x = torch.complex(x, torch.zeros_like(x))
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1 = torch.zeros([B, H, W, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2 = torch.zeros(x.shape, device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = H * W
        kept_modes = int(math.sqrt(total_modes) * self.hard_thresholding_fraction)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) -\
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = idht2d(x)
        x = x.reshape(B, N, C)
        x = x.type(dtype)
        return x + bias
