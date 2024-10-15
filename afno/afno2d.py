import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import math

def get_transform_dims(x: torch.Tensor):
    """
    Determine the transform dimensions based on the input tensor's dimensionality.
    
    - 3D tensor: [batch, channel, length] -> dim=[2]
    - 4D tensor: [batch, channel, height, width] -> dim=[2, 3]
    - 5D tensor: [batch, channel, depth, height, width] -> dim=[2, 3, 4]
    
    Raises:
        ValueError: If tensor dimensionality is not 3, 4, or 5.
    
    Returns:
        List[int]: List of dimensions to perform the transform on.
    """
    if x.dim() == 3:
        return [2]
    elif x.dim() == 4:
        return [2, 3]
    elif x.dim() == 5:
        return [2, 3, 4]
    else:
        raise ValueError(f"Unsupported tensor dimension: {x.dim()}. Supported dimensions are 3, 4, or 5.")

def compl_mul(x1: torch.Tensor, x2: torch.Tensor, num_transform_dims: int) -> torch.Tensor:
    """
    Generalized convolution using torch.einsum for 1D, 2D, and 3D cases.
    
    Args:
        x1 (torch.Tensor): Input tensor with shape [batch, in_channels, ...]
        x2 (torch.Tensor): Kernel tensor with shape [in_channels, out_channels, ...]
        num_transform_dims (int): Number of dimensions to transform (1, 2, or 3)
    
    Returns:
        torch.Tensor: Convolved tensor with shape [batch, out_channels, ...]
    """
    # Define letters for transform dimensions
    letters = ['x', 'y', 'z']
    if num_transform_dims > len(letters):
        raise ValueError(f"Number of transform dimensions {num_transform_dims} exceeds supported letters {letters}.")
    
    transform_letters = ''.join(letters[:num_transform_dims])
    
    # Construct einsum equation
    # x1: 'b i' + transform_letters
    # x2: 'i o' + transform_letters
    # output: 'b o' + transform_letters
    x1_subscript = 'bi' + transform_letters
    x2_subscript = 'io' + transform_letters
    output_subscript = 'bo' + transform_letters
    equation = f'{x1_subscript},{x2_subscript}->{output_subscript}'
    
    return torch.einsum(equation, x1, x2)

def flip_periodic(x: torch.Tensor) -> torch.Tensor:
    """
    Perform a periodic flip of the tensor along the transform dimensions.
    
    For each set of transform dimensions, concatenate the first element with the flipped remaining elements.
    
    Args:
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Periodically flipped tensor.
    """
    transform_dims = get_transform_dims(x)
    num_transform_dims = len(transform_dims)
    
    # Ensure transform dimensions are the last N dimensions
    expected_transform_dims = list(range(x.dim() - num_transform_dims, x.dim()))
    if transform_dims != expected_transform_dims:
        raise NotImplementedError("Transform dimensions must be the last N dimensions of the tensor.")
    
    # Compute the total number of elements along transform dims
    # Flatten the transform dims into one dimension
    batch_shape = x.shape[:-num_transform_dims]
    flattened_dim = 1
    for dim in transform_dims:
        flattened_dim *= x.shape[dim]
    x_flat = x.view(*batch_shape, flattened_dim)
    
    # Split the first element and the remaining
    first = x_flat[..., :1]  # Shape: (..., 1)
    remaining = x_flat[..., 1:]  # Shape: (..., flattened_dim - 1)
    
    # Flip the remaining elements
    remaining_flipped = torch.flip(remaining, dims=[-1])
    
    # Concatenate first and flipped remaining
    Z_flat = torch.cat([first, remaining_flipped], dim=-1)
    
    # Reshape back to original shape
    Z = Z_flat.view_as(x)
    return Z

def dht(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Discrete Hartley Transform (DHT) of the input tensor.
    
    Args:
        x (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: DHT of the input tensor.
    """
    transform_dims = get_transform_dims(x)
    X = torch.fft.fftn(x, dim=transform_dims)
    X = X.real - X.imag
    return X

def idht(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the Inverse Discrete Hartley Transform (IDHT) of the input tensor.
    
    Since the DHT is involutory, IDHT(x) = (1/n) * DHT(DHT(x))
    
    Args:
        X (torch.Tensor): Input tensor in the DHT domain.
        
    Returns:
        torch.Tensor: Inverse DHT of the input tensor.
    """
    transform_dims = get_transform_dims(X)
    # Compute the product of sizes along transform dims
    n = 1
    for dim in transform_dims:
        n *= X.shape[dim]
    x = dht(X)
    x = x / n
    return x

def dht_conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the DHT of the convolution of two tensors using the convolution theorem and torch.einsum.
    
    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, ...]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, ...]
        
    Returns:
        torch.Tensor: DHT of the convolution of x and y.
        
    Raises:
        AssertionError: If x and y do not have the same shape except for the out_channels dimension.
    """
    # Ensure x and y have compatible shapes
    # x: [batch, in_channels, ...]
    # y: [in_channels, out_channels, ...]
    assert x.dim() == y.dim(), "x and y must have the same number of dimensions."
    assert y.shape[0] == x.shape[1], "y's in_channels must match x's in_channels."
    num_transform_dims = x.dim() - 2  # Exclude batch and channel dimensions
    
    # Compute DHTs
    X = dht(x)  # [batch, in_channels, ...]
    Y = dht(y)  # [in_channels, out_channels, ...]
    
    # Compute flipped versions
    Xflip = flip_periodic(X)
    Yflip = flip_periodic(Y)
    
    # Compute even and odd components
    Yeven = 0.5 * (Y + Yflip)
    Yodd  = 0.5 * (Y - Yflip)
    
    # Perform convolution using the generalized compl_mul with torch.einsum
    # Z = X * Yeven + Xflip * Yodd
    # We'll use compl_mul for the tensor contractions
    
    # First term: compl_mul(X, Yeven)
    term1 = compl_mul(X, Yeven, num_transform_dims)
    
    # Second term: compl_mul(Xflip, Yodd)
    term2 = compl_mul(Xflip, Yodd, num_transform_dims)
    
    # Combine terms
    Z = term1 + term2  # [batch, out_channels, ...]
    
    return Z

def conv(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the convolution of two tensors using the DHT.
    
    Args:
        x (torch.Tensor): First input tensor with shape [batch, in_channels, ...]
        y (torch.Tensor): Second input tensor with shape [in_channels, out_channels, ...]
        
    Returns:
        torch.Tensor: Convolution of x and y with shape [batch, out_channels, ...]
    """
    Z = dht_conv(x, y)
    z = idht(Z)
    return z

def dht2d(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D Discrete Hartley Transform (DHT) of the input tensor.
    
    Args:
        x (torch.Tensor): Input tensor with shape [B, C, H, W]
        
    Returns:
        torch.Tensor: 2D DHT of the input tensor with shape [B, C, H, W]
    """
    # Apply DHT along height and width dimensions
    transform_dims = [2, 3]
    X = torch.fft.fftn(x, dim=transform_dims)
    X = X.real - X.imag
    return X

def idht2d(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the 2D Inverse Discrete Hartley Transform (IDHT) of the input tensor.
    
    Args:
        X (torch.Tensor): Input tensor in the DHT domain with shape [B, C, H, W]
        
    Returns:
        torch.Tensor: Inverse 2D DHT of the input tensor with shape [B, C, H, W]
    """
    # Compute the inverse DHT by applying DHT again and scaling
    transform_dims = [2, 3]
    n = X.shape[2] * X.shape[3]  # H * W
    x = torch.fft.fftn(X, dim=transform_dims)
    x = x.real - x.imag
    x = x / n
    return x

class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        """
        Initialize the 2D Attention-Free Neural Operator (AFNO) module using DHT.
        
        Args:
            hidden_size (int): Total hidden size (must be divisible by num_blocks).
            num_blocks (int): Number of blocks to divide the hidden size into.
            sparsity_threshold (float): Threshold for sparsity (unused in current implementation).
            hard_thresholding_fraction (float): Fraction of modes to keep after thresholding.
            hidden_size_factor (int): Factor to increase hidden size in intermediate layers.
        """
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        # Initialize weights and biases for two linear layers with DHT-based convolution
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, spatial_size=None):
        """
        Forward pass of the AFNO2D module.
        
        Args:
            x (torch.Tensor): Input tensor with shape [batch, N, C], where N = H * W.
            spatial_size (tuple, optional): Spatial dimensions (H, W). If None, assumes square spatial dimensions.
        
        Returns:
            torch.Tensor: Output tensor with shape [batch, N, C].
        """
        bias = x  # Save the input for residual connection

        dtype = x.dtype
        x = x.float()
        B, N, C = x.shape

        if spatial_size is None:
            H = W = int(math.sqrt(N))
        else:
            H, W = spatial_size

        assert H * W == N, "spatial_size does not match the number of spatial points in x."

        # Reshape x to [B, C, H, W]
        x = x.reshape(B, C, H, W)

        # Compute DHT of x
        X_H_k = dht2d(x)  # [B, C, H, W]
        # Compute periodic flip for negative frequencies
        X_H_neg_k = flip_periodic(X_H_k)  # [B, C, H, W]

        block_size = self.block_size
        hidden_size_factor = self.hidden_size_factor

        # Initialize output tensors for first linear layer
        o1_H_k = torch.zeros(B, self.num_blocks, H, W, block_size * hidden_size_factor, device=x.device)
        o1_H_neg_k = torch.zeros(B, self.num_blocks, H, W, block_size * hidden_size_factor, device=x.device)

        total_modes = H * W
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # Reshape X_H_k and X_H_neg_k to [B, C, H, W, num_blocks, block_size]
        X_H_k = X_H_k.view(B, C, H, W, self.num_blocks, block_size)
        X_H_neg_k = X_H_neg_k.view(B, C, H, W, self.num_blocks, block_size)

        # Perform first convolution (linear transformation in frequency domain)
        for b in range(self.num_blocks):
            # Select the b-th block
            X_block = X_H_k[:, :, :, :, b, :]  # [B, C, H, W, block_size]
            X_neg_block = X_H_neg_k[:, :, :, :, b, :]  # [B, C, H, W, block_size]

            # Reshape for convolution: merge batch, spatial dimensions
            X_block = X_block.view(B, C, H, W, block_size)
            X_neg_block = X_neg_block.view(B, C, H, W, block_size)

            # Perform convolution using dht_conv
            # Reshape to [B, C, H, W, block_size] -> [B, C, H, W * block_size]
            X_block = X_block.view(B, C, H, W * block_size)
            X_neg_block = X_neg_block.view(B, C, H, W * block_size)

            # Apply convolution
            o1_H_k[:, b, :, :, :] = F.relu(torch.einsum('bchw,bohw->bcoh', X_block, self.w1[0, b]))
            o1_H_neg_k[:, b, :, :, :] = F.relu(torch.einsum('bchw,bohw->bcoh', X_neg_block, self.w1[1, b]))

        # Combine positive and negative frequency components
        o1_H = o1_H_k + o1_H_neg_k  # [B, num_blocks, H, W, block_size * hidden_size_factor]

        # Reshape for second convolution
        o1_H = o1_H.view(B, self.num_blocks, H, W, block_size * hidden_size_factor)

        # Initialize output tensors for second linear layer
        o2_H_k = torch.zeros(B, self.num_blocks, H, W, block_size, device=x.device)
        o2_H_neg_k = torch.zeros(B, self.num_blocks, H, W, block_size, device=x.device)

        # Perform second convolution (linear transformation in frequency domain)
        for b in range(self.num_blocks):
            # Select the b-th block
            o1_block = o1_H[:, b, :, :, :]  # [B, H, W, block_size * hidden_size_factor]
            o1_neg_block = o1_H_neg_k[:, b, :, :, :]  # [B, H, W, block_size * hidden_size_factor]

            # Reshape for convolution
            o1_block = o1_block.view(B, H, W * block_size * hidden_size_factor)
            o1_neg_block = o1_neg_block.view(B, H, W * block_size * hidden_size_factor)

            # Apply convolution
            o2_H_k[:, b, :, :, :] = torch.einsum('bhw,bohw->bcoh', o1_block, self.w2[0, b])
            o2_H_neg_k[:, b, :, :, :] = torch.einsum('bhw,bohw->bcoh', o1_neg_block, self.w2[1, b])

        # Combine positive and negative frequency components
        o2_H = o2_H_k + o2_H_neg_k  # [B, num_blocks, H, W, block_size]

        # Reshape back to [B, C, H, W]
        o2_H = o2_H.view(B, self.hidden_size, H, W)

        # Compute inverse DHT to get back to spatial domain
        x = idht2d(o2_H)  # [B, C, H, W]

        # Reshape back to [B, N, C]
        x = x.view(B, self.hidden_size, H * W)
        x = x.permute(0, 2, 1)  # [B, N, C]

        # Add residual connection
        x = x + bias.type_as(x)

        return x.real  # Return the real part
