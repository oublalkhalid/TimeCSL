import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce

# helpers
def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

# TCN Layer (Causal Convolution with Dilations)
class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,  # Ensure causal convolution
        )
        self.activation = nn.ReLU()  # You can change this to LeakyReLU if needed

    def forward(self, x):
        return self.activation(self.conv(x))

# Affine Layer
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

# Pre-Affine Post-Layer Scale (as in the original ResMLP model)
class PreAffinePostLayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x

# ResTimeCSL Model (with TCN layers)
def ResTimeCSL(*, features_in, features_out, window_time_input, window_time_output, dim, depth, expansion_factor=4, activation="ReLU"):
    """
    ResTimeCSL Model with Temporal Convolutions (TCN layers) and specified activation functions (ReLU or LeakyReLU).
    
    :param features_in: Input feature size
    :param features_out: Output feature size
    :param window_time_input: Input time window size
    :param window_time_output: Output time window size
    :param dim: Hidden dimension for model layers
    :param depth: Depth of the model (number of blocks)
    :param expansion_factor: Expansion factor for feed-forward layers
    :param activation: Activation function to use ('ReLU' or 'LeakyReLU')
    """
    
    # Define the number of patches in the input time series
    num_patches = window_time_input
    
    # Select the activation function
    if activation == "LeakyReLU":
        activation_fn = nn.LeakyReLU(negative_slope=0.2)  # LeakyReLU with default slope
    else:
        activation_fn = nn.ReLU()  # Default to ReLU
    
    # Wrapper function for each residual block
    wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)

    # Create model layers
    return nn.Sequential(
        # Initial feature reshaping for time series data (N, features_in, window_time_input)
        Rearrange('b c t -> b t c', c=features_in),  # Rearrange input (N, features_in, window_time_input)
        
        # First temporal convolution layer (Initial transformation of features)
        nn.Conv1d(features_in, dim, kernel_size=1),  # Convert features_in to the desired hidden dimension

        # Residual blocks with TCN layers
        *[nn.Sequential(
            wrapper(i, TCNLayer(in_channels=dim, out_channels=dim, kernel_size=3, dilation=2**i)),  # TCN layer with increasing dilation
            wrapper(i, nn.Sequential(
                nn.Conv1d(dim, dim * expansion_factor, 1),  # Temporal convolutions as expansions
                activation_fn,  # Use the selected activation function
                nn.Conv1d(dim * expansion_factor, dim, 1)  # Return to original hidden dimension
            ))
        ) for i in range(depth)],

        # Final affine transformation
        Affine(dim),
        
        # Temporal average pooling
        Reduce('b t c -> b c', 'mean'),
        
        # Final output layer (convolution to adjust output dimension)
        nn.Conv1d(dim, features_out, kernel_size=1),  # Convert hidden dimension to output dimension
        
        # Output reshaping
        Rearrange('b c t -> b t c', c=features_out)  # Reshape to (N, features_out, window_time_output)
    )

# from res_time_csl import ResTimeCSL
if __name__ == "__main__":
    # Define the model with example parameters
    model = ResTimeCSL(
        features_in=64,
        features_out=128,
        window_time_input=100,
        window_time_output=100,
        dim=256,
        depth=4,
        expansion_factor=4,
        activation="ReLU"  # Can also use "LeakyReLU"
    )

    # Print the model architecture
    print("Model Architecture:")
    print(model)

    # Create dummy input tensor with batch size N=10, features_in=64, window_time_input=100
    dummy_input = torch.randn(10, 64, 100)  # Example: Batch size 10, 64 features, 100 time steps

    # Run a forward pass
    output = model(dummy_input)

    # Print the output shape to verify the model's functionality
    print("\nOutput Shape:", output.shape)
