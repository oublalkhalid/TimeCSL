# Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing
# Copyright 2024, ICLR 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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
def ResTimeCSL(
    *,
    features_in,
    features_out,
    window_time_input,
    window_time_output,
    dim,
    depth,
    expansion_factor=4,
    activation="ReLU"
):
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

    # Modify the initial feature reshaping and convolution layer to ensure dimensionality match
    return nn.Sequential(
        # Initial feature reshaping for time series data (N, features_in, window_time_input)
        Rearrange(
            "b c t -> b c t", c=features_in
        ),  # Keep the shape (N, features_in, time_steps)
        # First temporal convolution layer (Initial transformation of features)
        nn.Conv1d(features_in, dim, kernel_size=1),  # Now features_in -> dim (256)
        # Residual blocks with TCN layers
        *[
            nn.Sequential(
                wrapper(
                    i,
                    TCNLayer(
                        in_channels=dim,
                        out_channels=dim,
                        kernel_size=3,
                        dilation=2**i,
                    ),
                ),
                wrapper(
                    i,
                    nn.Sequential(
                        nn.Conv1d(dim, dim * expansion_factor, 1),
                        activation_fn,
                        nn.Conv1d(dim * expansion_factor, dim, 1),
                    ),
                ),
            )
            for i in range(depth)
        ],
        # Final affine transformation
        Affine(dim),
        # Temporal average pooling
        Reduce("b t c -> b c", "mean"),
        # Final output layer
        nn.Conv1d(dim, features_out, kernel_size=1),
        # Output reshaping
        Rearrange("b c t -> b t c", c=features_out)
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
        activation="ReLU",  # Can also use "LeakyReLU"
    )

    # Print the model architecture
    print("Model Architecture:")
    print(model)

    # Create dummy input tensor with batch size N=10, features_in=64, window_time_input=100
    dummy_input = torch.randn(
        10, 64, 100
    )  # Example: Batch size 10, 64 features, 100 time steps

    # Run a forward pass
    output = model(dummy_input)

    # Print the output shape to verify the model's functionality
    print("\nOutput Shape:", output.shape)


#     input = module(input)
#   File "/tsi/data_education/Ladjal/koublal/main/env_miniconda/envs/torch-nilm/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/tsi/data_education/Ladjal/koublal/main/env_miniconda/envs/torch-nilm/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/tsi/data_education/Ladjal/koublal/main/env_miniconda/envs/torch-nilm/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 308, in forward
#     return self._conv_forward(input, self.weight, self.bias)
#   File "/tsi/data_education/Ladjal/koublal/main/env_miniconda/envs/torch-nilm/lib/python3.9/site-packages/torch/nn/modules/conv.py", line 304, in _conv_forward
#     return F.conv1d(input, weight, bias, self.stride,
# RuntimeError: Given groups=1, weight of size [256, 64, 1], expected input[10, 100, 64] to have 64 channels, but got 100 channels instead
