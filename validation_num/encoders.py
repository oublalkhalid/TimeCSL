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

from torch import nn
from typing import List, Union
from typing_extensions import Literal


__all__ = ["get_mlp"]


def get_mlp(
    n_in: int,
    n_out: int,
    layers: List[int],
    layer_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
    output_normalization: Union[None, Literal["bn"], Literal["gn"]] = None,
    output_normalization_kwargs=None,
    act_inf_param=0.02,
    linear=False,
):
    """
    Creates an MLP.

    Args:
        n_in: Dimensionality of the input data
        n_out: Dimensionality of the output data
        layers: Number of neurons for each hidden layer
        layer_normalization: Normalization for each hidden layer.
            Possible values: bn (batch norm), gn (group norm), None
        output_normalization: Normalization applied to output of network.
        output_normalization_kwargs: Arguments passed to the output normalization, e.g., the radius for the sphere.
    """
    modules: List[nn.Module] = []

    def add_module(
        n_layer_in: int,
        n_layer_out: int,
        last_layer: bool = False,
        mid_layer: bool = False,
    ):
        modules.append(nn.Linear(n_layer_in, n_layer_out))
        # perform normalization & activation not in last layer
        if not last_layer:
            if layer_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))
            elif layer_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))
            if not linear:
                modules.append(nn.LeakyReLU(negative_slope=act_inf_param))
        else:
            if output_normalization == "bn":
                modules.append(nn.BatchNorm1d(n_layer_out))

            elif output_normalization == "gn":
                modules.append(nn.GroupNorm(1, n_layer_out))

        return n_layer_out

    if len(layers) > 0:
        n_out_last_layer = n_in
    else:
        assert n_in == n_out, "Network with no layers must have matching n_in and n_out"
        modules.append(layers.Lambda(lambda x: x))

    layers.append(n_out)

    for i, l in enumerate(layers):
        n_out_last_layer = add_module(n_out_last_layer, l, i == len(layers) - 1)

    if output_normalization_kwargs is None:
        output_normalization_kwargs = {}
    return nn.Sequential(*modules)
