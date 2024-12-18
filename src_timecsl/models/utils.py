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

"""
This file contains the encoder and decoder architectures for the simple AE model.
Architecure closely follows the one from the https://arxiv.org/abs/1804.03599

Burgess, Christopher P., et al. 
"Understanding disentangling in $\beta $-VAE." arXiv preprint arXiv:1804.03599 (2018).
"""
import torch.nn as nn


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


def get_encoder(in_channels, out_dim):
    """
    Encoder f^{-1}: X -> Z; X - input image, Z - latent space.
    """
    modules_list = [
        nn.Conv2d(in_channels, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ELU(),
        View((-1, 32 * 4 * 4)),
        nn.Linear(32 * 4 * 4, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, out_dim),
    ]

    encoder = nn.Sequential(*modules_list)
    return encoder


def get_decoder(in_dim, out_channels):
    """
    Decoder f: Z -> X; X - input image, Z - latent space.
    """
    module_list = [
        nn.Linear(in_dim, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, 32 * 4 * 4),
        nn.ELU(),
        View((-1, 32, 4, 4)),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ELU(),
        nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
    ]

    decoder = nn.Sequential(*module_list)
    return decoder
