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
import torch.nn as nn
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .residual_timecsl import ResTimeCSL


class TimeCSL(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        input_window=100,
        output_window=100,
        series_channels=[8, 16, 32, 64],
        dim=384,
        depth=12,
        mlp_dim=384,
        latent_dim=16,
        latent_channels=8,
        covariance=None,
    ):
        super(TimeCSL, self).__init__()

        self.pi_prior = nn.Parameter(
            torch.ones(latent_channels * latent_dim) / latent_channels * latent_dim
        )
        self.mu_prior = nn.Parameter(10 * torch.randn(latent_dim, latent_channels))

        self.encoder = nn.Sequential(
            ResTimeCSL(
                in_channels=in_channels,
                output_channels=latent_dim,
                input_window=input_window,
                output_window=input_window // 2,
                series_channels=series_channels,
                dim=dim,
                depth=depth,
                mlp_dim=mlp_dim,
            ),
            ResTimeCSL(
                in_channels=latent_dim,
                output_channels=latent_dim,
                input_window=input_window // 2,
                output_window=input_window // 4,
                series_channels=series_channels,
                dim=dim,
                depth=depth,
                mlp_dim=mlp_dim,
            ),
            ResTimeCSL(
                in_channels=latent_dim,
                output_channels=latent_dim,
                input_window=input_window // 4,
                output_window=latent_channels,
                series_channels=series_channels,
                dim=dim,
                depth=depth,
                mlp_dim=mlp_dim,
            ),
        )

        self.covariance = covariance
        if covariance == "full":
            self.var_prior = nn.Parameter(
                7 * torch.randn(latent_dim, latent_channels, latent_dim)
            )
            self.var_log_det = torch.randn(latent_dim)
        else:
            self.var_prior = nn.Parameter(torch.randn(latent_channels, latent_dim) / 10)
            self.log_var_prior = nn.Parameter(torch.randn(latent_dim, latent_dim) - 3)
        # Flatten to compute latent space
        # self.to_latent = nn.Sequential(
        #     nn.Conv1d(series_channels[2], latent_channels, kernel_size=1),
        #     nn.AdaptiveAvgPool1d(latent_dim)
        # )
        # Latent space
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.mu = nn.Linear(latent_channels * latent_dim, latent_dim * latent_channels)
        self.logvar = nn.Linear(
            latent_channels * latent_dim, latent_dim * latent_channels
        )

        # Decoder
        self.decoder_input = nn.Linear(
            latent_dim * latent_channels, latent_channels * latent_dim
        )
        self.decoder = nn.Sequential(
            ResTimeCSL(
                in_channels=latent_dim,
                output_channels=latent_dim,
                input_window=latent_channels,
                output_window=latent_channels * 2,
                series_channels=series_channels,
                dim=dim,
                depth=depth,
                mlp_dim=mlp_dim,
            ),
            ResTimeCSL(
                in_channels=latent_dim,
                output_channels=out_channels,
                input_window=latent_channels * 2,
                output_window=output_window // 2,
                series_channels=series_channels,
                dim=dim,
                depth=depth,
                mlp_dim=mlp_dim,
            ),
            ResTimeCSL(
                in_channels=out_channels,
                output_channels=out_channels,
                input_window=output_window // 2,
                output_window=output_window,
                series_channels=series_channels,
                dim=dim,
                depth=depth,
                mlp_dim=mlp_dim,
            ),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        print("x", x.shape)
        # Flatten to (B, latent_dim * latent_channels)
        mu = self.mu(x.flatten(1))
        logvar = self.logvar(x.flatten(1))

        z = self.reparameterize(mu, logvar)
        print("z", z.shape)

        flag = False  # np.random.rand()
        self.epoch = 1
        if flag:  # (flag>0.7*np.power(0.85, self.epoch)) or True: #remove later
            z = self.decoder_input(z).view(
                x.size(0), self.latent_dim, self.latent_channels
            )  # Reshape to (B, latent_channels, latent_dim)
        else:
            gamma = self.compute_gamma(z, self.pi_prior)
            pred = torch.argmax(gamma, dim=1)
            if self.covariance == "full":
                transform = torch.linalg.pinv(self.var_prior[pred])

                z_new = self.mu_prior[pred] + torch.matmul(
                    transform, torch.randn(transform.shape[1])
                )
            else:
                z_new = self.reparameterize(
                    self.mu_prior[pred], self.log_var_prior[pred]
                )
            z_new = self.decoder_input(z).view(
                z_new.size(0), self.latent_dim, self.latent_channels
            )  # Reshape to (B, latent_channels, latent_dim)
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

    def compute_gamma(self, z, p_c):
        if self.covariance == "full":

            h = z.unsqueeze(1) - self.mu_prior

            h = (
                torch.matmul(self.sqrt_var_prior, h.permute(1, 2, 0))
                .permute(2, 0, 1)
                .pow(2)
            )
            h += torch.Tensor([np.log(np.pi * 2)])
            p_z_c = (
                torch.exp(
                    torch.log(p_c + 1e-9).unsqueeze(0)
                    - 0.5 * torch.sum(h, dim=2)
                    + 0.5 * self.var_log_det
                )
                + 1e-9
            )
            # print(p_z_c)
        else:

            h = (z.unsqueeze(1) - self.mu_prior).pow(2)
            h = h / self.log_var_prior.exp()
            h += self.log_var_prior
            h += torch.Tensor([np.log(np.pi * 2)])
            p_z_c = (
                torch.exp(
                    torch.log(p_c + 1e-9).unsqueeze(0) - 0.5 * torch.sum(h, dim=2)
                )
                + 1e-9
            )
        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
        return gamma
