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

import os.path
from contextlib import nullcontext

import torch
import torch.utils.data

import src.datasets.wrappers
import src.metrics as metrics
import src.models
import src.utils.training_utils as training_utils
from .models import base_models, slot_attention


def one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    mode,
    epoch,
    consistency_ignite_epoch=0,
    use_consistency_loss=True,
    freq=1,
    unsupervised_mode=True,
    **kwargs,
):
    if mode == "train":
        model.train()
    elif mode in ["test_ID", "test_OOD", "test_RDM"]:
        model.eval()
    else:
        raise ValueError("mode must be either train or test")

    accum_total_loss = 0
    accum_model_loss = 0
    accum_reconstruction_loss = 0
    accum_reconstruction_r2 = 0
    accum_slots_loss = 0
    accum_r2_score = 0
    accum_ari_score = 0
    per_latent_r2_score = 0
    accum_consistency_loss = 0
    accum_consistency_encoder_loss = 0
    accum_consistency_decoder_loss = 0
    for batch_idx, (time_series, true_latents) in enumerate(dataloader):
        total_loss = torch.tensor(0.0, device=device)
        accum_adjustment = len(time_series) / len(dataloader.dataset)
        time_series = time_series.to(device)
        true_figures = time_series[:, :-1, ...]
        time_series = time_series[:, -1, ...].squeeze(1)
        true_latents = true_latents.to(device)

        if mode == "train":
            optimizer.zero_grad()

        output_dict = model(
            time_series,
            use_consistency_loss=use_consistency_loss
            * (epoch >= consistency_ignite_epoch),
        )

        if "loss" in output_dict:
            model_loss = output_dict["loss"]
            accum_model_loss += model_loss.item() * accum_adjustment

        reconstruction_loss = metrics.reconstruction_loss(
            time_series, output_dict["reconstructed_time_series"]
        )
        accum_reconstruction_loss += reconstruction_loss.item() * accum_adjustment

        reconstruction_r2 = metrics.time_series_r2_score(
            time_series.clone(), output_dict["reconstructed_time_series"]
        )
        accum_reconstruction_r2 += reconstruction_r2.item() * accum_adjustment

        if (
            model.model_name not in ["SlotMLPAdditive", "SlotMLPMonolithic"]
            and epoch % freq == 0
        ):
            true_masks = training_utils.get_masks(time_series, true_figures)
            ari_score = metrics.ari(
                true_masks,
                output_dict["reconstructed_masks"].detach().permute(1, 0, 2, 3, 4),
            )
            true_masks = true_masks.detach().permute(1, 0, 2, 3, 4)
            accum_ari_score += ari_score.item() * accum_adjustment

        total_loss += reconstruction_loss

        if not unsupervised_mode:
            slots_loss, inds = metrics.hungarian_slots_loss(
                true_latents,
                output_dict["predicted_latents"],
                device,
            )
            accum_slots_loss += slots_loss.item() * accum_adjustment

            avg_r2, raw_r2 = metrics.r2_score(
                true_latents, output_dict["predicted_latents"], inds
            )
            accum_r2_score += avg_r2 * accum_adjustment
            per_latent_r2_score += raw_r2 * accum_adjustment

            total_loss += slots_loss

        if model.model_name != "SlotMLPMonolithic":
            with nullcontext() if use_consistency_loss else torch.no_grad():
                consistency_encoder_loss, _ = metrics.hungarian_slots_loss(
                    output_dict["sampled_latents"],
                    output_dict["predicted_sampled_latents"],
                    device,
                )
                accum_consistency_encoder_loss += (
                    consistency_encoder_loss.item() * accum_adjustment
                )

            consistency_loss = consistency_encoder_loss * use_consistency_loss

        if model.model_name != "SlotMLPMonolithic":
            with torch.no_grad():
                consistency_decoder_loss = metrics.reconstruction_loss(
                    output_dict["sampled_time_series"],
                    output_dict["reconstructed_sampled_time_series"],
                )
                accum_consistency_decoder_loss += (
                    consistency_decoder_loss.item() * accum_adjustment
                )

        if model.model_name != "SlotMLPMonolithic":
            accum_consistency_loss += consistency_loss.item() * accum_adjustment

        if (use_consistency_loss) and epoch >= consistency_ignite_epoch:
            total_loss += consistency_loss

        accum_total_loss += total_loss.item() * accum_adjustment
        if mode == "train":
            total_loss.backward()
            optimizer.step()

    training_utils.print_metrics_to_console(
        epoch,
        accum_total_loss,
        accum_reconstruction_loss,
        accum_consistency_loss,
        accum_r2_score,
        accum_slots_loss,
        accum_consistency_encoder_loss,
        accum_consistency_decoder_loss,
    )
    return accum_reconstruction_loss


def run(
    *,
    model_name,
    dataset_path,
    checkpoint_path,
    dataset_name,
    device,
    epochs,
    batch_size,
    lr,
    lr_scheduler_step,
    consistency_ignite_epoch,
    unsupervised_mode,
    use_consistency_loss,
    softmax,
    sampling,
    n_slots,
    n_slot_latents,
    sample_mode_train,
    sample_mode_test_id,
    sample_mode_test_ood,
    seed,
    load_checkpoint,
    evaluation_frequency,
):
    dataset_path = os.path.join(dataset_path, dataset_name)
    signature_args = locals().copy()

    training_utils.set_seed(seed)

    os.path.isdir(dataset_path)
    wrapper = src.datasets.wrappers.get_wrapper(
        dataset_name,
        path=dataset_path,
    )

    test_loader_id = wrapper.get_test_loader(
        sample_mode_test=sample_mode_test_id, **signature_args
    )
    test_loader_ood = wrapper.get_test_loader(
        sample_mode_test=sample_mode_test_ood, **signature_args
    )
    train_loader = wrapper.get_train_loader(**signature_args)

    if dataset_name == "ukdale":
        resolution = (256, 1)
        latent_dim = 5
        in_channels = 1

    elif dataset_name == "reded":
        resolution = (256, 1)
        latent_dim = 5
        in_channels = 1

    elif dataset_name == "synth_1":
        resolution = (256, 1)
        latent_dim = 5
        in_channels = 1

    elif dataset_name == "synth_2":
        resolution = (256, 1)
        latent_dim = 7
        in_channels = 1

    elif dataset_name == "synth_3":
        resolution = (256, 1)
        latent_dim = 7
        in_channels = 3
    else:
        NotImplementedError

    if model_name == "TimeCSL":
        model = base_models.TimeCSL(
            in_channels,
            n_slots,
            n_slot_latents,
        ).to(device)

    if model_name == "iVAE":
        model = base_models.iVAE(
            in_channels,
            n_slots,
            n_slot_latents,
        ).to(device)

    if model_name == "DiffusionVAE":
        model = base_models.DiffusionVAE(
            in_channels,
            n_slots,
            n_slot_latents,
        ).to(device)

    if model_name == "DCVAE":
        model = base_models.DiffusionVAE(
            in_channels,
            n_slots,
            n_slot_latents,
        ).to(device)

    elif model_name == "SlowVAE":
        model = base_models.SlowVAE(in_channels, n_slots, n_slot_latents).to(device)
    elif model_name == "SlotAttention":
        encoder = slot_attention.SlotAttentionEncoder(
            resolution=resolution,
            hid_dim=n_slot_latents,
            ch_dim=ch_dim,
        ).to(device)
        decoder = slot_attention.SlotAttentionDecoder(
            hid_dim=n_slot_latents,
            ch_dim=ch_dim,
            resolution=resolution,
        ).to(device)
        model = slot_attention.SlotAttentionAutoEncoder(
            encoder=encoder,
            decoder=decoder,
            num_slots=n_slots,
            num_iterations=3,
            hid_dim=n_slot_latents,
            sampling=sampling,
            softmax=softmax,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2)

    start_epoch = 0
    if load_checkpoint:
        model, optimizer, scheduler, start_epoch = training_utils.load_checkpoint(
            model, optimizer, scheduler, load_checkpoint
        )
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 2

    start_epoch += 1

    for epoch in range(start_epoch, epochs + 1):
        if epoch == consistency_ignite_epoch and use_consistency_loss:
            training_utils.save_checkpoint(
                path=checkpoint_path,
                **locals(),
                checkpoint_name=f"before_ignite_model_{sample_mode_train}_{seed}",
            )
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 2

        rec_loss = one_epoch(
            model,
            train_loader,
            optimizer,
            mode="train",
            epoch=epoch,
            **signature_args,
        )

        if scheduler.get_last_lr()[0] >= 1e-7:
            scheduler.step()

        if scheduler.get_last_lr()[0] > lr:
            optimizer.param_groups[0]["lr"] = lr
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_scheduler_step, gamma=0.5
            )

        if epoch % evaluation_frequency == 0:
            if model_name in ["SlotAttention", "SlotMLPAdditive"] and epoch % 1 == 0:
                if dataset_name == "dsprites":
                    categorical_dimensions = [2]

                id_score_id, id_score_ood = metrics.identifiability_score(
                    model,
                    test_loader_id,
                    test_loader_ood,
                    categorical_dimensions,
                    device,
                )

            id_rec_loss = one_epoch(
                model,
                test_loader_id,
                optimizer,
                mode="test_ID",
                epoch=epoch,
                **signature_args,
            )

            ood_rec_loss = one_epoch(
                model,
                test_loader_ood,
                optimizer,
                mode="test_OOD",
                epoch=epoch,
                **signature_args,
            )

            save_name = f"{model_name}_{sample_mode_train}_{seed}"
            training_utils.save_checkpoint(
                path=checkpoint_path,
                **locals(),
                checkpoint_name=f"{save_name}_{epoch}",
            )

    training_utils.save_checkpoint(
        path=checkpoint_path,
        **locals(),
        checkpoint_name=f"{save_name}_{epoch}",
    )
