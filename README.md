
# Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing
Official code for the paper **Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing**

We formalize identifiability  problem for time series variable model where factors are represented by latent slots. 

We are excited to announce the release of 221 models, now available for exploration and download at [Hugging Face](https://huggingface.co/IPP-Paris/TimeCSL/tree/main). These models are part of our ongoing effort to push the boundaries of machine learning and provide the community with cutting-edge tools for time series analysis. Visit the repository to access the models and integrate them into your projects.

![Overview](assets/fig1_v12.png)

## Environment Setup
This code was tested for Python 3.10. 

Then, set up your environment by choosing one of the following methods:

<details open>
<summary><strong>Option 1: Installing Dependencies Directly</strong></summary>

```bash
pip install -r requirements.txt
```

</details>

Or, alternatively, you can use Docker:

<details open>
<summary><strong>Option 2: Building a Docker</strong></summary>

Build and run a Docker container using the provided Dockerfile:
```bash
docker build -t TimeCSL .
docker-compose up
```

</details>

## Time series representation

🔗 For understanding how the data looks and to play with the data generation, please refer to the `notebooks/0.  Dataset Example.ipynb` notebook.

🔗 For the actual data generation, please refer to the `notebooks/1. Data.ipynb` notebook. The folder used for saving the dataset at this point will be used for training and evaluation.

## Training and Evaluation

### Training
To train the model, run the following command:

```bash
python main.py --dataset_path last_step" --model_name "TimeCSL" --num_slots 2 --epochs 200 --use_addititvity_loss True
```

For complete details on the parameters, please refer to the `main.py` file.

You can find some example commands for training below:

<details open>
<summary><strong>Different Training Setups</strong></summary>

- <details>
  <summary><strong>Training SlotAttention</strong></summary>

  Training vanilla SlotAttention with 2 slots:
  ```bash
  python main.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss False
  ```

  Training vanilla SlotAttention with 2 slots and consistency loss:
  ```bash
  python main.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss True --consistency_ignite_epoch 150
  ```

  Training SlotAttention with 2 slots, fixed SoftMax and sampling:
  ```bash
  python main.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss True --consistency_ignite_epoch 150 --softmax False --sampling False
  ```
</details>

- <details>
  <summary><strong>Training AE Model</strong></summary>

  Training vanilla autoencoder with 2 slots:
  ```bash
  python main.py --dataset_path "/path/from/previous/step" --model_name "SlotMLPAdditive" --epochs 300 --num_slots 2 -n_slot_latents 6 --use_consistency_loss False
  ```

  Training vanilla autoencoder with 2 slots and consistency loss:
  ```bash
  python main.py --dataset_path "/path/from/previous/step" --model_name "SlotMLPAdditive" --epochs 300 --num_slots 2 -n_slot_latents 6 --use_consistency_loss True --consistency_ignite_epoch 100
  ```

</details>

</details>

### Evaluation

Evaluation can be done using the `evaluate.py` script and closely follows the procedure and metrics used in the training script. The main difference is in calculating the compositional contrast (note: it might cause OOM issues, thus is calculated only for the AE model).

Here is an example command for evaluation:
```bash
python src/evaluation.py --dataset_path "/path/from/previous/step" --model_path "checkpoints/SlotMLPAdditive.pt" --model_name "SlotMLPAdditive" --n_slot_latents 6
```

### Metrics

```python
def r2_score(
    true_latents: torch.Tensor, predicted_latents: torch.Tensor, indices: torch.Tensor
) -> Tuple[int, torch.Tensor]:
    """
    Calculates R2 score. Slots are flattened before calculating R2 score.

    Args:
        true_latents: tensor of shape (batch_size, n_slots, n_latents)
        predicted_latents: tensor of shape (batch_size, n_slots, n_latents)
        indices: tensor of shape (batch_size, n_slots, 2) with indices of matched slots

    Returns:
        avg_r2_score: average R2 score over all latents
        r2_score_raw: R2 score for each latent
    """
    indices = torch.LongTensor(indices)
    predicted_latents = predicted_latents.detach().cpu()
    true_latents = true_latents.detach().cpu()

    # shuffling predicted latents to match true latents
    predicted_latents = predicted_latents.gather(
        1,
        indices[:, :, 1].unsqueeze(-1).expand(-1, -1, predicted_latents.shape[-1]),
    )
    true_latents = true_latents.flatten(start_dim=1)
    predicted_latents = predicted_latents.flatten(start_dim=1)
    r2 = R2Score(true_latents.shape[1], multioutput="raw_values")
    r2_score_raw = r2(predicted_latents, true_latents)
    r2_score_raw[torch.isinf(r2_score_raw)] = torch.nan
    avg_r2_score = torch.nanmean(r2_score_raw).item()
    return avg_r2_score, r2_score_raw
```

**Sequence $R^2$**

```python
def sequences_r2_score(true_sequences: torch.Tensor, predicted_sequences: torch.Tensor) -> float:
    """
    Calculates R2 score for sequences. Used for sequence reconstruction evaluation.

    Args:
        true_sequences: tensor of shape (batch_size, n_channels, T)
        predicted_sequences: tensor of shape (batch_size, n_channels, T)

    Returns:
        reconstruction_error: R2 score
    """

    r2_vw = R2Score(
        num_outputs=np.prod(true_sequences.shape[1:]), multioutput="variance_weighted"
    ).to(true_sequences.device)

    # add eps to avoid division by zero
    true_sequences += 1e-7

    reconstruction_error = r2_vw(
        predicted_images.reshape(predicted_sequences.shape[0], -1),
        true_sequences.reshape(true_sequences.shape[0], -1),
    )

    return reconstruction_error
```