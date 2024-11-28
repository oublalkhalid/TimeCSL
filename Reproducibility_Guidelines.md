
# Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing
Official code for the paper **Identifiability Guarantees For Time Series Representation via Contrastive Sparsity-inducing**

We define the identifiability problem for time series variable models, where factors are represented by latent slots.  

We are thrilled to announce the release of 221 models, now available in the `checkpoints` folder and downloadable from [https://huggingface.co/anonymousModelsTimeCSL/TimeCSL](https://huggingface.co/anonymousModelsTimeCSL/TimeCSL). These models are part of our commitment to advancing machine learning and equipping the community with state-of-the-art tools for time series analysis. Visit the repository to explore the models and seamlessly integrate them into your projects.

Our code is available at https://anonymous.4open.science/r/TimeCSL-4320.

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

## Time series representation (demo)

🔗 For understanding how the model looks and to play with the notebook, please refer to the `notebooks/.ipynb` notebook.

## Training and Evaluation

### Training
To train the model, run the following command:

```bash
cd src_timecsl/
python train.py --dataset_path last_step" --model_name "TimeCSL" --num_slots 2 --epochs 200 --use_invariance_loss True
```

For complete details on the parameters, please refer to the `main.py` file.

You can find some example commands for training below:

<details open>
<summary><strong>Different Training Setups</strong></summary>

- <details>
  <summary><strong>Training SlotAttention</strong></summary>

  Training vanilla TimeCSL with 5 slots (latent of size n=5 and d=16):
  ```bash
  python main.py --dataset_path "/path/from/previous/step" --model_name "TimeCSL" --num_slots 5 --use_generalization_loss False
  ```

  Training vanilla iVAE with 2 slots and consistency loss:
  ```bash
  python main.py --dataset_path "/path/from/previous/step" --model_name "iVAE" --num_slots 5 --use_generalization_loss True --consistency_ignite_epoch 150
  ```

  Training SlowVAE with 2 slots, fixed SoftMax and sampling:
  ```bash
  python main.py --dataset_path "/path/from/previous/step" --model_name "SlowVAE" --num_slots 5 --use_generalization_loss True --consistency_ignite_epoch 150 --softmax False --sampling False
  ```
</details>

</details>

### Evaluation

Evaluation can be done using the `evaluate.py` script and closely follows the procedure and metrics used in the training script. The main difference is in calculating the compositional contrast (note: it might cause OOM issues, thus is calculated only for the AE model).

Here is an example command for evaluation:
```bash
python evaluation.py --dataset_path "/path/from/previous/step" --model_path "checkpoints/SlotMLPAdditive.pt" --model_name "TimeCSL" --n_slot_latents 5
```

### Metrics for Identifiability and Disentanglement (RMIG, DCI, betaVAE metrics) 

We implement two types of MCC:

- **Weak MCC**: Computed after aligning sequences, measuring pattern similarity without considering order.
- **Strong MCC**: Calculated without alignment, preserving sequence order to assess accuracy.

Both metrics offer complementary insights into sequence prediction performance. Both metrics provide complementary insights into sequence prediction performance. Additional metrics can be found in the `metrics/` folder.

**Sequence $MCC$**

```python
from sklearn.metrics import matthews_corrcoef
import torch

def sequences_mcc(true_sequences: torch.Tensor, predicted_sequences: torch.Tensor) -> float:
    """
    Calculates Matthews Correlation Coefficient (MCC) for sequences. Used for sequence classification evaluation.

    Args:
        true_sequences: tensor of shape (batch_size, n_channels, T)
        predicted_sequences: tensor of shape (batch_size, n_channels, T)

    Returns:
        mcc_score: Matthews Correlation Coefficient
    """

    # Flatten the sequences into 1D arrays for classification
    true_sequences_flat = true_sequences.view(-1).cpu().numpy()
    predicted_sequences_flat = predicted_sequences.view(-1).cpu().numpy()

    # Calculate MCC using sklearn's matthews_corrcoef
    mcc_score = matthews_corrcoef(true_sequences_flat, predicted_sequences_flat)

    return mcc_score

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


## 📢 Reproducing the Results from the Paper

The code used to generate the results from the experiments in Sections 5.2 to 5.6 is provided. All scripts are located in the `src/experiments/` folder. Each script can be executed in two modes:
### Section 4.1 of our paper
- `run section4.1_disentnaglement_identifiability.py`: Training,Computes and saves metric scores
- plots: Some plots are generated after runing ``section4.1_disentnaglement_identifiability`` and saves plots for each metric.

### Section 4.1 of our paper
- `run section4.2_generalization.py`: Trains the model, computes, and saves metric scores.
- Plots: Generated automatically after running `section4.2_generalization.py` and saved for each metric.

Both the metric scores and plots are automatically saved in the `results/` folder at the root of the repository.

To reproduce the experiments with synthetic data or datasets like REDD/REFIT/UKDALE (available in the `dataset/` folder) or to generate new synthetic data: 
1. Navigate to the dataset folder: `cd dataset/`
2. Run the script to generate data: `python generate_dataset.py run`. The data will be saved in the `dataset/` folder as `synthetic_<name>_<version>.csv`.
