# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fed-ViM is a federated learning framework for Out-of-Distribution (OOD) detection using Global Subspace Alignment. It implements a novel approach where clients and servers align on a global feature subspace learned through federated PCA aggregation, enabling effective OOD detection in privacy-preserving distributed learning scenarios.

**Key Innovation**: Instead of sharing raw features, clients share only first and second-order statistics (mean and covariance), which are aggregated to compute a global principal subspace. This subspace is used to compute residual-based OOD scores.

## Running the Code

### Main Experiment
```bash
python fed_vim_cifar.py > fed_vim_full_run.log 2>&1 &
tail -f fed_vim_full_run.log
```

**Important**: Do NOT use `conda run -n base python` as it has output buffering issues. Use direct `python` instead.

### Expected Outputs
- `fed_vim_result_svhn.png` - OOD detection score distribution (ID vs SVHN)
- `fed_vim_training_curves.png` - Training curves (4 subplots: client losses, client accuracies, test accuracy, alpha coefficient)
- `fed_vim_full_run.log` - Complete training log with per-round client metrics

### Training Configuration
Current setup: **50 rounds × 3 local epochs** (long training time for convergence)
- Data: CIFAR-10 (ID), SVHN (OOD)
- Clients: 5 with Dirichlet α=0.5 (moderate Non-IID)
- Subspace dimension: 20/128 (~16%, fixed)
- Normalization: CIFAR-10 standard (mean: 0.4914, 0.4822, 0.4465; std: 0.2023, 0.1994, 0.2010)

## Code Architecture

This repository has **two implementations** for different purposes:

### 1. fed_vim_cifar.py (Quick Prototype)
**Single-file design** for rapid algorithm validation and debugging.
- **Part A**: Data partitioning (`dirichlet_partition`)
- **Part C**: Core framework (SimpleCNN, GSALoss, Client, Server)
- **Part B**: Evaluation & visualization

**Use this for**:
- Understanding the Fed-ViM algorithm flow
- Quick iteration on hyperparameters
- Debugging core algorithm issues
- Educational demonstrations

### 2. experiments/plankton/ (Production Implementation)
**Modular design** for real-world datasets and paper experiments.

```
experiments/plankton/
├── train_federated.py    # Main entry point with argument parsing
├── client.py             # Client class (NEEDS MODIFICATION for Fed-ViM)
├── server.py             # Server class (NEEDS MODIFICATION for Fed-ViM)
├── models.py             # DenseNet + FedRoD dual-head architecture
├── data_utils.py         # Complex plankton data loading & ID/OOD split
├── eval_utils.py         # Evaluation metrics and report generation
├── split_dataset.py      # Dataset partitioning tool
└── utils/                # Helper utilities
```

**Use this for**:
- Paper experiments on real datasets
- Performance benchmarking
- Production deployment
- Extending to new datasets

**Key files to modify for Fed-ViM integration**:
- [client.py](experiments/plankton/client.py): Add statistics computation in `train_step()`
- [server.py](experiments/plankton/server.py): Add PCA aggregation logic in `aggregate()`
- [models.py](experiments/plankton/models.py): Already has dual-head structure compatible with ViM

### Key Classes

**Client** (`fed_vim_cifar.py:106-194`)
- Receives global model + global statistics (P, μ)
- Local training: CE Loss + λ·GSA Loss
- Returns: model weights, statistics (Σz, Σzzᵀ), ViM stats (Σlogit, Σresidual), epoch_losses, accuracy
- **Critical optimization**: `torch.matmul(features.T, features)` instead of batch outer products (2GB → 16MB)

**Server** (`fed_vim_cifar.py:197-220`)
- Aggregates model weights via FedAvg
- Reconstructs global covariance: `Cov = E[ZZᵀ] - E[Z]E[Z]ᵀ`
- Performs eigendecomposition to extract top-k subspace P
- Updates α coefficient (logit/residual ratio)

**GSALoss** (`fed_vim_cifar.py:93-102`)
- Penalizes deviation from global subspace
- Loss = ‖(I - PPᵀ)z‖ (residual norm after projection)

### Data Flow

```
Round 1..R:
  For each client i:
    1. Download (w_global, P_global, μ_global) from server
    2. Local train: L = CE(y, f_w(x)) + λ·GSA(z, P, μ)
    3. Compute local stats: Σz, Σzzᵀ, Σmax_logit, Σresidual
    4. Upload (w_i, stats_i) to server

  Server aggregates:
    1. w_global = FedAvg(w_1, ..., w_N)
    2. μ_global = average(Σz_i)
    3. Cov = (ΣΣzzᵀ_i)/N - μ_globalμ_globalᵀ
    4. P_global = top-k eigenvectors(Cov)
    5. α = (Σmax_logit) / (Σresidual)
```

## Critical Implementation Details

### Memory Optimization
```python
# ❌ WRONG: Creates (Batch, D, D) tensor → ~2GB for ResNet50
stat_sum_zzT = torch.matmul(features.unsqueeze(2), features.unsqueeze(1)).sum(dim=0)

# ✅ RIGHT: Uses matrix multiplication → ~16MB
stat_sum_zzT = torch.matmul(features.T, features)
```

### Matplotlib Backend
**Must set non-interactive backend before importing pyplot**:
```python
import matplotlib
matplotlib.use('Agg')  # Line 9 - prevents GUI hang
import matplotlib.pyplot as plt
```

### Subspace Dimension
Current: **20 dimensions** (line 201)
- Lower dimension → stricter subspace → larger OOD residuals → better detection
- Too low → loss of discriminative power
- If `OOD_Res < ID_Res`, consider reducing dimension further

### Normalization Consistency
Both ID (CIFAR-10) and OOD (SVHN) use the **same** normalization parameters (CIFAR-10 stats). This ensures distribution shift is genuine, not artifact of different preprocessing.

## Troubleshooting

### "conda run" Shows No Output
**Problem**: Output buffering causes silent hanging
**Solution**: Use direct `python fed_vim_cifar.py`

### OOD Residual < ID Residual
**Diagnosis**:
```python
# Check evaluate_and_plot() output
if np.mean(ood_res) < np.mean(id_res):
    print("⚠️ WARNING: Subspace too large or features not learned")
```
**Fixes**:
1. Reduce subspace dimension (line 201: `k = 20`)
2. Increase training rounds/epochs
3. Check if model has converged (test accuracy > 60%)

### Low AUROC (< 0.7)
**Common causes**:
1. Insufficient training (current: 50 rounds × 3 epochs)
2. Subspace dimension too large
3. λ_gsa weight too small/large (current: 0.1)

## Modifying the Framework

### Adding New OOD Datasets
In `main()` around line 372:
```python
new_ood_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
new_ood_data = CustomDataset(root='./data', transform=new_ood_transform)
```

### Changing Non-IID Severity
Modify Dirichlet α in `main()` line 360:
- `alpha=0.1` → Highly Non-IID (clients have very skewed class distributions)
- `alpha=1.0` → IID (uniform distribution)
- `alpha=0.5` → Moderate (current)

### Adjusting Training Speed
Line 393-394:
```python
rounds = 50  # Reduce to 20 for faster iteration
local_epochs = 3  # Reduce to 1 for faster iteration
```

## Key Equations

**ViM Score** (for OOD detection):
```
Score(x) = max_logit(f_w(x)) - α × ‖(I - PPᵀ)(f_w(x) - μ)‖
```
Lower score → more likely OOD

**GSA Loss** (for training):
```
L_GSA = ‖(I - PPᵀ)(z - μ)‖₂
```
Penalizes features outside global consensus subspace

## Running experiments/plankton

### Prerequisites

1. **Install dependencies**:
```bash
cd experiments/plankton
pip install -r requirements.txt
```

2. **Prepare dataset**:
The plankton dataset should be placed in the parent directory:
```
../../Plankton_OOD_Dataset/
```

3. **Verify environment**:
```bash
python utils/test_pipeline.py
```

### Running Training

**Basic run** (using default hyperparameters):
```bash
python train_federated.py
```

**Custom configuration**:
```bash
python train_federated.py \
    --num_clients 5 \
    --alpha 0.5 \
    --rounds 50 \
    --local_epochs 3 \
    --lr 0.001 \
    --gpu_id 0
```

### Key Hyperparameters in plankton/experiments

- `--num_clients`: Number of federated clients (default: 5)
- `--alpha`: Dirichlet concentration for Non-IID data
  - 0.1: Highly Non-IID
  - 0.5: Moderate (current default)
  - 1.0: IID
- `--rounds`: Number of federated rounds (default: 50)
- `--local_epochs`: Local training epochs per round (default: 3)
- `--lr`: Learning rate (default: 0.001)
- `--gpu_id`: GPU device ID (default: 0)

### Expected Outputs

Training will generate:
- Checkpoints: `experiments/YYYYMMDD-HHMMSS/`
- Training logs: Console output with per-round metrics
- Evaluation results: Saved in checkpoint directory

## Integrating Fed-ViM into experiments/plankton

### Step 1: Modify client.py

Add statistics computation in the `train_step()` method:

```python
# In client.py, around line 100-150
def train_step(self, batch, global_P, global_mu):
    images, labels = batch
    features = self.model.forward_features(images)  # Extract features

    # ✅ NEW: Compute local statistics
    with torch.no_grad():
        batch_size = features.shape[0]
        stat_sum_z = features.sum(dim=0)
        stat_sum_zzT = torch.matmul(features.T, features)  # Memory-efficient!

    # Existing training code...
    logits = self.model.classifier(features)
    loss = self.criterion(logits, labels)

    # Compute ViM statistics
    with torch.no_grad():
        max_logit = logits.max(dim=1).values
        residual = torch.matmul(torch.eye(features.shape[1]) - global_P @ global_P.T,
                                (features - global_mu).T)
        stat_sum_max_logit = max_logit.sum()
        stat_sum_residual = residual.norm(dim=0).sum()

    return loss, {
        'stat_sum_z': stat_sum_z,
        'stat_sum_zzT': stat_sum_zzT,
        'stat_sum_max_logit': stat_sum_max_logit,
        'stat_sum_residual': stat_sum_residual
    }
```

### Step 2: Modify server.py

Add PCA aggregation in the `aggregate()` method:

```python
# In server.py, around line 80-120
def aggregate(self, client_updates):
    # Existing FedAvg for model weights
    aggregated_weights = self._fedavg_weights(client_updates)

    # ✅ NEW: Aggregate statistics
    total_samples = sum(update['num_samples'] for update in client_updates)

    # Aggregate first-order statistics (mean)
    mu_global = sum(update['stats']['stat_sum_z'] * update['num_samples']
                    for update in client_updates) / total_samples

    # Aggregate second-order statistics (covariance)
    E_zzT = sum(update['stats']['stat_sum_zzT'] * update['num_samples']
                for update in client_updates) / total_samples
    Cov_global = E_zzT - torch.outer(mu_global, mu_global)

    # Perform PCA to extract subspace
    eig_vals, eig_vecs = torch.linalg.eigh(Cov_global)
    k = 20  # Subspace dimension
    P_global = eig_vecs[:, -k:]  # Top-k eigenvectors

    # Compute α coefficient for ViM
    total_max_logit = sum(update['stats']['stat_sum_max_logit']
                          for update in client_updates)
    total_residual = sum(update['stats']['stat_sum_residual']
                         for update in client_updates)
    alpha = total_max_logit / total_residual

    return {
        'weights': aggregated_weights,
        'P_global': P_global,
        'mu_global': mu_global,
        'alpha': alpha
    }
```

### Step 3: Update train_federated.py

Modify the training loop to pass global statistics to clients and receive their statistics back:

```python
# In train_federated.py, main training loop
for round_idx in range(args.rounds):
    # Server broadcasts global model + statistics
    global_state = {
        'model': server.global_model,
        'P': server.global_P,
        'mu': server.global_mu
    }

    client_updates = []
    for client in clients:
        # Client trains with global statistics
        update = client.train(global_state)
        client_updates.append(update)

    # Server aggregates weights + statistics
    aggregated = server.aggregate(client_updates)
    server.update_global_state(aggregated)
```

## Migration Checklist

When migrating Fed-ViM from `fed_vim_cifar.py` to `experiments/plankton/`:

- [ ] Add GSALoss class to [models.py](experiments/plankton/models.py)
- [ ] Modify [client.py](experiments/plankton/client.py) to compute local statistics
- [ ] Modify [server.py](experiments/plankton/server.py) to perform PCA aggregation
- [ ] Update [train_federated.py](experiments/plankton/train_federated.py) main loop
- [ ] Test on small dataset (e.g., 1 round, 2 clients)
- [ ] Verify OOD detection performance
- [ ] Compare with baseline results
