# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fed-ViM is a federated learning framework for Out-of-Distribution (OOD) detection using Global Subspace Alignment. It implements a novel approach where clients and servers align on a global feature subspace learned through federated PCA aggregation, enabling effective OOD detection in privacy-preserving distributed learning scenarios.

**Key Innovation**: Instead of sharing raw features, clients share only first and second-order statistics (mean and covariance), which are aggregated to compute a global principal subspace. This subspace is used to compute residual-based OOD scores.

## Architecture Overview

The repository contains **two independent implementations**:

| Aspect | fed_vim_cifar.py | experiments/plankton/ |
|--------|------------------|----------------------|
| **Purpose** | Quick algorithm validation | Production experiments |
| **Design** | Single-file prototype | Modular architecture |
| **Dataset** | CIFAR-10 + SVHN | Plankton (26 classes) |
| **Model** | SimpleCNN (128-dim) | DenseNet121/169 (1024/1664-dim) |
| **Fed-ViM Status** | ✅ Fully implemented | ✅ Fully implemented |
| **Use Case** | Understanding algorithm, debugging | Paper experiments, benchmarking |
| **File Count** | 1 file | 8+ files (client, server, models, data, eval) |

**Key Point**: Both implementations are **independently functional**. The CIFAR version is NOT a dependency for the plankton version.

## Running the Code

### Quick Reference

**CIFAR-10 Quick Prototype** (from repository root):
```bash
python fed_vim_cifar.py
```

**Plankton Fed-ViM Training** (from experiments/plankton/):
```bash
cd experiments/plankton
python train_federated.py --use_fedvim --data_root ../../Plankton_OOD_Dataset
```

**Plankton FOOGD Training** (generative OOD baseline):
```bash
cd experiments/plankton
python train_federated.py --use_foogd --data_root ../../Plankton_OOD_Dataset
```

**Plankton Baseline** (no OOD module):
```bash
cd experiments/plankton
python train_federated.py --data_root ../../Plankton_OOD_Dataset
```

### Main Experiment (CIFAR-10)
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

**Fed-ViM Integration Status** ✅ **COMPLETED**
- [client.py](experiments/plankton/client.py): ✅ Added `use_fedvim` parameter, GSA Loss computation, and `_compute_local_statistics()` method
- [server.py](experiments/plankton/server.py): ✅ Added `update_global_subspace()` method for PCA aggregation
- [models.py](experiments/plankton/models.py): ✅ Added `GSALoss` class
- [train_federated.py](experiments/plankton/train_federated.py): ✅ Integrated Fed-ViM training loop with `--use_fedvim` flag

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

### Dataset Not Found Error
**Error**: `ValueError: num_samples should be a positive integer value, but got num_samples=0`
**Cause**: Incorrect `--data_root` path
**Solution**: Ensure dataset path points to `Plankton_OOD_Dataset`:
```bash
# From experiments/plankton/
--data_root ../../Plankton_OOD_Dataset

# Or use absolute path
--data_root /home/dell7960/桌面/FedRoD/Fed-ViM/Plankton_OOD_Dataset
```

### OOD Residual < ID Residual
**Diagnosis**:
```python
# Check evaluate_and_plot() output
if np.mean(ood_res) < np.mean(id_res):
    print("⚠️ WARNING: Subspace too large or features not learned")
```
**Fixes**:
1. Reduce subspace dimension (line 201: `k = 20` for fed_vim_cifar.py)
2. For plankton experiments, reduce k in `server.update_global_subspace()` call
3. Increase training rounds/epochs
4. Check if model has converged (test accuracy > 60%)

### Low AUROC (< 0.7)
**Common causes**:
1. Insufficient training (current: 50 rounds × 3-4 epochs)
2. Subspace dimension too large
3. λ_gsa weight too small/large (current: 1.0 for plankton)
4. Model not converged (check test accuracy)

### Memory Issues During Training
**Symptoms**: CUDA out of memory
**Solutions**:
1. Reduce `--batch_size` (default: 64, try 32 or 16)
2. Reduce `--image_size` (default: 224, try 160)
3. Use smaller model: `--model_type densenet121` instead of `densenet169`
4. Reduce number of clients: `--num_clients 5` instead of 10

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
The plankton dataset location is:
```
/home/dell7960/桌面/FedRoD/Fed-ViM/Plankton_OOD_Dataset/
```
When running from `experiments/plankton/`, use:
```bash
--data_root ../../Plankton_OOD_Dataset
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
    --gpu_id 0 \
    --data_root ../../Plankton_OOD_Dataset
```

**Fed-ViM Training** (with Global Subspace Alignment):
```bash
python train_federated.py \
    --use_fedvim \
    --alpha 0.1 \
    --communication_rounds 50 \
    --model_type densenet121 \
    --batch_size 64 \
    --image_size 320 \
    --base_lr 0.01 \
    --data_root ../../Plankton_OOD_Dataset \
    --output_dir ./experiments_fedvim_dense121_320
```

### Key Hyperparameters in plankton/experiments

- `--num_clients`: Number of federated clients (default: 10)
- `--alpha`: Dirichlet concentration for Non-IID data
  - 0.1: Highly Non-IID
  - 0.5: Moderate
  - 1.0: IID (uniform distribution)
- `--communication_rounds`: Number of federated rounds (default: 50)
- `--local_epochs`: Local training epochs per round (default: 4)
- `--base_lr`: Learning rate (default: 0.001)
- `--model_type`: Backbone network - `densenet121` (1024-dim) or `densenet169` (1664-dim)
- `--use_fedvim`: Enable Fed-ViM (GSA Loss + PCA subspace alignment)
- `--use_foogd`: Enable FOOGD (generative OOD with score matching)
- `--image_size`: Input image size (default: 224, recommended: 320)
- `--data_root`: Path to Plankton_OOD_Dataset (default: `./Plankton_OOD_Dataset`)

### Expected Outputs

Training will generate:
- Checkpoints: `experiments/YYYYMMDD-HHMMSS/`
- Training logs: Console output with per-round metrics
- Evaluation results: Saved in checkpoint directory

## Fed-ViM Implementation Details (experiments/plankton)

### Client-Side Changes ([client.py](experiments/plankton/client.py))

**Key Modifications**:
1. **Constructor** (line 20-34): Added `use_fedvim` parameter and initializes `GSALoss` when enabled
2. **train_step** (line 88-222):
   - Accepts `global_stats` dict containing `P_global` and `mu_global`
   - Computes GSA Loss on both original and augmented features (line 182-191)
   - Returns `client_vim_stats` via `_compute_local_statistics()` (line 242-243)
3. **_compute_local_statistics** (line 330-354): Memory-efficient statistics computation
   ```python
   sum_z += features.sum(dim=0)
   sum_zzT += torch.matmul(features.T, features)  # Critical optimization!
   ```

**GSA Loss Computation** (line 182-191):
```python
if self.use_fedvim and P_global is not None:
    loss_clean = self.gsa_criterion(features, P_global, mu_global)
    loss_aug = self.gsa_criterion(features_aug, P_global, mu_global)
    gsa_loss = 0.5 * loss_clean + 0.5 * loss_aug  # Average over clean + aug
```

### Server-Side Changes ([server.py](experiments/plankton/server.py))

**Key Addition** - `update_global_subspace` method (line 98-153):
```python
def update_global_subspace(self, client_stats_list, k=64):
    # 1. Aggregate sufficient statistics
    total_count = sum([s['count'] for s in client_stats_list])
    global_sum_z = sum(s['sum_z'] for s in client_stats_list)
    global_sum_zzT = sum(s['sum_zzT'] for s in client_stats_list)

    # 2. Reconstruct global covariance
    mu_global = global_sum_z / total_count
    E_zzT = global_sum_zzT / total_count
    cov_global = E_zzT - torch.outer(mu_global, mu_global)

    # 3. Eigendecomposition
    eig_vals, eig_vecs = torch.linalg.eigh(cov_global)
    self.P_global = eig_vecs[:, -k:]  # Top-k eigenvectors

    return {'P': self.P_global, 'mu': self.mu_global}
```

### Training Loop Integration ([train_federated.py](experiments/plankton/train_federated.py))

**Key Changes**:
1. **Initialize global statistics** (line 331):
   ```python
   global_vim_stats = {'P': None, 'mu': None}
   ```

2. **Client training with statistics** (line 356-368):
   ```python
   client_update, client_loss, client_stats = client.train_step(
       local_epochs=args.local_epochs,
       current_round=round_num,
       global_stats=global_vim_stats  # Broadcast P, mu to clients
   )
   if args.use_fedvim and client_stats is not None:
       client_vim_stats_list.append(client_stats)
   ```

3. **Server subspace update** (line 380-383):
   ```python
   if args.use_fedvim and len(client_vim_stats_list) > 0:
       global_vim_stats = server.update_global_subspace(client_vim_stats_list, k=64)
   ```

### GSA Loss Implementation ([models.py](experiments/plankton/models.py))

The `GSALoss` class (line 101-134) computes the residual norm outside the global subspace:
```python
def forward(self, features, P_global, mu_global):
    P = P_global.detach()  # Block gradient flow to global parameters
    mu = mu_global.detach()

    z_centered = features - mu
    z_projected_coeffs = torch.matmul(z_centered, P)
    z_reconstructed = torch.matmul(z_projected_coeffs, P.T)

    diff = z_centered - z_reconstructed
    loss = torch.norm(diff, p=2, dim=1).mean()  # Residual norm
    return loss
```

### Critical Design Decisions

1. **Subspace Dimension**: k=64 for DenseNet (1024-1664 feature dimensions)
   - Higher k → more discriminative power but weaker OOD separation
   - Lower k → stricter subspace constraint → larger OOD residuals

2. **GSA Loss Weight**: λ_gsa = 1.0 (client.py line 194)
   - Balances classification accuracy vs. subspace alignment

3. **Gradient Detachment**: Essential to prevent backpropagation to global P and μ
   - Applied in GSALoss.forward() using `.detach()`

4. **First Round Handling**: GSA Loss is skipped when P_global is None (first round)
   - Allows model to learn initial features before subspace constraint

5. **Augmentation Strategy**: GSA Loss applied to both clean and Fourier-augmented features
   - Improves robustness to style variations
   - Equal weighting (0.5 each)
