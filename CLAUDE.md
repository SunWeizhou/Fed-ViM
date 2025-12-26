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

### Single-File Design
The entire framework is implemented in `fed_vim_cifar.py` with clear section markers:
- **Part A**: Data partitioning (`dirichlet_partition`)
- **Part C**: Core framework (SimpleCNN, GSALoss, Client, Server)
- **Part B**: Evaluation & visualization

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
