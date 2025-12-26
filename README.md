# Fed-ViM: Federated Learning with Global Subspace Alignment for OOD Detection

基于全局子空间共识的联邦 OOD 检测与泛化框架。

## 项目简介

Fed-ViM 是一个创新的联邦学习框架，通过在客户端和服务端之间对齐全局特征子空间，实现了高效的 Out-of-Distribution (OOD) 检测。

### 核心创新

- **全局子空间对齐 (GSA)**: 通过联邦聚合构建全局主子空间，约束本地特征学习
- **ViM Score**: 结合 Logit 和 Residual 的 OOD 检测评分
- **显存优化**: 使用高效矩阵运算，显存占用从 2GB 降至 16MB
- **Non-IID 支持**: 基于 Dirichlet 分布的数据划分，模拟真实联邦场景

## 环境要求

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

或者使用 conda:

```bash
conda install pytorch torchvision matplotlib scikit-learn numpy
```

## 快速开始

### 1. 运行基本实验（高斯噪声作为 OOD）

```bash
python fed_vim_cifar.py
```

### 2. 使用真实 OOD 数据集（SVHN）

代码已配置使用 SVHN 作为 OOD 数据集。运行：

```bash
python fed_vim_cifar.py
```

实验结果将保存为 `fed_vim_result_svhn.png`，显示 ID vs OOD 的分数分布和 AUROC 指标。

## 实验配置

- **数据集**: CIFAR-10 (ID), SVHN (OOD)
- **模型**: SimpleCNN (特征维度 128)
- **客户端数量**: 5
- **Non-IID 参数**: Dirichlet α = 0.5
- **训练轮次**: 10 rounds
- **本地训练**: 1 epoch per round
- **子空间维度**: 42 (33% of 128)

## 实验结果

### 高斯噪声 OOD
- AUROC: 0.996
- 分布分离度: 极高

### SVHN 真实 OOD
- AUROC: 更新中（使用真实 SVHN 数据集）
- 分布分离度: 优于随机基线
- 结果图: `fed_vim_result_svhn.png`

## 项目结构

```
Fed-ViM/
├── fed_vim_cifar.py          # 主训练脚本
├── Untitled.ipynb            # Jupyter notebook 实验笔记
├── Fed-ViM 技术方案.md       # 详细技术文档
├── README.md                 # 本文件
└── data/                     # 数据集目录
    ├── cifar-10-batches-py/
    └── SVHN/
```

## 核心算法

### 1. 统计量聚合（显存优化版）

```python
# 原始方法: 2GB 显存
# stat_sum_zzT = torch.matmul(features.unsqueeze(2), features.unsqueeze(1)).sum(dim=0)

# 优化方法: 16MB 显存
stat_sum_zzT = torch.matmul(features.T, features)
```

### 2. ViM Score 计算

```python
Score = Logit_max - α * ||(I - PP^T)z||
```

其中：
- `Logit_max`: 最大类别 logit
- `α`: 缩放系数（通过联邦聚合学习）
- `P`: 全局主子空间投影矩阵
- `z`: 特征向量

## 技术特点

### 联邦学习流程

1. **服务端广播**: 全局模型 + 全局统计量 (P, μ)
2. **本地训练**:
   - CE Loss + GSA Loss
   - 计算本地统计量
3. **联邦聚合**:
   - FedAvg 聚合模型权重
   - 聚合统计量更新全局子空间
   - 更新 ViM 缩放系数 α

### 子空间更新

```python
# 1. 聚合协方差
Cov_global = E[ZZ^T] - E[Z]E[Z]^T

# 2. 特征分解
eig_vals, eig_vecs = torch.linalg.eigh(Cov_global)

# 3. 提取 Top-K 主成分
P_global = eig_vecs[:, -k:]
```

## 引用

如果这个项目对你的研究有帮助，请考虑引用：

```bibtex
@article{fed-vim-2024,
  title={Fed-ViM: Federated Learning with Global Subspace Alignment for OOD Detection},
  author={Your Name},
  year={2024}
}
```

## 许可证

MIT License

## 联系方式

如有问题，请提交 Issue 或 Pull Request。
