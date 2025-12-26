# Fed-ViM 技术方案：基于全局子空间共识的联邦 OOD 检测与泛化框架

## 1. 问题定义与动机 (Problem Statement & Motivation)

在联邦学习（FL）场景下，我们面临两个核心挑战：

1. **数据异质性（Non-IID）导致的特征空间偏斜**：由于每个客户端仅拥有局部数据分布，其学习到的特征主子空间（Principal Subspace）是片面的。直接在本地应用基于特征残差的 OOD 检测（如 ViM 3333），会因为子空间估计不准而失效。

   

2. **通信与计算效率的制约**：现有的联邦 OOD 方法（如 FOOGD 4）通常依赖于训练额外的生成模型（如 Score Model）来估计全局密度。这种方法计算开销大（需要 Langevin 采样 5），且参数传输增加了通信负担。

   

3. **Fed-ViM 的核心思想**：利用**分布式 PCA** 思想，在服务端构建轻量级的“全局统计共识”（全局均值与子空间），并将其反馈给客户端。这不仅实现了低成本的全局 OOD 检测，还通过 **GSA Loss** 强迫本地模型向全局特征流形对齐，从而解决 Non-IID 问题。

------

## 2. 数学框架与核心算法 (Methodology)

假设有 $K$ 个客户端，第 $k$ 个客户端拥有数据集 $\mathcal{D}_k = \{(x, y)\}$。全局模型参数为 $W$。

### 阶段一：服务端全局统计共识 (Server-Side Global Consensus)

此阶段的目标是利用各客户端上传的统计量，重构全局特征分布的低维流形（主子空间 $P$）。

1. 统计量聚合：

   服务端接收各客户端上传的权重 $w_k$、局部特征均值 $\mu_k$ 和局部二阶原点矩 $S_k = \mathbb{E}[zz^T]$。

   - 全局均值 (Global Mean)：

     $$\mu_{global} = \sum_{k=1}^K w_k \mu_k$$

     

   - 全局协方差矩阵 (Global Covariance)：

     利用统计学性质 $Cov(Z) = E[ZZ^T] - E[Z]E[Z]^T$，重构全局中心化协方差：

     $$\Sigma_{global} = \left( \sum_{k=1}^K w_k S_k \right) - \mu_{global}\mu_{global}^T$$

     

2. 子空间提取 (Subspace Extraction)：

   对 $\Sigma_{global}$ 进行特征值分解（Eigendecomposition）：

   $$\Sigma_{global} U = U \Lambda$$

   选取对应前 $D'$ 个最大特征值的特征向量组成矩阵 $P \in \mathbb{R}^{D \times D'}$，这代表了全局数据的**主子空间 (Principal Subspace)** 6。

   

3. 全局 ViM 参数计算：

   为了平衡 Logits 和残差的尺度，需要计算缩放系数 $\alpha$ 7。服务端聚合各端上传的平均最大 Logit ($\bar{L}_k$) 和平均残差 ($\bar{R}_k$)：

   $$\alpha_{global} = \frac{\sum w_k \bar{L}_k}{\sum w_k \bar{R}_k}$$

### 阶段二：客户端子空间感知训练 (Client-Side Subspace-Aware Training)

在第 $t$ 轮训练中，客户端利用上一轮下发的全局子空间 $P_{t-1}$ 来约束本地特征学习。

1. 双流特征提取：

   对于输入 $x$，生成原始特征 $z = f(x)$ 和增强样本特征 $\hat{z} = f(\text{Aug}(x))$。

   

2. GSA Loss (Global Subspace Alignment)：

   为了提高泛化性并克服 Non-IID，我们要求增强样本的特征 $\hat{z}$ 必须落在全局共识子空间 $P$ 内。定义 GSA Loss 为特征在 $P$ 之外的投影（即残差）的范数：

   $$\mathcal{L}_{GSA} = \| (I - P_{t-1}P_{t-1}^T)(\hat{z} - \mu_{global}) \|^2$$

   注：此处的 $\hat{z}$ 需先进行中心化处理。

   

3. 总优化目标：

   $$\min_{\theta} \mathcal{L}_{CE}(z, y) + \lambda \cdot \mathcal{L}_{GSA}$$

   这与 FOOGD 中的 SAG 模块 8 异曲同工，旨在利用全局信息正则化特征提取器，但计算代价仅为一次矩阵乘法。

   

4. 统计量计算与上传：

   训练结束后，客户端遍历本地数据计算 $\mu_k, S_k, \bar{L}_k, \bar{R}_k$ 并上传。

### 阶段三：推理与 OOD 检测 (Inference Phase)

在测试阶段，使用全局参数进行标准的 ViM 检测流程 ：

1. 计算 Logit Score：$S_{logit} = \max_c (\text{Softmax}(W^T z + b))$ 或 Energy Score。

2. 计算残差 Score：$S_{res} = \| (I - P P^T)(z - \mu_{global}) \|$。

3. 最终 ViM Score：

   $$\text{Score}(x) = S_{logit} - \alpha_{global} \cdot S_{res}$$

   

   (注：这里取负号是因为残差越大越可能是 OOD，通常 ViM 会将其转化为概率或 Energy 形式，此处仅为示意)

------

## 3. 算法流程 (Algorithm Workflow)

Input: 通信轮次 $T$，客户端数量 $K$，局部 Epoch $E$。

Initialize: 全局模型 $W_0$，初始全局子空间 $P_0$ (可选随机初始化或 Warm-up)。

For round $t = 0, 1, ..., T-1$ do:

1. Server 分发: 下发 $W_t, P_t, \mu_{global}, \alpha_t$ 给活跃客户端。
2. Client Update (并行):

\* 接收全局信息。

\* If $t > 0$: 使用 $P_t$ 计算 $\mathcal{L}_{GSA}$ 辅助训练。

\* If $t == 0$: 仅使用 $\mathcal{L}_{CE}$ 训练 (Warm-up)。

\* 更新本地模型 $W_k$。

\* 计算本地统计量: $\mu_k, S_k = \mathbb{E}[zz^T]$。

\* 上传 $W_k$ 及统计量。

3. Server Aggregation:

\* 聚合模型: $W_{t+1} \leftarrow \sum w_k W_k$。

\* 聚合统计量: 计算 $\mu_{global}$ 和 $\Sigma_{global}$。

\* 执行 SVD: 更新 $P_{t+1}$。

\* 更新 $\alpha_{t+1}$。

End For

------

## 4. 实施细节与可行性分析

### 4.1 通信开销对比

- **FOOGD**: 需传输 Score Model（通常是一个 MLP 或 U-Net），参数量可能与 Backbone 相当。
- **Fed-ViM**: 仅需传输 $P$ (大小 $D \times D'$) 和 $\Sigma$ 的相关统计量。
  - *优化*：当特征维度 $D$ 较大（如 ResNet-50 的 2048）时，直接传输协方差矩阵 ($2048^2$) 可能较大。此时可以让客户端先在本地做 PCA，只上传 Top-K 特征向量，服务端做近似聚合；或者使用 Power Iteration 方法。对于 CIFAR 实验（通常 $D=512$），直接传输矩阵是完全可接受的。

### 4.2 隐私保护 (Privacy)

- 上传协方差矩阵 $\Sigma$ 相比上传原始数据更为安全，但仍可能受到重构攻击。

- *扩展性*：该框架天然兼容 **差分隐私 (Differential Privacy)**。可以在上传的 $\mu_k$ 和 $S_k$ 上添加高斯噪声，这是联邦统计学中的标准操作，你可以将其作为未来的一个讨论点 。

  

  

------

## 5. 实验设计计划 (Experimental Plan)

为了验证该方案，建议按照以下设置进行实验：

- **数据集**:
  - **ID (In-Distribution)**: CIFAR-10 (按 Dirichlet 分布 $\alpha=0.1, 0.5$ 划分，模拟强 Non-IID) 11。
  - **OOD (Detection)**: SVHN, LSUN, Texture, Places365 12。
  - **OOD (Generalization)**: CIFAR-10-C (Corrupted) 13。
- **Backbone**: ResNet-18 或 WideResNet-40-2。
- **对比基线 (Baselines)**:
  1. **FedAvg**: 仅做聚合，使用 MSP 或 Energy 做检测（作为 Lower Bound）。
  2. **FedAvg + Local ViM**: 客户端仅利用本地数据计算 $P$ 进行检测（验证全局共识的必要性）。
  3. **FedProx / Scaffold**: 加上 ViM 推理（验证 GSA Loss 是否比单纯的优化器修正更有效）。
  4. **FOOGD**: 相比之下，Fed-ViM 应具有训练速度快、收敛稳的优势 14。
- **评价指标**:
  - 检测：AUROC, FPR95 15。
  - 泛化：Accuracy on CIFAR-10-C。

