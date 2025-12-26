import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import copy
import os

# ==========================================
# Part A: Non-IID 数据划分 (Data Engine)
# ==========================================

def dirichlet_partition(dataset, num_clients, alpha=0.5, seed=42):
    """
    使用 Dirichlet 分布划分数据，模拟 Non-IID 场景。
    Alpha 越小，Non-IID 程度越严重。
    """
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = 10

    # 记录每个客户端拥有的索引
    client_indices = {i: [] for i in range(num_clients)}

    for k in range(num_classes):
        # 获取属于第 k 类的所有样本索引
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)

        # 从 Dirichlet 分布采样比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # 根据比例分割该类样本
        # 计算分割点
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        split_idx = np.split(idx_k, proportions)

        for client_id in range(num_clients):
            if client_id < len(split_idx):
                client_indices[client_id].append(split_idx[client_id])

    # 展平并打乱每个客户端的索引
    for client_id in range(num_clients):
        if len(client_indices[client_id]) > 0:
            client_indices[client_id] = np.concatenate(client_indices[client_id]).astype(int)
            np.random.shuffle(client_indices[client_id])
        else:
            client_indices[client_id] = np.array([], dtype=int)

    return client_indices

# ==========================================
# Part C: 核心架构 (Core Framework)
# ==========================================

# 1. 网络结构 (稍微加强版 SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, feature_dim=128): # 增加维度以便观察
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, feature_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def forward(self, x, return_feature=False):
        z = self.encoder(x)
        logits = self.head(z)
        if return_feature:
            return logits, z
        return logits

# 2. GSA Loss (复用你验证过的模块)
class GSALoss(nn.Module):
    def forward(self, features, P_global, mu_global):
        P = P_global.detach()
        mu = mu_global.detach()
        z_centered = features - mu
        # 投影计算
        z_proj = torch.matmul(z_centered, P)
        z_recon = torch.matmul(z_proj, P.T)
        diff = z_centered - z_recon
        return torch.norm(diff, p=2, dim=1).mean()

# 3. 客户端 (Client)
class Client:
    def __init__(self, client_id, train_set, idxs, device='cpu'):
        self.id = client_id
        self.loader = DataLoader(Subset(train_set, idxs), batch_size=64, shuffle=True)
        self.data_size = len(idxs)
        self.device = device

    def local_train(self, global_weights, global_stats, args):
        model = SimpleCNN(feature_dim=args['feature_dim']).to(self.device)
        model.load_state_dict(global_weights)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_gsa = GSALoss()

        P_global, mu_global = global_stats['P'], global_stats['mu']

        # 训练循环
        model.train()
        for epoch in range(args['local_epochs']):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, features = model(x, return_feature=True)

                loss = criterion_ce(logits, y)
                if P_global is not None:
                    loss += args['lambda_gsa'] * criterion_gsa(features, P_global, mu_global)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 计算统计量 (ViM Requirements)
        model.eval()
        sum_z = torch.zeros(args['feature_dim']).to(self.device)
        sum_zzT = torch.zeros(args['feature_dim'], args['feature_dim']).to(self.device)

        # 用于计算 Alpha 的统计量
        sum_max_logit = 0.0
        sum_residual = 0.0
        count = 0

        with torch.no_grad():
            for x, _ in self.loader:
                x = x.to(self.device)
                logits, features = model(x, return_feature=True)

                # 显存优化: features.T @ features
                sum_z += features.sum(dim=0)
                sum_zzT += torch.matmul(features.T, features)

                # 如果有全局参数，计算 ViM 相关的 Logit 和 Residual
                if P_global is not None and mu_global is not None:
                    # 1. Max Logit
                    max_logits, _ = torch.max(logits, dim=1)
                    sum_max_logit += max_logits.sum().item()

                    # 2. Residual
                    z_centered = features - mu_global
                    z_proj = torch.matmul(z_centered, P_global)
                    z_recon = torch.matmul(z_proj, P_global.T)
                    residuals = torch.norm(z_centered - z_recon, p=2, dim=1)
                    sum_residual += residuals.sum().item()

                count += x.size(0)

        return model.state_dict(), (sum_z, sum_zzT), (sum_max_logit, sum_residual, count)

# 4. 服务端 (Server)
class Server:
    def __init__(self, feature_dim, device='cpu'):
        self.model = SimpleCNN(feature_dim=feature_dim).to(device)
        self.feature_dim = feature_dim
        self.device = device
        self.P_global = None
        self.mu_global = torch.zeros(feature_dim).to(device)
        self.alpha = 1.0 # ViM 缩放系数

    def aggregate(self, local_weights, local_stats, local_vim_stats, sizes):
        total = sum(sizes)

        # 1. 聚合权重
        w_avg = copy.deepcopy(local_weights[0])
        for key in w_avg.keys():
            w_avg[key] *= sizes[0]/total
            for i in range(1, len(local_weights)):
                w_avg[key] += local_weights[i][key] * (sizes[i]/total)
        self.model.load_state_dict(w_avg)

        # 2. 聚合统计量并更新 P (PCA)
        global_sum_z = sum([s[0] for s in local_stats])
        global_sum_zzT = sum([s[1] for s in local_stats])

        self.mu_global = global_sum_z / total
        E_zzT = global_sum_zzT / total
        Cov = E_zzT - torch.outer(self.mu_global, self.mu_global)

        # SVD 分解
        try:
            # 加上微小噪声防止数值不稳定
            Cov += torch.eye(self.feature_dim).to(self.device) * 1e-6
            vals, vecs = torch.linalg.eigh(Cov)
            # 取前 33% 的主成分
            k = int(self.feature_dim * 0.33)
            self.P_global = vecs[:, -k:]
        except Exception as e:
            print(f"PCA Error: {e}")

        # 3. 更新 Alpha (Logit/Residual 比例)
        total_logit = sum([v[0] for v in local_vim_stats])
        total_res = sum([v[1] for v in local_vim_stats])
        # 避免除以0 (第一轮 total_res 为 0)
        if total_res > 1e-5:
            self.alpha = total_logit / total_res
            print(f"[Server] Updated Alpha: {self.alpha:.4f}")

        return {'P': self.P_global, 'mu': self.mu_global}

# ==========================================
# Part B: 评估与可视化 (Evaluation & Visualization)
# ==========================================

def get_vim_score(model, x, P, mu, alpha, device):
    """计算单个 Batch 的 ViM Score"""
    x = x.to(device)
    with torch.no_grad():
        logits, features = model(x, return_feature=True)
        # 1. Logit Score (Energy or MaxLogit)
        s_logit = torch.max(logits, dim=1)[0] # Max Logit

        # 2. Residual Score
        z_centered = features - mu
        z_proj = torch.matmul(z_centered, P)
        z_recon = torch.matmul(z_proj, P.T)
        s_res = torch.norm(z_centered - z_recon, p=2, dim=1)

        # 3. Combine
        # 注意: 残差越大越是OOD，Logit越小越是OOD
        # Score = Logit - alpha * Residual
        # 这样 Score 越小，越可能是 OOD
        scores = s_logit - alpha * s_res
    return scores.cpu().numpy()

def evaluate_and_plot(server, test_loader, ood_loader, device):
    print("\n--- Starting Evaluation ---")
    model = server.model
    model.eval()
    P = server.P_global
    mu = server.mu_global
    alpha = server.alpha

    if P is None:
        print("Model not warmed up yet (No P). Skipping eval.")
        return

    id_scores = []
    ood_scores = []

    # 计算 ID 分数
    for x, _ in test_loader:
        s = get_vim_score(model, x, P, mu, alpha, device)
        id_scores.append(s)
    id_scores = np.concatenate(id_scores)

    # 计算 OOD 分数
    for x, _ in ood_loader:
        s = get_vim_score(model, x, P, mu, alpha, device)
        ood_scores.append(s)
    ood_scores = np.concatenate(ood_scores)

    # 计算 AUROC
    # Label: ID=1, OOD=0 (常规做法，或者反过来，取决于 AUROC 定义)
    # 这里我们定义: 检测 OOD 的能力。
    # 通常 OOD Detection 任务中，Label OOD=1, ID=0
    # ViM Score 越小越是 OOD。所以我们取负号 -Score 作为检测指标
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([-id_scores, -ood_scores]) # 取负，让OOD分数变高

    auroc = roc_auc_score(y_true, y_scores)
    print(f"AUROC: {auroc:.4f}")

    # 画图
    plt.figure(figsize=(10, 6))
    plt.hist(id_scores, bins=50, alpha=0.5, label='ID (CIFAR-10)', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD (SVHN)', density=True)
    plt.title(f'Fed-ViM Score Distribution - Real OOD (AUROC={auroc:.3f})')
    plt.xlabel('ViM Score (Logit - alpha * Residual)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fed_vim_result_svhn.png')
    print(f"Result saved to 'fed_vim_result_svhn.png'")

# ==========================================
# Main Execution
# ==========================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 准备数据
    print("Downloading CIFAR-10...")
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

    # Non-IID 划分
    client_indices = dirichlet_partition(train_data, num_clients=5, alpha=0.5)

    # 构建 Clients
    clients = [Client(i, train_data, idxs, device) for i, idxs in client_indices.items()]
    client_sizes = [c.data_size for c in clients]

    # 准备 DataLoader (Test & OOD)
    test_loader = DataLoader(test_data, batch_size=64)

    # 准备真实 OOD 数据 (SVHN)
    print("Downloading SVHN (OOD)...")
    svhn_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    ood_data = datasets.SVHN(root='./data', split='test', download=True, transform=svhn_transform)

    # 为了快速测试，只取前 2000 张
    ood_subset = Subset(ood_data, list(range(2000)))
    # SVHN 的 label 对我们不重要，统一定义为 0 (OOD)
    ood_loader = DataLoader([(x, 0) for x, _ in ood_subset], batch_size=64)

    # 2. 初始化服务端
    server = Server(feature_dim=128, device=device)
    global_stats = {'P': None, 'mu': None}

    # 3. 联邦训练循环
    rounds = 10 # 建议跑 10-20 轮看效果
    for r in range(rounds):
        print(f"\n--- Round {r+1}/{rounds} ---")

        local_weights = []
        local_stats = []
        local_vim_stats = []

        for client in clients:
            w, s, v_s = client.local_train(server.model.state_dict(), global_stats,
                                          {'local_epochs': 1, 'feature_dim': 128, 'lambda_gsa': 0.1})
            local_weights.append(w)
            local_stats.append(s)
            local_vim_stats.append(v_s)

        # 聚合
        print("Aggregating...")
        global_stats = server.aggregate(local_weights, local_stats, local_vim_stats, client_sizes)

    # 4. 最终评估
    evaluate_and_plot(server, test_loader, ood_loader, device)

if __name__ == "__main__":
    main()
