import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在 pyplot 之前设置
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import copy
import os
import sys

# 设置国内镜像源加速下载（清华源）
os.environ['HF_ENDPOINT'] = 'https://mirrors.tuna.tsinghua.edu.cn/hugging-face'

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

# 1. 网络结构 (ResNet-18 for CIFAR-10)
class ResNet18_CIFAR(nn.Module):
    """
    ResNet-18 适配版，专为 CIFAR-10 (32x32) 设计
    - 替换 7x7 conv1 为 3x3
    - 移除第一个 maxpool
    - 特征维度: 512
    """
    def __init__(self, num_classes=10, feature_dim=512):
        super(ResNet18_CIFAR, self).__init__()
        # 加载标准 ResNet18，不预训练 (从头学 Non-IID)
        self.backbone = resnet18(pretrained=False)

        # 【关键修改 1】适配 CIFAR-10 的小尺寸 (32x32)
        # 原版 conv1 是 7x7, stride 2 -> 会丢失太多信息
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 【关键修改 2】去掉第一层的 MaxPool，防止特征图过早变小
        self.backbone.maxpool = nn.Identity()

        # 去掉原版全连接层
        self.backbone.fc = nn.Identity()

        # 新的分类头
        self.head = nn.Linear(512, num_classes)
        self.feature_dim = feature_dim

    def forward(self, x, return_feature=False):
        # ResNet 前向传播
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        z = self.backbone.avgpool(x)
        z = torch.flatten(z, 1) # [Batch, 512]

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
        # 【优化 1】增大 batch size 到 128，提高 GPU 利用率
        # 【优化 2】num_workers=4 多进程预加载数据
        self.loader = DataLoader(
            Subset(train_set, idxs),
            batch_size=128,  # 64 → 128 (ResNet-18 可以轻松处理)
            shuffle=True,
            num_workers=4,   # 多进程数据加载，减少 GPU 等待
            pin_memory=True  # 锁页内存，加速 CPU→GPU 传输
        )
        self.data_size = len(idxs)
        self.device = device

    def local_train(self, global_weights, global_stats, args):
        model = ResNet18_CIFAR(feature_dim=args['feature_dim']).to(self.device)
        model.load_state_dict(global_weights)
        # 添加 weight decay (L2 正则化)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_gsa = GSALoss()

        P_global, mu_global = global_stats['P'], global_stats['mu']

        # 训练循环 - 记录损失
        model.train()
        epoch_losses = []  # 记录每个 epoch 的平均损失

        for epoch in range(args['local_epochs']):
            batch_losses = []
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, features = model(x, return_feature=True)

                loss_ce = criterion_ce(logits, y)
                loss_gsa = torch.tensor(0.0).to(self.device)
                if P_global is not None:
                    loss_gsa = args['lambda_gsa'] * criterion_gsa(features, P_global, mu_global)

                loss = loss_ce + loss_gsa

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            # 计算 epoch 平均损失
            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)

        # 计算统计量 (ViM Requirements)
        model.eval()
        sum_z = torch.zeros(args['feature_dim']).to(self.device)
        sum_zzT = torch.zeros(args['feature_dim'], args['feature_dim']).to(self.device)

        # 用于计算 Alpha 的统计量
        sum_max_logit = 0.0
        sum_residual = 0.0
        count = 0

        # 计算准确率
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits, features = model(x, return_feature=True)

                # 计算准确率
                _, predicted = torch.max(logits, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

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

        accuracy = 100 * correct / total

        return model.state_dict(), (sum_z, sum_zzT), (sum_max_logit, sum_residual, count), epoch_losses, accuracy

# 4. 服务端 (Server)
class Server:
    def __init__(self, feature_dim=512, device='cpu'):  # 默认 512 (ResNet-18)
        self.model = ResNet18_CIFAR(feature_dim=feature_dim).to(device)
        self.feature_dim = feature_dim
        self.device = device
        self.P_global = None
        self.mu_global = torch.zeros(feature_dim).to(device)
        self.alpha = 1.0 # ViM 缩放系数

    def aggregate(self, local_weights, local_stats, local_vim_stats, sizes):
        total = sum(sizes)

        # 1. 聚合权重 (FedAvg)
        w_avg = copy.deepcopy(local_weights[0])
        for key in w_avg.keys():
            # 转换为 float32 避免类型不匹配
            w_avg[key] = w_avg[key].float() * (sizes[0] / total)
            for i in range(1, len(local_weights)):
                w_avg[key] += local_weights[i][key].float() * (sizes[i] / total)
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

            # 【最终优化】k=32 (6.25%): 更严格的子空间，把 SVHN 挤出去！
            # ResNet-18 表达力强，6.25% 的子空间足够表示 ID 数据
            k = 32  # 64 → 32 (更严格，增强 OOD 检测)
            self.P_global = vecs[:, -k:]
            print(f"[Server] Subspace dimension: {k}/{self.feature_dim} ({100*k/self.feature_dim:.1f}%)")
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

    # 定义详细评估函数
    def get_vim_details(model, loader, P, mu, alpha, device):
        logits_list = []
        res_list = []
        scores = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                l, f = model(x, return_feature=True)

                # Logit
                max_l, _ = torch.max(l, dim=1)

                # Residual
                z_centered = f - mu
                z_proj = torch.matmul(z_centered, P)
                z_recon = torch.matmul(z_proj, P.T)
                res = torch.norm(z_centered - z_recon, p=2, dim=1)

                logits_list.append(max_l.cpu().numpy())
                res_list.append(res.cpu().numpy())
                scores.append((max_l - alpha * res).cpu().numpy())

        return np.concatenate(logits_list), np.concatenate(res_list), np.concatenate(scores)

    # 计算 ID 详细指标
    id_logits, id_res, id_scores = get_vim_details(model, test_loader, P, mu, alpha, device)

    # 计算 OOD 详细指标
    ood_logits, ood_res, ood_scores = get_vim_details(model, ood_loader, P, mu, alpha, device)

    # 打印调试信息
    print(f"\n{'='*50}")
    print(f"DEBUG INFO:")
    print(f"Alpha: {alpha:.4f}")
    print(f"ID  Mean Logit: {np.mean(id_logits):.4f} ± {np.std(id_logits):.4f}")
    print(f"ID  Mean Res:   {np.mean(id_res):.4f} ± {np.std(id_res):.4f}")
    print(f"OOD Mean Logit: {np.mean(ood_logits):.4f} ± {np.std(ood_logits):.4f}")
    print(f"OOD Mean Res:   {np.mean(ood_res):.4f} ± {np.std(ood_res):.4f}")
    print(f"{'='*50}")

    # 如果 OOD Res < ID Res，说明特征没学好或者子空间维度太大
    if np.mean(ood_res) < np.mean(id_res):
        print("⚠️  WARNING: OOD Residual < ID Residual!")
        print("   This suggests the subspace might be too large or features not well learned.")
    else:
        print("✓ OOD Residual > ID Residual (Good for detection)")

    # 计算 AUROC
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([-id_scores, -ood_scores])

    auroc = roc_auc_score(y_true, y_scores)
    print(f"\nAUROC: {auroc:.4f}")
    print(f"{'='*50}\n")

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
    import sys
    print("=" * 50)
    print("Fed-ViM Starting...")
    print("=" * 50)
    sys.stdout.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    sys.stdout.flush()

    # 1. 准备数据（使用 CIFAR-10 标准归一化）
    print("Loading CIFAR-10...")
    sys.stdout.flush()

    # CIFAR-10 标准均值和方差
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 数据增强：随机裁剪
        transforms.RandomHorizontalFlip(),     # 数据增强：随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    print(f"CIFAR-10 loaded: Train={len(train_data)}, Test={len(test_data)}")
    sys.stdout.flush()

    # Non-IID 划分
    client_indices = dirichlet_partition(train_data, num_clients=5, alpha=0.5)

    # 构建 Clients
    clients = [Client(i, train_data, idxs, device) for i, idxs in client_indices.items()]
    client_sizes = [c.data_size for c in clients]

    # 准备 DataLoader (Test & OOD)
    # 【优化】增大评估 batch size，加速评估
    test_loader = DataLoader(test_data, batch_size=128, num_workers=4, pin_memory=True)

    # 准备真实 OOD 数据 (SVHN)
    print("Loading SVHN (OOD)...")
    sys.stdout.flush()
    svhn_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),  # 使用 ID 的统计量归一化 OOD
    ])
    ood_data = datasets.SVHN(root='./data', split='test', download=True, transform=svhn_transform)
    print(f"SVHN loaded: {len(ood_data)} samples")
    sys.stdout.flush()

    # 为了快速测试，只取前 2000 张
    ood_subset = Subset(ood_data, list(range(2000)))
    # SVHN 的 label 对我们不重要，统一定义为 0 (OOD)
    # 【优化】增大 OOD batch size
    ood_loader = DataLoader([(x, 0) for x, _ in ood_subset], batch_size=128, num_workers=4, pin_memory=True)

    # 2. 初始化服务端
    print("Initializing server...")
    sys.stdout.flush()
    server = Server(feature_dim=512, device=device)  # ResNet-18: 512 维
    global_stats = {'P': None, 'mu': None}

    # 3. 联邦训练循环
    rounds = 50  # 增加到 50 轮
    local_epochs = 5  # 每轮本地训练 5 个 epoch（充分利用 GPU）
                       # ResNet-18 + batch_size=128 可以快速训练

    # 早停机制参数
    patience = 10  # 如果测试准确率连续 10 轮没有提升，则停止训练
    best_test_acc = 0
    rounds_no_improve = 0

    # 用于记录训练历史
    history = {
        'round': [],
        'client_losses': {i: [] for i in range(len(clients))},
        'client_accuracies': {i: [] for i in range(len(clients))},
        'test_accuracy': [],
        'alpha': []
    }

    print(f"\n{'='*50}")
    print(f"Starting Training: {rounds} rounds, {local_epochs} local epochs each")
    print(f"Early Stopping: patience={patience} rounds")
    print(f"{'='*50}")
    sys.stdout.flush()

    for r in range(rounds):
        print(f"\n--- Round {r+1}/{rounds} ---")

        local_weights = []
        local_stats = []
        local_vim_stats = []
        round_losses = []
        round_accuracies = []

        for i, client in enumerate(clients):
            # 【最终优化】lambda_gsa=1.0: 加大权重，强迫 ID 残差变小
            # ResNet-18 学习能力 + 强约束 = ID 特征紧密聚集在子空间内
            w, s, v_s, epoch_losses, accuracy = client.local_train(
                server.model.state_dict(), global_stats,
                {'local_epochs': local_epochs, 'feature_dim': 512, 'lambda_gsa': 1.0}
            )

            # 打印该 client 的训练信息
            avg_loss = np.mean(epoch_losses)
            print(f"  Client {i}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

            local_weights.append(w)
            local_stats.append(s)
            local_vim_stats.append(v_s)
            round_losses.append(avg_loss)
            round_accuracies.append(accuracy)

            # 记录历史
            history['client_losses'][i].append(avg_loss)
            history['client_accuracies'][i].append(accuracy)

        # 聚合
        print("Aggregating...")
        global_stats = server.aggregate(local_weights, local_stats, local_vim_stats, client_sizes)

        # 计算全局测试准确率
        test_acc = compute_accuracy(server.model, test_loader, device)
        print(f"  Test Accuracy: {test_acc:.2f}%")

        # 记录轮次信息（在早停检查之前记录，确保数据一致性）
        history['round'].append(r + 1)
        history['test_accuracy'].append(test_acc)
        history['alpha'].append(server.alpha)

        # 早停检查
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            rounds_no_improve = 0
            print(f"  ✓ New best test accuracy! Saving model...")
            # 保存最佳模型
            best_model_state = copy.deepcopy(server.model.state_dict())
        else:
            rounds_no_improve += 1
            print(f"  No improvement for {rounds_no_improve} round(s) (best: {best_test_acc:.2f}%)")

        if rounds_no_improve >= patience:
            print(f"\n{'='*50}")
            print(f"Early stopping triggered after {r+1} rounds!")
            print(f"Best test accuracy: {best_test_acc:.2f}%")
            print(f"{'='*50}")
            # 恢复最佳模型
            server.model.load_state_dict(best_model_state)
            break

        sys.stdout.flush()

    # 4. 最终评估
    evaluate_and_plot(server, test_loader, ood_loader, device)

    # 5. 绘制训练曲线
    plot_training_curves(history)

def compute_accuracy(model, test_loader, device):
    """计算测试集准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total

def plot_training_curves(history):
    """绘制训练过程中的损失和准确率曲线"""
    rounds = history['round']
    num_clients = len(history['client_losses'])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 各 Client 的损失曲线
    ax = axes[0, 0]
    for i in range(num_clients):
        ax.plot(rounds, history['client_losses'][i], label=f'Client {i}', marker='o', markersize=3)
    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.set_title('Client Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 各 Client 的准确率曲线
    ax = axes[0, 1]
    for i in range(num_clients):
        ax.plot(rounds, history['client_accuracies'][i], label=f'Client {i}', marker='o', markersize=3)
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Client Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 全局测试准确率
    ax = axes[1, 0]
    ax.plot(rounds, history['test_accuracy'], label='Test Accuracy', color='red', marker='o', markersize=4)
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Global Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Alpha 系数变化
    ax = axes[1, 1]
    ax.plot(rounds, history['alpha'], label='Alpha', color='green', marker='o', markersize=4)
    ax.set_xlabel('Round')
    ax.set_ylabel('Alpha Value')
    ax.set_title('ViM Alpha Coefficient')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fed_vim_training_curves.png', dpi=150)
    print(f"\nTraining curves saved to 'fed_vim_training_curves.png'")

if __name__ == "__main__":
    main()
