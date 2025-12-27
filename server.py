import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any

class FLServer:
    # 修改初始化，接收 foogd_module
    def __init__(self, global_model, foogd_module, device):
        self.global_model = global_model
        self.foogd_module = foogd_module # 新增
        self.device = device
        self.global_model.to(device)
        if self.foogd_module:
            self.foogd_module.to(device)

        # [新增] 存储全局统计量 (Fed-ViM)
        self.P_global = None
        self.mu_global = None

    def get_global_parameters(self):
        """获取全局参数 (Model + FOOGD) - 修复版"""
        # [修复] 返回带前缀的参数，与 client.set_generic_parameters 保持一致
        params = {}
        model_state = self.global_model.state_dict()
        for key, value in model_state.items():
            params[f"model.{key}"] = value.clone()
        if self.foogd_module:
            foogd_state = self.foogd_module.state_dict()
            for key, value in foogd_state.items():
                params[f"foogd.{key}"] = value.clone()
        return params

    def set_global_parameters(self, params):
        """设置全局参数 - 修复版：支持带前缀的参数拆解"""

        # 1. 初始化两个空字典
        model_params = {}
        foogd_params = {}

        # 2. 遍历聚合后的参数，根据前缀拆分
        for key, value in params.items():
            if key.startswith("model."):
                # 去掉 "model." 前缀，还原为模型能识别的键名
                new_key = key.replace("model.", "")
                model_params[new_key] = value
            elif key.startswith("foogd."):
                # 去掉 "foogd." 前缀
                new_key = key.replace("foogd.", "")
                foogd_params[new_key] = value
            else:
                # 兼容旧逻辑（如果某个参数没有前缀，尝试直接归类给model）
                model_params[key] = value

        # 3. 加载主模型参数
        # strict=False 仍然很重要，以防有些缓冲变量不匹配，但核心权重现在能对上了
        self.global_model.load_state_dict(model_params, strict=False)

        # 4. 加载 FOOGD 参数 [关键修复点]
        if self.foogd_module and foogd_params:
            self.foogd_module.load_state_dict(foogd_params, strict=False)

        print(f"  [Server] Updated Global Model with {len(model_params)} params")
        if self.foogd_module:
             print(f"  [Server] Updated FOOGD Module with {len(foogd_params)} params")

    def aggregate(self, updates, sample_sizes):
        """
        聚合函数 - [已修正] 恢复 BN 统计量的聚合
        """
        total_samples = sum(sample_sizes)
        # 1. 初始化聚合参数为第一个客户端的参数副本
        aggregated_params = copy.deepcopy(updates[0])

        for key in aggregated_params.keys():
            # [关键修改]：
            # 我们只跳过 'num_batches_tracked' (它是整数，记录训练步数，不需要平均)
            # 删除了之前过滤 'running_mean' 和 'running_var' 的逻辑
            if 'num_batches_tracked' in key:
                continue

            # 2. 准备进行加权平均
            # 先将当前参数置为 0
            # 注意：这里我们假设参数都是 Tensor 类型
            aggregated_params[key] = torch.zeros_like(aggregated_params[key], dtype=torch.float)

            for update, n_samples in zip(updates, sample_sizes):
                weight = n_samples / total_samples

                param_data = update[key]

                # 确保参与计算的数据是 float 类型
                if param_data.dtype != torch.float:
                    param_data = param_data.float()

                # 加权累加
                aggregated_params[key] += param_data * weight

        return aggregated_params

    # [新增] 核心 PCA 更新逻辑 (Fed-ViM)
    def update_global_subspace(self, client_stats_list, k=64):
        """
        基于充分统计量重构全局协方差并提取子空间
        Args:
            client_stats_list: 客户端上传的统计量列表
            k: 子空间保留维度 (建议 32-64)
        """
        if not client_stats_list:
            return {'P': None, 'mu': None}

        print("  [Server] Updating Global Subspace (PCA)...")

        # 1. 聚合统计量
        total_count = sum([s['count'] for s in client_stats_list])
        if total_count == 0: return {'P': None, 'mu': None}

        feature_dim = client_stats_list[0]['sum_z'].shape[0]
        global_sum_z = torch.zeros(feature_dim).to(self.device)
        global_sum_zzT = torch.zeros(feature_dim, feature_dim).to(self.device)

        for s in client_stats_list:
            global_sum_z += s['sum_z']
            global_sum_zzT += s['sum_zzT']

        # 2. 重构均值和协方差
        # mu = E[Z]
        self.mu_global = global_sum_z / total_count

        # E[ZZ^T]
        E_zzT = global_sum_zzT / total_count

        # Cov = E[ZZ^T] - mu * mu^T (利用方差分解公式)
        cov_global = E_zzT - torch.outer(self.mu_global, self.mu_global)

        # 3. 特征分解 (SVD)
        # 添加微小扰动防止数值不稳定
        cov_global += torch.eye(feature_dim).to(self.device) * 1e-6

        try:
            # torch.linalg.eigh 适用于对称矩阵，比 svd 快且稳
            eig_vals, eig_vecs = torch.linalg.eigh(cov_global)

            # eigh 返回的是特征值升序排列，我们需要最大的前 k 个
            # 取最后 k 个 -> 反转顺序
            self.P_global = eig_vecs[:, -k:]

            print(f"  [Server] PCA Done. Subspace shape: {self.P_global.shape}")

        except Exception as e:
            print(f"  [Server] PCA Failed: {e}")

        return {'P': self.P_global, 'mu': self.mu_global}

    def _compute_scores_and_metrics(self, data_loader, vim_stats=None):
        """
        [优化] 一次性计算 Loss, Accuracy 和 OOD Score
        避免重复前向传播

        Args:
            data_loader: 数据加载器
            vim_stats: Fed-ViM 全局统计信息，包含 'P' (子空间投影矩阵) 和 'mu' (全局均值)
        """
        self.global_model.eval()
        if self.foogd_module:
            self.foogd_module.eval()

        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []
        all_scores = []

        import torch.nn.functional as F

        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                # 1. 前向传播 (Backbone + Heads)
                # 处理不同模型的返回值：FedAvg_Model返回2个值，FedRoD返回3个值
                model_output = self.global_model(data)
                if len(model_output) == 2:
                    logits_g, features = model_output
                else:  # len(model_output) == 3
                    logits_g, _, features = model_output

                # 2. 计算分类指标（仅当标签有效时）
                # OOD数据的标签为-1，跳过损失和准确率计算
                valid_mask = targets >= 0
                if valid_mask.any():
                    valid_targets = targets[valid_mask]
                    valid_logits = logits_g[valid_mask]

                    loss = F.cross_entropy(valid_logits, valid_targets)
                    total_loss += loss.item() * valid_targets.size(0)
                    total_samples += valid_targets.size(0)

                    _, preds = torch.max(valid_logits, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(valid_targets.cpu().numpy())

                # 3. 计算 OOD Score - 对所有样本都计算
                if self.foogd_module:
                    # FOOGD 逻辑
                    features_norm = F.normalize(features, p=2, dim=1)
                    _, _, scores = self.foogd_module(features_norm)

                elif vim_stats is not None and vim_stats['P'] is not None:
                    # Fed-ViM 评分逻辑 (完整版 ViM Score)
                    # 1. 获取 P 和 mu
                    P = vim_stats['P']
                    mu = vim_stats['mu']

                    # 2. 中心化
                    z_centered = features - mu

                    # 3. 计算残差 (Residual)
                    # z_recon = (z-mu) @ P @ P.T
                    z_proj = torch.matmul(z_centered, P)
                    z_recon = torch.matmul(z_proj, P.T)
                    residual = torch.norm(z_centered - z_recon, p=2, dim=1)

                    # 4. 计算 Logit 项 (Max Logit 作为 ID 置信度)
                    max_logit, _ = torch.max(logits_g, dim=1)

                    # 5. 组合 ViM Score
                    # OOD Score = Residual - alpha * Max Logit
                    # - Residual 越大 → 偏离全局子空间 → 越可能是 OOD
                    # - Max Logit 越小 → 模型置信度低 → 越可能是 OOD
                    # alpha = 1.0 作为平衡系数（Residual 和 Logit 通常在相同数量级，约 10.0）
                    alpha = 1.0
                    scores = residual - alpha * max_logit

                else:
                    # 保底逻辑 (特征范数)
                    scores = torch.norm(features, dim=1)

                all_scores.extend(scores.cpu().numpy())

        # 整理结果
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # 处理空数组情况
        if len(all_preds) > 0 and len(all_targets) > 0:
            accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        else:
            accuracy = 0.0  # 没有有效标签时，准确率为0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'scores': np.array(all_scores),
            'targets': np.array(all_targets) # 保留targets以防未来需要
        }

    def evaluate_global_model(self, test_loader, near_ood_loader, far_ood_loader, inc_loader=None, vim_stats=None):
        """
        [优化版] 评估全局模型性能 - 消除冗余计算

        Args:
            test_loader: 测试数据加载器 (ID 数据)
            near_ood_loader: Near-OOD 数据加载器
            far_ood_loader: Far-OOD 数据加载器
            inc_loader: IN-C 数据加载器 (可选)
            vim_stats: Fed-ViM 全局统计信息 (可选)
        """
        metrics = {}
        print("  正在评估 Global Model (优化版)...")

        # 1. 计算 ID (Clean) 的所有指标 [只跑一次!]
        # 这包含了 Accuracy, Loss, 和用于 OOD 对比的 ID Scores
        id_results = self._compute_scores_and_metrics(test_loader, vim_stats)

        metrics['id_accuracy'] = id_results['accuracy']
        metrics['id_loss'] = id_results['loss']
        id_scores = id_results['scores'] # 缓存下来，后面复用！

        print(f"    -> ID Acc: {metrics['id_accuracy']:.4f}")

        # 2. 评估 IN-C (如果存在)
        if inc_loader:
            inc_results = self._compute_scores_and_metrics(inc_loader, vim_stats)
            metrics['inc_accuracy'] = inc_results['accuracy']
            print(f"    -> IN-C Acc: {metrics['inc_accuracy']:.4f}")

        # 3. 评估 OOD 检测 (复用 id_scores)
        from sklearn.metrics import roc_auc_score

        # 辅助函数：计算 AUROC
        def compute_auroc(id_s, ood_s):
            y_true = np.concatenate([np.zeros_like(id_s), np.ones_like(ood_s)])
            y_scores = np.concatenate([id_s, ood_s])
            return roc_auc_score(y_true, y_scores)

        # Near-OOD
        if near_ood_loader:
            # 只需跑 Near-OOD 的前向传播
            near_results = self._compute_scores_and_metrics(near_ood_loader, vim_stats)
            near_scores = near_results['scores']
            metrics['near_auroc'] = compute_auroc(id_scores, near_scores)
            print(f"    -> Near AUROC: {metrics['near_auroc']:.4f}")

        # Far-OOD
        if far_ood_loader:
            # 只需跑 Far-OOD 的前向传播
            far_results = self._compute_scores_and_metrics(far_ood_loader, vim_stats)
            far_scores = far_results['scores']
            metrics['far_auroc'] = compute_auroc(id_scores, far_scores)
            print(f"    -> Far AUROC: {metrics['far_auroc']:.4f}")

        return metrics

if __name__ == "__main__":
    # 测试服务端
    print("测试联邦学习服务端...")

    # 创建模型
    from models import create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model, foogd_module = create_model()
    server = FLServer(model, foogd_module, device)

    # 测试参数获取和设置
    print("\n测试参数管理...")
    global_params = server.get_global_parameters()
    print(f"全局参数数量: {len(global_params)}")

    # 测试聚合
    print("\n测试参数聚合...")
    client_updates = []
    for _ in range(3):
        client_update = {}
        for name, param in global_params.items():
            # 只对浮点类型参数生成随机数
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                client_update[name] = torch.randn_like(param)
            else:
                # 对于整数类型参数，保持原值
                client_update[name] = param.clone()
        client_updates.append(client_update)
    client_sample_sizes = [100, 150, 200]

    aggregated_params = server.aggregate(client_updates, client_sample_sizes)
    print(f"聚合参数数量: {len(aggregated_params)}")

    # 测试评估
    print("\n测试模型评估...")
    from data_utils import create_federated_loaders

    data_root = "./data"
    # 注意：这里使用 try-except 块，以防没有数据时报错，仅作测试逻辑演示
    try:
        _, test_loader, near_ood_loader, far_ood_loader, _ = create_federated_loaders(
            data_root, n_clients=3, batch_size=4, image_size=224
        )

        metrics = server.evaluate_global_model(
            test_loader, near_ood_loader, far_ood_loader, None
        )

        print("评估指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"跳过数据评估测试（可能缺少数据）: {e}")

    print("服务端测试完成!")