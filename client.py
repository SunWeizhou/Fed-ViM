#!/usr/bin/env python3
"""
联邦学习客户端模块 (修正版)
包含: Sigmoid Warm-up, LR=0.001, 显存优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import collections
import numpy as np  # 必须导入 numpy
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

class FLClient:
    """联邦学习客户端"""

    def __init__(self, client_id, model, foogd_module, train_loader, device,
                 compute_aug_features=True, freeze_bn=True, base_lr=0.001, use_fedvim=False):
        self.client_id = client_id
        self.model = model
        self.foogd_module = foogd_module
        self.train_loader = train_loader
        self.device = device
        self.compute_aug_features = compute_aug_features
        self.freeze_bn = freeze_bn
        self.base_lr = base_lr  # 基础学习率，可根据 batch_size 调整
        self.use_fedvim = use_fedvim

        # 如果使用 Fed-ViM，初始化 GSA Loss
        if self.use_fedvim:
            from models import GSALoss
            self.gsa_criterion = GSALoss()

        # 在初始化时定义优化器 (只做一次)
        self.optimizer_main = torch.optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,  # 使用传入的基础学习率
            momentum=0.9,
            weight_decay=1e-5
        )

        # FOOGD 优化器
        if self.foogd_module:
            self.optimizer_foogd = torch.optim.Adam(
                self.foogd_module.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999)
            )

        self.lambda_ksd = 0.01
        self.lambda_sm = 0.1

        # 傅里叶增强参数
        self.use_fourier_aug = True
        self.fourier_beta = 0.4
        self.fourier_prob = 0.9
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        self.scaler = torch.amp.GradScaler('cuda')

    def _fourier_augmentation(self, images, beta=None):
        """向量化傅里叶增强"""
        if beta is None: beta = self.fourier_beta
        mean = self.mean.to(images.device).view(1, 3, 1, 1)
        std = self.std.to(images.device).view(1, 3, 1, 1)
        x_unnorm = images * std + mean

        batch_size = images.size(0)
        perm = torch.randperm(batch_size).to(images.device)
        target_x_unnorm = x_unnorm[perm]

        fft_x = torch.fft.fftn(x_unnorm, dim=(-2, -1))
        fft_target = torch.fft.fftn(target_x_unnorm, dim=(-2, -1))

        amp_x, pha_x = torch.abs(fft_x), torch.angle(fft_x)
        amp_target = torch.abs(fft_target)

        amp_new = (1.0 - beta) * amp_x + beta * amp_target
        fft_new = amp_new * torch.exp(1j * pha_x)

        x_aug_unnorm = torch.fft.ifftn(fft_new, dim=(-2, -1)).real
        x_aug_unnorm = torch.clamp(x_aug_unnorm, 0, 1)
        x_aug = (x_aug_unnorm - mean) / std
        return x_aug

    def _apply_hybrid_augmentation(self, images):
        if torch.rand(1).item() < self.fourier_prob:
            return self._fourier_augmentation(images, beta=self.fourier_beta)
        return images

    # [修正2] 接收 current_round 参数并实现 Sigmoid Warm-up
    def train_step(self, local_epochs=1, current_round=0, global_stats=None):
        # =================================================================
        # 【优化】: 不再每次重新创建优化器，只调整学习率
        # 这样可以保留优化器的状态（如 momentum），同时确保学习率正确
        # =================================================================
        # 3. 动态调整学习率 (可选，但推荐)
        # 这是一个小技巧：虽然不重置优化器，但我们要确保学习率是正确的
        # 如果你有 decay 逻辑，可以在这里重新赋值 lr
        target_lr = self.base_lr  # 使用基础学习率
        # 建议：如果 Batch=64, 这里可以尝试 0.01 或保持 0.001
        for param_group in self.optimizer_main.param_groups:
            param_group['lr'] = target_lr

        # 如果有FOOGD模块，也调整其学习率
        if self.foogd_module:
            for param_group in self.optimizer_foogd.param_groups:
                param_group['lr'] = 1e-3

        self.model.train()
        if self.foogd_module:
            self.foogd_module.train()

        # [修复] 强制冻结 BN 层 (如果 freeze_bn=True)
        if self.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        total_loss = 0.0
        total_samples = 0
        epoch_log = {'cls': 0.0, 'ksd': 0.0, 'sm': 0.0, 'gsa': 0.0}

        # --- 解析全局统计量 (Fed-ViM) ---
        P_global = global_stats.get('P') if global_stats else None
        mu_global = global_stats.get('mu') if global_stats else None

        # --- 智能动态权重 (Sigmoid Warm-up) ---
        if self.foogd_module:
            warm_up_center = 10
            slope = 0.5
            # 计算 alpha (0~1)
            alpha = 1 / (1 + np.exp(-slope * (current_round - warm_up_center)))

            target_lambda_ksd = 0.01
            target_lambda_sm = 0.1

            effective_lambda_ksd = target_lambda_ksd * alpha
            effective_lambda_sm = target_lambda_sm * alpha

            # 打印当前权重
            if current_round % 5 == 0 and current_round > 0:
                print(f"  [Auto-Weight] Round {current_round}: Alpha={alpha:.4f} | KSD_w={effective_lambda_ksd:.6f}")
        else:
            effective_lambda_ksd = 0.0
            effective_lambda_sm = 0.0
        # -----------------------------------

        for epoch in range(local_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                self.optimizer_main.zero_grad()
                if self.foogd_module:
                    self.optimizer_foogd.zero_grad()

                with torch.amp.autocast('cuda'):
                    data_aug = self._apply_hybrid_augmentation(data)
                    logits, features = self.model(data)
                    _, features_aug = self.model(data_aug)

                    # 归一化 (关键)
                    features_norm = F.normalize(features, p=2, dim=1)
                    features_aug_norm = F.normalize(features_aug, p=2, dim=1)

                    # =================================================
                    # Loss 计算逻辑
                    # =================================================

                    # --- 计算分类 Loss (CrossEntropy) ---
                    classification_loss = F.cross_entropy(logits, targets)

                    # [修复] 初始化所有损失变量
                    ksd_loss = torch.tensor(0.0, device=self.device)
                    ksd_loss_val = 0.0
                    sm_loss = torch.tensor(0.0, device=self.device)
                    sm_loss_val = 0.0
                    loss_for_foogd = torch.tensor(0.0, device=self.device)
                    gsa_loss = torch.tensor(0.0, device=self.device)
                    gsa_loss_val = 0.0

                    if self.foogd_module:
                        ksd_loss, sm_loss, _ = self.foogd_module(features_norm, features_aug_norm)
                        ksd_loss_val = ksd_loss.item()
                        sm_loss_val = sm_loss.item()
                        loss_for_foogd = sm_loss

                    # --- [修改] Fed-ViM GSA Loss ---
                    # 只有当启用了 Fed-ViM 且 服务端已经下发了 P (即非第一轮) 时才计算
                    if self.use_fedvim and P_global is not None:
                        # 1. 约束原始特征 (保证 ID 数据在子空间内)
                        loss_clean = self.gsa_criterion(features, P_global, mu_global)

                        # 2. 约束增强特征 (强迫模型对风格变化鲁棒，核心！)
                        loss_aug = self.gsa_criterion(features_aug, P_global, mu_global)

                        # 3. 总 GSA Loss (取平均)
                        gsa_loss = 0.5 * loss_clean + 0.5 * loss_aug
                        gsa_loss_val = gsa_loss.item()

                    # 应用动态权重
                    lambda_gsa = 1.0  # Fed-ViM 权重
                    loss_for_main = classification_loss + lambda_gsa * gsa_loss + effective_lambda_ksd * ksd_loss

                # [修正3] 删除 retain_graph=True，释放显存
                self.scaler.scale(loss_for_main).backward()
                self.scaler.unscale_(self.optimizer_main)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer_main)

                if self.foogd_module:
                    self.scaler.scale(loss_for_foogd).backward()
                    self.scaler.unscale_(self.optimizer_foogd)
                    torch.nn.utils.clip_grad_norm_(self.foogd_module.parameters(), 5.0)
                    self.scaler.step(self.optimizer_foogd)

                self.scaler.update()

                batch_size = data.size(0)
                # [修复] 统一使用标量值进行统计计算
                total_batch_loss = classification_loss.item() + lambda_gsa * gsa_loss_val + effective_lambda_ksd * ksd_loss_val
                total_loss += total_batch_loss * batch_size
                total_samples += batch_size
                epoch_log['cls'] += classification_loss.item() * batch_size
                epoch_log['ksd'] += ksd_loss_val * batch_size
                epoch_log['sm'] += sm_loss_val * batch_size
                epoch_log['gsa'] += gsa_loss_val * batch_size

        if total_samples > 0:
            # 计算平均损失并打印
            avg_cls = epoch_log['cls'] / total_samples
            avg_ksd = epoch_log['ksd'] / total_samples
            avg_sm = epoch_log['sm'] / total_samples
            avg_gsa = epoch_log['gsa'] / total_samples
            print(f"Client {self.client_id} - Epochs {local_epochs} - Avg Loss: {total_loss / total_samples:.4f} | "
                  f"Cls: {avg_cls:.4f}, KSD: {avg_ksd:.6f}, SM: {avg_sm:.6f}, GSA: {avg_gsa:.4f}")

        # --- [新增] 计算并返回统计量 ---
        client_vim_stats = None
        if self.use_fedvim:
            client_vim_stats = self._compute_local_statistics()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        generic_params = self.get_generic_parameters()
        # [修改] 返回值增加 client_vim_stats
        return generic_params, avg_loss, client_vim_stats

    def get_generic_parameters(self):
        params = {}
        model_state = self.model.state_dict()
        for key, value in model_state.items():
            params[f"model.{key}"] = value.clone()
        if self.foogd_module:
            foogd_state = self.foogd_module.state_dict()
            for key, value in foogd_state.items():
                params[f"foogd.{key}"] = value.clone()
        return params

    def set_generic_parameters(self, generic_params):
        model_params = {}
        foogd_params = {}
        for key, value in generic_params.items():
            if key.startswith("model."):
                model_params[key.replace("model.", "")] = value
            elif key.startswith("foogd."):
                foogd_params[key.replace("foogd.", "")] = value
        self.model.load_state_dict(model_params, strict=False)
        if self.foogd_module and foogd_params:
            self.foogd_module.load_state_dict(foogd_params, strict=False)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits, _ = self.model(data)
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item() * data.size(0)
                _, pred = torch.max(logits, 1)
                correct += (pred == targets).sum().item()
                total_samples += data.size(0)
        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'acc': correct / total_samples if total_samples > 0 else 0.0
        }

    def compute_ood_scores(self, data_loader):
        """
        计算OOD分数

        Args:
            data_loader: 数据加载器

        Returns:
            ood_scores: OOD分数列表
            labels: 真实标签列表
        """
        self.model.eval()
        if self.foogd_module:
            self.foogd_module.eval()

        all_ood_scores = []
        all_labels = []

        with torch.no_grad():
            for data, targets in data_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                _, features = self.model(data)

                if self.foogd_module:
                    # [修复] 必须先归一化，与训练时保持一致！
                    features_norm = F.normalize(features, p=2, dim=1)
                    _, _, ood_scores = self.foogd_module(features_norm)
                else:
                    # 如果没有FOOGD模块，使用特征范数作为OOD分数
                    ood_scores = torch.norm(features, dim=1)

                all_ood_scores.extend(ood_scores.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        return all_ood_scores, all_labels

    def _compute_local_statistics(self):
        """[新增] Fed-ViM: 计算本地二阶统计量 (显存优化版)"""
        self.model.eval()
        # 获取特征维度 (DenseNet121=1024, 169=1664)
        feature_dim = self.model.backbone.feature_dim

        sum_z = torch.zeros(feature_dim).to(self.device)
        sum_zzT = torch.zeros(feature_dim, feature_dim).to(self.device)
        count = 0

        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                _, features = self.model(data)  # 获取特征

                # 1. 累加一阶矩 sum(z)
                sum_z += features.sum(dim=0)

                # 2. 累加二阶矩 sum(zz^T)
                # 使用矩阵乘法技巧 features.T @ features 避免生成 (B, D, D) 的大张量
                sum_zzT += torch.matmul(features.T, features)

                count += features.size(0)

        return {'sum_z': sum_z, 'sum_zzT': sum_zzT, 'count': count}

