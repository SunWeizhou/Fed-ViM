#!/usr/bin/env python3
"""
模型评估脚本 - 用于评估训练好的 Fed-ViM 或 FOOGD 模型
"""

import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from models import FedAvg_Model, FOOGD_Module, DenseNetBackbone
from data_utils import create_test_loaders_only

# 设置非交互式后端
import matplotlib
matplotlib.use('Agg')


def load_checkpoint(checkpoint_path, device):
    """加载检查点"""
    print(f"正在加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

    # 提取关键信息
    round_num = checkpoint.get('round', 'Unknown')
    test_acc = checkpoint.get('test_acc', 0.0)

    print(f"  - Round: {round_num}")
    print(f"  - Test Accuracy: {test_acc:.2f}%")

    return checkpoint


def create_model_from_config(config, device):
    """根据配置创建模型"""
    model_type = config.get('model_type', 'densenet121')
    use_foogd = config.get('use_foogd', False)
    num_classes = 54  # 检查点显示模型是54类

    # 创建基础模型
    # 注意: 无论是否使用 FOOGD,检查点中都保存 FedAvg_Model 结构
    # FOOGD 模块是单独保存的
    backbone = DenseNetBackbone(model_type=model_type, pretrained=False)
    model = FedAvg_Model(backbone, num_classes=num_classes, hidden_dim=512)

    model = model.to(device)
    model.eval()

    return model


def evaluate_classification(model, test_loader, device, class_names=None):
    """评估分类性能"""
    print("\n" + "="*60)
    print("分类性能评估")
    print("="*60)

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = 100 * correct / total
    print(f"\n整体准确率: {accuracy:.2f}%")
    print(f"正确分类: {correct}/{total}")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 打印分类报告
    print("\n分类报告:")
    if class_names is None:
        class_names = [f"Class {i}" for i in range(26)]
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    return accuracy, cm


def plot_confusion_matrix(cm, class_names, output_path, normalize=True, mask_diagonal=False):
    """
    绘制科研级混淆矩阵 (针对多类别优化)

    Args:
        cm: 原始混淆矩阵 (numpy array)
        class_names: 类别名称列表
        output_path: 保存路径
        normalize: 是否按行归一化 (显示 Recall)
        mask_diagonal: 是否屏蔽对角线 (高亮显示错误)
    """
    # 1. 归一化处理
    if normalize:
        # 加上 epsilon 防止除零
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        title = 'Normalized Confusion Matrix (Recall)'
        fmt = '.2f' # 显示两位小数
    else:
        cm_norm = cm
        title = 'Confusion Matrix (Raw Counts)'
        fmt = 'd'

    # 2. 对角线屏蔽 (用于发现 Error Pattern)
    if mask_diagonal:
        # 创建一个掩码，只显示非对角线元素
        cm_plot = cm_norm.copy()
        np.fill_diagonal(cm_plot, 0)
        title += ' - Diagonal Masked (Errors Highlighted)'
        # vmax 设置为非对角线最大值的 1.2 倍，保证对比度
        vmax = np.max(cm_plot) * 1.2 if np.max(cm_plot) > 0 else 1.0
    else:
        cm_plot = cm_norm
        vmax = 1.0

    # 3. 绘图
    plt.figure(figsize=(24, 20))

    # 智能标注: 只有当数值 > 阈值时才显示数字，防止密密麻麻
    threshold = 0.01 if normalize else 0
    annot_array = cm_plot.copy()
    annot_array[annot_array <= threshold] = np.nan  # 小值不显示

    sns.heatmap(cm_plot, annot=annot_array if normalize else cm_plot,
                fmt=fmt, cmap='Reds',
                xticklabels=class_names, yticklabels=class_names,
                vmax=vmax, cbar_kws={'label': 'Recall / Probability'},
                linewidths=0.5, linecolor='gray')

    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=16, fontweight='bold')
    plt.title(title, fontsize=20, fontweight='bold')

    # 旋转轴标签，防止重叠
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n混淆矩阵已保存: {output_path}")
    if mask_diagonal:
        print(f"  提示: 对角线已屏蔽,红色高亮显示主要错误分类")


def compute_energy_scores(logits):
    """计算 Energy 分数 (LogSumExp)"""
    # Energy = -log(sum(exp(logits)))
    # 等价于: -LogSumExp
    return -torch.logsumexp(logits, dim=1).numpy()


def evaluate_ood(model, id_loader, ood_loader, device, use_energy=False):
    """评估 OOD 检测性能"""
    print("\n" + "="*60)
    print("OOD 检测性能评估")
    print("="*60)

    if use_energy:
        # FOOGD 模型 - 需要加载 FOOGD 模块
        print("\n警告: 当前评估脚本不支持 FOOGD 完整评估")
        print("建议查看训练日志中的 Near/Far AUROC 指标")

        # 临时使用 Energy 作为保底
        print("\n使用保底方法: Energy-based OOD 检测...")

        # 提取 ID 数据 logit
        id_logits = []
        with torch.no_grad():
            for images, _ in id_loader:
                images = images.to(device)
                logits, _ = model(images)
                id_logits.append(logits.cpu())

        id_logits = torch.cat(id_logits, dim=0)

        # 提取 OOD 数据 logit
        ood_logits = []
        with torch.no_grad():
            for images, _ in ood_loader:
                images = images.to(device)
                logits, _ = model(images)
                ood_logits.append(logits.cpu())

        ood_logits = torch.cat(ood_logits, dim=0)

        # 计算 Energy 分数 (负号: 低 energy = OOD)
        id_scores = -torch.logsumexp(id_logits, dim=1).numpy()
        ood_scores = -torch.logsumexp(ood_logits, dim=1).numpy()

        method_name = "Energy (保底方法 - 不准确)"

    else:
        # Fed-ViM 模型 - 使用正确的 ViM Score
        print("\n使用 Fed-ViM OOD 检测 (Residual - α·Energy)...")

        # 提取 ID 数据特征和 logit
        id_features = []
        id_logits = []

        with torch.no_grad():
            for images, labels in id_loader:
                images = images.to(device)
                logits, features = model(images)
                id_features.append(features.cpu())
                id_logits.append(logits.cpu())

        id_features = torch.cat(id_features, dim=0)
        id_logits = torch.cat(id_logits, dim=0)

        # 提取 OOD 数据特征
        ood_features = []
        ood_logits = []

        with torch.no_grad():
            for images, _ in ood_loader:
                images = images.to(device)
                logits, features = model(images)
                ood_features.append(features.cpu())
                ood_logits.append(logits.cpu())

        ood_features = torch.cat(ood_features, dim=0)
        ood_logits = torch.cat(ood_logits, dim=0)

        # 计算特征均值 (作为全局子空间中心)
        feature_mean = id_features.mean(dim=0)

        # 计算残差 (Residual)
        # Residual = ||(I - PP^T)(z - mu)||
        # 简化版本: 使用到特征均值的距离作为残差代理
        id_residual = torch.norm(id_features - feature_mean, p=2, dim=1).numpy()
        ood_residual = torch.norm(ood_features - feature_mean, p=2, dim=1).numpy()

        # 计算 Energy (LogSumExp)
        id_energy = torch.logsumexp(id_logits, dim=1).numpy()
        ood_energy = torch.logsumexp(ood_logits, dim=1).numpy()

        # 自动校准 alpha
        # alpha = mean(Residual) / mean(Energy)
        alpha = np.mean(id_residual) / (np.mean(id_energy) + 1e-8)
        print(f"  - 自动校准 Alpha: {alpha:.4f}")
        print(f"  - ID Residual 均值: {np.mean(id_residual):.4f}")
        print(f"  - ID Energy 均值: {np.mean(id_energy):.4f}")
        print(f"  - OOD Residual 均值: {np.mean(ood_residual):.4f}")
        print(f"  - OOD Energy 均值: {np.mean(ood_energy):.4f}")

        # ViM Score = Residual - alpha * Energy
        # 分数越高 → 越可能是 OOD
        id_scores = id_residual - alpha * id_energy
        ood_scores = ood_residual - alpha * ood_energy

        method_name = "ViM (Residual - α·Energy)"

    # 评估 OOD 检测
    id_labels_numpy = np.zeros(len(id_scores))
    ood_labels_numpy = np.ones(len(ood_scores))

    all_scores = np.concatenate([id_scores, ood_scores])
    all_labels = np.concatenate([id_labels_numpy, ood_labels_numpy])

    # 计算 AUROC
    auroc = roc_auc_score(all_labels, all_scores)

    # 计算 AUPR
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    aupr = auc(recall, precision)

    # 计算 FPR95 (TPR=0.95 时的 FPR)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    print(f"\n{method_name} OOD 检测结果:")
    print(f"  - AUROC: {auroc:.4f}")
    print(f"  - AUPR:  {aupr:.4f}")
    print(f"  - FPR95: {fpr95:.4f}")

    results = {
        'method': method_name,
        'auroc': auroc,
        'aupr': aupr,
        'fpr95': fpr95,
        'id_scores_mean': np.mean(id_scores),
        'id_scores_std': np.std(id_scores),
        'ood_scores_mean': np.mean(ood_scores),
        'ood_scores_std': np.std(ood_scores)
    }

    return results, (id_scores, ood_scores, method_name)


def roc_curve(labels, scores):
    """计算 ROC 曲线"""
    from sklearn.metrics import roc_curve as sklearn_roc_curve
    fpr, tpr, thresholds = sklearn_roc_curve(labels, scores)
    return fpr, tpr, thresholds


def plot_ood_distribution(id_scores, ood_scores, method_name, output_path):
    """绘制 OOD 分数分布"""
    plt.figure(figsize=(12, 5))

    # 分数分布
    plt.subplot(1, 2, 1)
    plt.hist(id_scores, bins=50, alpha=0.6, label='ID (In-Distribution)', color='blue', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.6, label='OOD (Out-of-Distribution)', color='red', density=True)
    plt.xlabel(f'{method_name} Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'ID vs OOD Score Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # 箱线图
    plt.subplot(1, 2, 2)
    data_to_plot = [id_scores, ood_scores]
    plt.boxplot(data_to_plot, labels=['ID', 'OOD'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel(f'{method_name} Score', fontsize=12)
    plt.title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"OOD 分布图已保存: {output_path}")


def plot_top_confusion_pairs(cm, class_names, output_path, top_k=20):
    """
    绘制最容易混淆的 Top-K 类别对

    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        output_path: 保存路径
        top_k: 显示前 K 个最易混淆的对
    """
    # 归一化
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    # 将矩阵展平为 (True, Pred, Value) 的列表
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:  # 忽略对角线
                confusions.append({
                    'True Class': class_names[i],
                    'Predicted Class': class_names[j],
                    'Error Rate': cm_norm[i, j],
                    'Raw Count': cm[i, j]
                })

    # 转为 DataFrame 并排序
    import pandas as pd
    df = pd.DataFrame(confusions)
    df = df.sort_values('Error Rate', ascending=False).head(top_k)

    # 绘图
    plt.figure(figsize=(14, 8))
    # 创建标签: "True -> Pred"
    df['Label'] = df.apply(lambda x: f"{x['True Class']} → {x['Predicted Class']}", axis=1)

    bars = plt.bar(range(len(df)), df['Error Rate'], color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.2)

    plt.ylabel('Error Rate (Recall Loss)', fontsize=13, fontweight='bold')
    plt.title(f'Top {top_k} Most Confused Class Pairs', fontsize=15, fontweight='bold')
    plt.xticks(range(len(df)), df['Label'].values, rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xlabel('True Class → Predicted Class', fontsize=12, fontweight='bold')

    # 在柱子上标数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Top-{top_k} 混淆对分析图已保存: {output_path}")


def plot_tsne_feature_space(model, id_loader, ood_loader, device, output_path, max_samples=2000):
    """
    绘制 ID vs OOD 的 t-SNE 特征分布图

    Args:
        model: 训练好的模型
        id_loader: ID 数据加载器
        ood_loader: OOD 数据加载器
        device: 计算设备
        output_path: 保存路径
        max_samples: 最大样本数 (防止计算过慢)
    """
    from sklearn.manifold import TSNE

    model.eval()
    features_list = []
    labels_list = []  # 0 for ID, 1 for OOD

    print("\n正在收集特征用于 t-SNE 可视化...")

    # 1. 收集 ID 特征
    with torch.no_grad():
        count = 0
        for data, _ in id_loader:
            data = data.to(device)
            _, feats = model(data)
            features_list.append(feats.cpu().numpy())
            labels_list.append(np.zeros(len(feats)))
            count += len(feats)
            if count > max_samples // 2:
                break

    # 2. 收集 OOD 特征
    with torch.no_grad():
        count = 0
        for data, _ in ood_loader:
            data = data.to(device)
            _, feats = model(data)
            features_list.append(feats.cpu().numpy())
            labels_list.append(np.ones(len(feats)))
            count += len(feats)
            if count > max_samples // 2:
                break

    X = np.concatenate(features_list)
    y = np.concatenate(labels_list)

    print(f"  - ID 样本: {np.sum(y==0)}")
    print(f"  - OOD 样本: {np.sum(y==1)}")
    print(f"  - 特征维度: {X.shape[1]}")

    # 3. t-SNE 降维
    print("\n正在计算 t-SNE (这可能需要几分钟)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto',
                perplexity=min(30, len(X) // 4))  # 自适应调整 perplexity
    X_embedded = tsne.fit_transform(X)

    print("  t-SNE 计算完成!")

    # 4. 绘图
    plt.figure(figsize=(12, 10))

    # ID 数据 (蓝色)
    id_mask = y == 0
    plt.scatter(X_embedded[id_mask, 0], X_embedded[id_mask, 1],
               c='blue', alpha=0.5, s=15, label='In-Distribution (ID)', edgecolors='none')

    # OOD 数据 (红色)
    ood_mask = y == 1
    plt.scatter(X_embedded[ood_mask, 0], X_embedded[ood_mask, 1],
               c='red', alpha=0.5, s=15, label='Out-of-Distribution (OOD)', edgecolors='none')

    plt.legend(fontsize=12, markerscale=2)
    plt.title('t-SNE Feature Space Visualization\n(ID vs OOD Separability)',
              fontsize=15, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.grid(alpha=0.2)

    # 添加说明
    plt.text(0.02, 0.98,
             f"ID (blue, n={np.sum(id_mask)}): {np.sum(y==0)} samples\n"
             f"OOD (red, n={np.sum(ood_mask)}): {np.sum(y==1)} samples\n"
             f"Feature dim: {X.shape[1]} → t-SNE (2D)",
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"t-SNE 特征空间可视化已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='检查点路径 (例如: experiments_fedvim_dense121_224/experiment_XXX/best_model.pth)')
    parser.add_argument('--data_root', type=str, default='./Plankton_OOD_Dataset',
                       help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='结果输出目录 (默认与检查点同目录)')
    parser.add_argument('--enable_tsne', action='store_true', default=False,
                       help='是否启用 t-SNE 特征空间可视化 (耗时较长)')
    parser.add_argument('--tsne_samples', type=int, default=2000,
                       help='t-SNE 使用的最大样本数 (默认: 2000)')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(checkpoint_dir, 'config.json')

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("\n实验配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print(f"警告: 未找到配置文件 {config_path}")
        config = {}

    # 加载检查点
    checkpoint = load_checkpoint(args.checkpoint, device)

    # 创建模型
    model = create_model_from_config(config, device)
    model.load_state_dict(checkpoint['global_model_state_dict'])

    # 加载数据
    print("\n加载数据集...")
    id_loader, ood_loader, _, num_classes = create_test_loaders_only(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=config.get('image_size', 224)
    )

    print(f"  - ID 类别数: {num_classes}")
    print(f"  - ID 批次数: {len(id_loader)}")
    print(f"  - OOD 批次数: {len(ood_loader)}")

    # 评估分类
    num_classes = 54  # 实际类别数
    class_names = [f"Class {i}" for i in range(num_classes)]
    accuracy, cm = evaluate_classification(model, id_loader, device, class_names)

    # 评估 OOD 检测
    ood_results, (id_scores, ood_scores, method_name) = evaluate_ood(
        model, id_loader, ood_loader, device,
        use_energy=config.get('use_foogd', False)
    )

    # 保存结果
    output_dir = args.output_dir if args.output_dir else checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("生成可视化结果")
    print("="*60)

    # 1. 保存标准混淆矩阵 (归一化)
    cm_path = os.path.join(output_dir, 'confusion_matrix_normalized.png')
    plot_confusion_matrix(cm, class_names, cm_path, normalize=True, mask_diagonal=False)

    # 2. 保存错误高亮混淆矩阵 (对角线屏蔽)
    cm_masked_path = os.path.join(output_dir, 'confusion_matrix_errors_highlighted.png')
    plot_confusion_matrix(cm, class_names, cm_masked_path, normalize=True, mask_diagonal=True)

    # 3. 保存 Top-K 混淆对分析
    topk_path = os.path.join(output_dir, 'top_confusion_pairs.png')
    plot_top_confusion_pairs(cm, class_names, topk_path, top_k=20)

    # 4. 保存 OOD 分布图
    ood_plot_path = os.path.join(output_dir, f'ood_distribution.png')
    plot_ood_distribution(id_scores, ood_scores, method_name, ood_plot_path)

    # 5. t-SNE 特征空间可视化 (可选,耗时较长)
    if args.enable_tsne:
        tsne_path = os.path.join(output_dir, 'tsne_feature_space.png')
        plot_tsne_feature_space(model, id_loader, ood_loader, device, tsne_path,
                               max_samples=args.tsne_samples)

    # 保存评估结果摘要
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("模型评估摘要\n")
        f.write("="*60 + "\n\n")
        f.write(f"检查点: {args.checkpoint}\n")
        f.write(f"数据集: Plankton ({num_classes} classes)\n\n")

        f.write("分类性能:\n")
        f.write(f"  - 准确率: {accuracy:.2f}%\n\n")

        f.write("OOD 检测性能:\n")
        f.write(f"  - 方法: {method_name}\n")
        f.write(f"  - AUROC: {ood_results['auroc']:.4f}\n")
        f.write(f"  - AUPR: {ood_results['aupr']:.4f}\n")
        f.write(f"  - FPR95: {ood_results['fpr95']:.4f}\n")
        f.write(f"  - ID 分数均值: {ood_results['id_scores_mean']:.4f} ± {ood_results['id_scores_std']:.4f}\n")
        f.write(f"  - OOD 分数均值: {ood_results['ood_scores_mean']:.4f} ± {ood_results['ood_scores_std']:.4f}\n")

    print(f"\n{'='*60}")
    print(f"评估完成!")
    print(f"{'='*60}")
    print(f"\n评估结果已保存到: {output_dir}")
    print(f"  1. 归一化混淆矩阵: {os.path.basename(cm_path)}")
    print(f"  2. 错误高亮混淆矩阵: {os.path.basename(cm_masked_path)}")
    print(f"  3. Top-20 混淆对分析: {os.path.basename(topk_path)}")
    print(f"  4. OOD 分数分布: {os.path.basename(ood_plot_path)}")
    if args.enable_tsne:
        print(f"  5. t-SNE 特征空间: {os.path.basename(tsne_path)}")
    print(f"  6. 评估摘要: {os.path.basename(summary_path)}")
    print(f"\n提示: 查看 '{os.path.basename(cm_masked_path)}' 可以快速定位主要错误分类")


if __name__ == '__main__':
    main()
