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


def plot_confusion_matrix(cm, class_names, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n混淆矩阵已保存: {output_path}")


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
        # FOOGD 模型 (使用 Energy)
        print("\n使用 Energy-based OOD 检测...")

        # 提取 ID 数据 logit
        id_logits = []
        id_labels = []

        with torch.no_grad():
            for images, labels in id_loader:
                images = images.to(device)
                logits, _ = model(images)
                id_logits.append(logits.cpu())
                id_labels.extend(labels.numpy())

        id_logits = torch.cat(id_logits, dim=0)

        # 提取 OOD 数据 logit
        ood_logits = []

        with torch.no_grad():
            for images, _ in ood_loader:
                images = images.to(device)
                logits, _ = model(images)
                ood_logits.append(logits.cpu())

        ood_logits = torch.cat(ood_logits, dim=0)

        # 计算 Energy 分数
        id_energy = compute_energy_scores(id_logits)
        ood_energy = compute_energy_scores(ood_logits)

        # Energy 越高越可能是 ID,越低越可能是 OOD
        # 但为了统一评估 (分数高=ID,分数低=OOD),我们取负号
        id_scores = -id_energy
        ood_scores = -ood_energy

        method_name = "Energy (LogSumExp)"

    else:
        # Fed-ViM 模型 (使用 ViM: Max Logit - Distance)
        print("\n使用 ViM (Max Logit - α·Distance) OOD 检测...")

        # 提取 ID 数据特征和 logit
        id_features = []
        id_logits = []
        id_labels = []

        with torch.no_grad():
            for images, labels in id_loader:
                images = images.to(device)
                logits, features = model(images)

                id_features.append(features.cpu())
                id_logits.append(logits.cpu())
                id_labels.extend(labels.numpy())

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

        # 计算 Max Logit
        id_max_logit = torch.max(id_logits, dim=1)[0].numpy()
        ood_max_logit = torch.max(ood_logits, dim=1)[0].numpy()

        # 计算特征均值
        feature_mean = id_features.mean(dim=0).numpy()

        # 计算距离 (欧氏距离)
        id_distances = np.linalg.norm(id_features.numpy() - feature_mean, axis=1)
        ood_distances = np.linalg.norm(ood_features.numpy() - feature_mean, axis=1)

        # 自动校准 alpha
        alpha = np.mean(id_max_logit) / (np.mean(id_distances) + 1e-8)
        print(f"  - 自动校准 Alpha: {alpha:.4f}")
        print(f"  - ID Max Logit 均值: {np.mean(id_max_logit):.4f}")
        print(f"  - ID Distance 均值: {np.mean(id_distances):.4f}")
        print(f"  - OOD Max Logit 均值: {np.mean(ood_max_logit):.4f}")
        print(f"  - OOD Distance 均值: {np.mean(ood_distances):.4f}")

        # ViM 分数 = Max Logit - alpha * Distance
        id_scores = id_max_logit - alpha * id_distances
        ood_scores = ood_max_logit - alpha * ood_distances

        method_name = "ViM (MaxLogit - α·Distance)"

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
    # 找到 TPR 最接近 0.95 的阈值
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

    # 保存混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)

    # 保存 OOD 分布图
    ood_plot_path = os.path.join(output_dir, f'ood_distribution.png')
    plot_ood_distribution(id_scores, ood_scores, method_name, ood_plot_path)

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
    print(f"  - 混淆矩阵: {cm_path}")
    print(f"  - OOD 分布图: {ood_plot_path}")
    print(f"  - 评估摘要: {summary_path}")


if __name__ == '__main__':
    main()
