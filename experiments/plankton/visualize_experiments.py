#!/usr/bin/env python3
"""
可视化实验结果的脚本
绘制损失曲线和模型性能指标
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（如果需要显示中文）
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'grid.alpha': 0.3
})

def load_experiment_data(experiment_dir):
    """加载实验数据"""
    config_path = os.path.join(experiment_dir, 'config.json')
    history_path = os.path.join(experiment_dir, 'training_history.json')

    if not os.path.exists(config_path) or not os.path.exists(history_path):
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    with open(history_path, 'r') as f:
        history = json.load(f)

    return {
        'config': config,
        'history': history,
        'name': os.path.basename(experiment_dir)
    }

def find_all_experiments(base_dir='./experiments'):
    """查找所有实验"""
    experiments = []

    # 查找所有包含training_history.json的文件夹
    history_files = glob.glob(os.path.join(base_dir, '**', 'training_history.json'), recursive=True)

    for history_file in history_files:
        experiment_dir = os.path.dirname(history_file)
        data = load_experiment_data(experiment_dir)
        if data:
            experiments.append(data)

    return experiments

def plot_training_losses(experiments, output_dir='./visualizations'):
    """Plot training loss curves"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 按alpha值分组
    alpha_groups = {}
    for exp in experiments:
        alpha = exp['config']['alpha']
        use_foogd = exp['config']['use_foogd']
        key = f"alpha={alpha}, FOOGD={'Enabled' if use_foogd else 'Disabled'}"

        if key not in alpha_groups:
            alpha_groups[key] = []
        alpha_groups[key].append(exp)

    # 绘制训练损失
    ax = axes[0]
    for key, exps in alpha_groups.items():
        # 取第一个实验（假设相同配置的实验结果相似）
        if exps:
            exp = exps[0]
            rounds = exp['history']['rounds']
            train_losses = exp['history']['train_losses']
            ax.plot(rounds, train_losses, label=key, marker='o', markersize=3)

    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True)

    # 绘制测试损失
    ax = axes[1]
    for key, exps in alpha_groups.items():
        if exps:
            exp = exps[0]
            rounds = exp['history']['rounds']
            test_losses = exp['history']['test_losses']
            ax.plot(rounds, test_losses, label=key, marker='s', markersize=3)

    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss Curves')
    ax.legend()
    ax.grid(True)

    # 绘制测试准确率
    ax = axes[2]
    for key, exps in alpha_groups.items():
        if exps:
            exp = exps[0]
            rounds = exp['history']['rounds']
            test_acc = exp['history']['test_accuracies']
            ax.plot(rounds, test_acc, label=key, marker='^', markersize=3)

    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Curves')
    ax.legend()
    ax.grid(True)

    # 绘制个人化准确率增益
    ax = axes[3]
    for key, exps in alpha_groups.items():
        if exps:
            exp = exps[0]
            rounds = exp['history']['rounds']
            person_acc = exp['history']['avg_person_acc']
            global_local_acc = exp['history']['avg_global_local_acc']

            # 计算个人化增益
            personalization_gain = [p - g for p, g in zip(person_acc, global_local_acc)]
            ax.plot(rounds, personalization_gain, label=key, marker='d', markersize=3)

    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Personalization Gain (Head_P - Head_G)')
    ax.set_title('Personalization Accuracy Gain Curves')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to: {os.path.join(output_dir, 'training_curves.png')}")

def plot_ood_performance(experiments, output_dir='./visualizations'):
    """Plot OOD detection performance"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 按alpha值分组
    alpha_groups = {}
    for exp in experiments:
        alpha = exp['config']['alpha']
        use_foogd = exp['config']['use_foogd']
        key = f"alpha={alpha}, FOOGD={'Enabled' if use_foogd else 'Disabled'}"

        if key not in alpha_groups:
            alpha_groups[key] = []
        alpha_groups[key].append(exp)

    # 绘制Near-OOD AUROC
    ax = axes[0]
    for key, exps in alpha_groups.items():
        if exps:
            exp = exps[0]
            rounds = exp['history']['rounds']
            near_auroc = exp['history']['near_auroc']
            ax.plot(rounds, near_auroc, label=key, marker='o', markersize=3)

    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('AUROC')
    ax.set_title('Near-OOD Detection Performance (AUROC)')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([0, 1])

    # 绘制Far-OOD AUROC
    ax = axes[1]
    for key, exps in alpha_groups.items():
        if exps:
            exp = exps[0]
            rounds = exp['history']['rounds']
            far_auroc = exp['history']['far_auroc']
            ax.plot(rounds, far_auroc, label=key, marker='s', markersize=3)

    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('AUROC')
    ax.set_title('Far-OOD Detection Performance (AUROC)')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([0, 1])

    # 绘制IN-C准确率
    ax = axes[2]
    for key, exps in alpha_groups.items():
        if exps:
            exp = exps[0]
            inc_acc = exp['history']['inc_accuracies']
            # IN-C准确率只有10个点（假设是10种扰动）
            rounds = list(range(1, len(inc_acc) + 1))
            ax.plot(rounds, inc_acc, label=key, marker='^', markersize=5)

    ax.set_xlabel('Corruption Type')
    ax.set_ylabel('Accuracy')
    ax.set_title('IN-C Robustness (10 Corruption Types)')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([0, 1])

    # 绘制最佳准确率对比
    ax = axes[3]
    alphas = []
    best_accs = []
    labels = []

    for key, exps in alpha_groups.items():
        if exps:
            exp = exps[0]
            alpha = exp['config']['alpha']
            use_foogd = exp['config']['use_foogd']
            best_acc = exp['history']['best_acc']

            alphas.append(alpha)
            best_accs.append(best_acc)
            labels.append(f"FOOGD={'Enabled' if use_foogd else 'Disabled'}")

    # 按alpha值排序
    sorted_indices = np.argsort(alphas)
    alphas = [alphas[i] for i in sorted_indices]
    best_accs = [best_accs[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    x = np.arange(len(alphas))
    ax.bar(x, best_accs, alpha=0.7)
    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Best Accuracy')
    ax.set_title('Best Accuracy Comparison Across Alpha Values')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n({l})" for a, l in zip(alphas, labels)])
    ax.grid(True, axis='y')

    # 在柱状图上添加数值
    for i, v in enumerate(best_accs):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ood_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"OOD performance chart saved to: {os.path.join(output_dir, 'ood_performance.png')}")

def plot_comparison_summary(experiments, output_dir='./visualizations'):
    """Plot comprehensive comparison summary"""
    os.makedirs(output_dir, exist_ok=True)

    # 提取关键指标
    metrics = []
    for exp in experiments:
        config = exp['config']
        history = exp['history']

        # 计算平均指标
        final_round = history['rounds'][-1]
        final_train_loss = history['train_losses'][-1]
        final_test_acc = history['test_accuracies'][-1]
        avg_near_auroc = np.mean(history['near_auroc'][-10:])  # 最后10轮的平均值
        avg_far_auroc = np.mean(history['far_auroc'][-10:])
        best_acc = history['best_acc']

        metrics.append({
            'alpha': config['alpha'],
            'use_foogd': config['use_foogd'],
            'final_train_loss': final_train_loss,
            'final_test_acc': final_test_acc,
            'avg_near_auroc': avg_near_auroc,
            'avg_far_auroc': avg_far_auroc,
            'best_acc': best_acc,
            'name': exp['name']
        })

    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 按是否使用FOOGD分组
    foogd_groups = {'Enabled': [], 'Disabled': []}
    for metric in metrics:
        key = 'Enabled' if metric['use_foogd'] else 'Disabled'
        foogd_groups[key].append(metric)

    # 1. 最终测试准确率对比
    ax = axes[0]
    for foogd_status, group_metrics in foogd_groups.items():
        if group_metrics:
            alphas = [m['alpha'] for m in group_metrics]
            test_accs = [m['final_test_acc'] for m in group_metrics]

            # 按alpha排序
            sorted_indices = np.argsort(alphas)
            alphas_sorted = [alphas[i] for i in sorted_indices]
            test_accs_sorted = [test_accs[i] for i in sorted_indices]

            ax.plot(alphas_sorted, test_accs_sorted, 'o-', label=f'FOOGD {foogd_status}',
                   markersize=8, linewidth=2)

    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Final Test Accuracy')
    ax.set_title('Final Test Accuracy Across Alpha Values')
    ax.legend()
    ax.grid(True)

    # 2. 最佳准确率对比
    ax = axes[1]
    for foogd_status, group_metrics in foogd_groups.items():
        if group_metrics:
            alphas = [m['alpha'] for m in group_metrics]
            best_accs = [m['best_acc'] for m in group_metrics]

            sorted_indices = np.argsort(alphas)
            alphas_sorted = [alphas[i] for i in sorted_indices]
            best_accs_sorted = [best_accs[i] for i in sorted_indices]

            ax.plot(alphas_sorted, best_accs_sorted, 's-', label=f'FOOGD {foogd_status}',
                   markersize=8, linewidth=2)

    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Best Test Accuracy Across Alpha Values')
    ax.legend()
    ax.grid(True)

    # 3. Near-OOD AUROC对比
    ax = axes[2]
    for foogd_status, group_metrics in foogd_groups.items():
        if group_metrics:
            alphas = [m['alpha'] for m in group_metrics]
            near_aurocs = [m['avg_near_auroc'] for m in group_metrics]

            sorted_indices = np.argsort(alphas)
            alphas_sorted = [alphas[i] for i in sorted_indices]
            near_aurocs_sorted = [near_aurocs[i] for i in sorted_indices]

            ax.plot(alphas_sorted, near_aurocs_sorted, '^-', label=f'FOOGD {foogd_status}',
                   markersize=8, linewidth=2)

    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Average Near-OOD AUROC')
    ax.set_title('Near-OOD Detection Performance Across Alpha Values')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([0, 1])

    # 4. Far-OOD AUROC对比
    ax = axes[3]
    for foogd_status, group_metrics in foogd_groups.items():
        if group_metrics:
            alphas = [m['alpha'] for m in group_metrics]
            far_aurocs = [m['avg_far_auroc'] for m in group_metrics]

            sorted_indices = np.argsort(alphas)
            alphas_sorted = [alphas[i] for i in sorted_indices]
            far_aurocs_sorted = [far_aurocs[i] for i in sorted_indices]

            ax.plot(alphas_sorted, far_aurocs_sorted, 'd-', label=f'FOOGD {foogd_status}',
                   markersize=8, linewidth=2)

    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Average Far-OOD AUROC')
    ax.set_title('Far-OOD Detection Performance Across Alpha Values')
    ax.legend()
    ax.grid(True)
    ax.set_ylim([0, 1])

    # 5. 最终训练损失对比
    ax = axes[4]
    for foogd_status, group_metrics in foogd_groups.items():
        if group_metrics:
            alphas = [m['alpha'] for m in group_metrics]
            train_losses = [m['final_train_loss'] for m in group_metrics]

            sorted_indices = np.argsort(alphas)
            alphas_sorted = [alphas[i] for i in sorted_indices]
            train_losses_sorted = [train_losses[i] for i in sorted_indices]

            ax.plot(alphas_sorted, train_losses_sorted, 'v-', label=f'FOOGD {foogd_status}',
                   markersize=8, linewidth=2)

    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Final Training Loss')
    ax.set_title('Final Training Loss Across Alpha Values')
    ax.legend()
    ax.grid(True)

    # 6. FOOGD效果对比（柱状图）
    ax = axes[5]
    if len(foogd_groups['Enabled']) > 0 and len(foogd_groups['Disabled']) > 0:
        # 这里需要匹配相同alpha值的实验进行对比
        # 由于当前数据只有FOOGD启用的实验，暂时显示说明
        ax.text(0.5, 0.5, 'Need paired experiments with\nFOOGD Enabled and Disabled\n\nCurrent data only has FOOGD Enabled',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('FOOGD Effect Comparison (Need Paired Data)')
    else:
        ax.text(0.5, 0.5, 'Insufficient experimental data\nCannot compare FOOGD effects',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('FOOGD Effect Comparison')

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison summary saved to: {os.path.join(output_dir, 'comparison_summary.png')}")

    # 打印详细数据表格
    print("\n" + "="*80)
    print("Experimental Data Summary:")
    print("="*80)
    print(f"{'Experiment Name':<30} {'Alpha':<8} {'FOOGD':<8} {'Final Test Acc':<15} {'Best Acc':<15} {'Near-OOD AUROC':<15} {'Far-OOD AUROC':<15}")
    print("-"*110)

    for metric in metrics:
        print(f"{metric['name'][:30]:<30} {metric['alpha']:<8} {'Enabled' if metric['use_foogd'] else 'Disabled':<8} "
              f"{metric['final_test_acc']:<15.4f} {metric['best_acc']:<15.4f} "
              f"{metric['avg_near_auroc']:<15.4f} {metric['avg_far_auroc']:<15.4f}")

def main():
    """Main function"""
    print("Loading experimental data...")
    experiments = find_all_experiments()

    if not experiments:
        print("No experimental data found! Please ensure experiments folder contains training_history.json files")
        return

    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        config = exp['config']
        print(f"  - {exp['name']}: alpha={config['alpha']}, FOOGD={'Enabled' if config['use_foogd'] else 'Disabled'}, "
              f"n_clients={config['n_clients']}, communication_rounds={config['communication_rounds']}")

    # 创建可视化输出目录
    output_dir = './visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制各种图表
    print("\nPlotting training curves...")
    plot_training_losses(experiments, output_dir)

    print("\nPlotting OOD detection performance...")
    plot_ood_performance(experiments, output_dir)

    print("\nPlotting comprehensive comparison summary...")
    plot_comparison_summary(experiments, output_dir)

    print(f"\nAll visualization charts saved to: {output_dir}/")

if __name__ == '__main__':
    main()