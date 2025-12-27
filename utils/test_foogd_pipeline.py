#!/usr/bin/env python3
"""
FOOGD模块测试脚本
分别测试使用FOOGD和不使用FOOGD的情况，设置少量轮数快速验证

作者: Claude Code
日期: 2025-12-07
"""

import os
import sys
import torch
import argparse
import json
from datetime import datetime
import subprocess
import time

def run_experiment(use_foogd, experiment_name, output_dir="./test_experiments"):
    """
    运行单个实验

    Args:
        use_foogd: 是否使用FOOGD模块
        experiment_name: 实验名称
        output_dir: 输出目录
    """
    print(f"\n{'='*60}")
    print(f"开始实验: {experiment_name}")
    print(f"使用FOOGD: {use_foogd}")
    print(f"{'='*60}")

    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 构建命令行参数
    cmd = [
        "python", "train_federated.py",
        "--data_root", "./Plankton_OOD_Dataset",
        "--n_clients", "3",  # 少量客户端
        "--alpha", "5.0",
        "--communication_rounds", "5",  # 少量轮数
        "--local_epochs", "1",
        "--batch_size", "16",  # 小批量大小
        "--model_type", "densenet121",  # 使用较小的模型
        "--eval_frequency", "1",
        "--save_frequency", "5",
        "--output_dir", exp_dir,
        "--seed", "42"
    ]

    if use_foogd:
        cmd.append("--use_foogd")

    print(f"执行命令: {' '.join(cmd)}")

    # 运行实验
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # 不抛出异常，让我们自己处理
        )

        elapsed_time = time.time() - start_time

        # 保存输出
        output_file = os.path.join(exp_dir, "run_output.txt")
        with open(output_file, 'w') as f:
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write(f"运行时间: {elapsed_time:.2f}秒\n")
            f.write(f"返回码: {result.returncode}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("标准输出:\n")
            f.write(result.stdout)
            f.write("\n" + "="*60 + "\n")
            f.write("标准错误:\n")
            f.write(result.stderr)

        # 检查结果
        if result.returncode == 0:
            print(f"✓ 实验成功完成! 耗时: {elapsed_time:.2f}秒")

            # 检查是否生成了必要的文件
            required_files = [
                "config.json",
                "training_history.json",
                "training_curves.png"
            ]

            missing_files = []
            for file in required_files:
                file_path = os.path.join(exp_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)

            if missing_files:
                print(f"⚠ 警告: 缺少以下文件: {missing_files}")
            else:
                print(f"✓ 所有必要文件已生成")

            # 读取训练历史
            history_path = os.path.join(exp_dir, "training_history.json")
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)

                if history.get('test_accuracies'):
                    final_acc = history['test_accuracies'][-1]
                    print(f"✓ 最终全局准确率: {final_acc:.4f}")

                if history.get('avg_person_acc'):
                    final_person_acc = history['avg_person_acc'][-1]
                    print(f"✓ 最终个性化准确率: {final_person_acc:.4f}")

                if use_foogd and history.get('near_auroc'):
                    final_near_auroc = history['near_auroc'][-1]
                    print(f"✓ 最终Near-OOD AUROC: {final_near_auroc:.4f}")

            return True, exp_dir, elapsed_time

        else:
            print(f"✗ 实验失败! 返回码: {result.returncode}")
            print(f"错误输出:\n{result.stderr[:500]}...")  # 只显示前500个字符
            return False, exp_dir, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ 实验异常: {str(e)}")
        return False, exp_dir, elapsed_time

def check_dataset():
    """检查数据集是否存在"""
    data_root = "./Plankton_OOD_Dataset"

    if not os.path.exists(data_root):
        print(f"✗ 数据集目录不存在: {data_root}")
        print("请先运行 split_dataset.py 准备数据集")
        return False

    # 检查必要的子目录
    required_dirs = ["ID_images", "OOD_Near", "OOD_Far"]
    missing_dirs = []

    for dir_name in required_dirs:
        dir_path = os.path.join(data_root, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"✗ 数据集不完整，缺少以下目录: {missing_dirs}")
        return False

    print(f"✓ 数据集检查通过: {data_root}")
    return True

def main():
    parser = argparse.ArgumentParser(description='FOOGD模块测试脚本')
    parser.add_argument('--output_dir', type=str, default='./test_experiments',
                       help='测试输出目录')
    parser.add_argument('--skip_dataset_check', action='store_true', default=False,
                       help='跳过数据集检查')
    args = parser.parse_args()

    print("FOOGD模块测试脚本")
    print("="*60)

    # 检查数据集
    if not args.skip_dataset_check and not check_dataset():
        print("\n请先准备数据集:")
        print("1. 确保数据集已放置在 ./Plankton_OOD_Dataset 目录")
        print("2. 运行: python split_dataset.py")
        print("3. 或使用 --skip_dataset_check 跳过检查")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行两个实验
    results = []

    # 实验1: 不使用FOOGD
    success1, exp_dir1, time1 = run_experiment(
        use_foogd=False,
        experiment_name="without_foogd",
        output_dir=args.output_dir
    )
    results.append({
        "experiment": "without_foogd",
        "success": success1,
        "directory": exp_dir1,
        "time": time1
    })

    # 实验2: 使用FOOGD
    success2, exp_dir2, time2 = run_experiment(
        use_foogd=True,
        experiment_name="with_foogd",
        output_dir=args.output_dir
    )
    results.append({
        "experiment": "with_foogd",
        "success": success2,
        "directory": exp_dir2,
        "time": time2
    })

    # 生成测试报告
    print(f"\n{'='*60}")
    print("测试报告")
    print(f"{'='*60}")

    report_file = os.path.join(args.output_dir, "test_report.txt")
    with open(report_file, 'w') as f:
        f.write("FOOGD模块测试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")

        all_success = True
        for result in results:
            status = "✓ 通过" if result["success"] else "✗ 失败"
            f.write(f"实验: {result['experiment']}\n")
            f.write(f"状态: {status}\n")
            f.write(f"耗时: {result['time']:.2f}秒\n")
            f.write(f"目录: {result['directory']}\n")
            f.write("-"*40 + "\n")

            if not result["success"]:
                all_success = False

        f.write("\n" + "="*60 + "\n")
        if all_success:
            f.write("✓ 所有测试通过!\n")
            f.write("FOOGD模块和不使用FOOGD的版本都能正常运行。\n")
        else:
            f.write("✗ 部分测试失败\n")
            f.write("请检查失败实验的输出日志。\n")

    print(f"测试报告已保存: {report_file}")

    # 打印总结
    print("\n测试总结:")
    for result in results:
        status = "✓ 通过" if result["success"] else "✗ 失败"
        print(f"  {result['experiment']}: {status} ({result['time']:.2f}秒)")

    if all_success:
        print("\n✓ 恭喜! 所有测试通过!")
        print("FOOGD模块和不使用FOOGD的版本都能正常运行。")
    else:
        print("\n✗ 部分测试失败，请检查输出日志。")

if __name__ == "__main__":
    main()