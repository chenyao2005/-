import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import re

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 模型名称与路径映射
models = {
    "Baseline1 (3层CNN)": "./results/cnn_baseline1",
    "Baseline3 (4层CNN)": "./results/cnn_baseline3",
    "ResNet18 Transfer": "./results/resnet18_transfer_basic",
    "ResNet34 Transfer": "./results/resnet34_transfer"
}

# 图一：验证集准确率曲线对比
plt.figure(figsize=(8, 5))
epochs = range(1, 41)

for name, path in models.items():
    if "ResNet" in name:
        stage1 = np.load(os.path.join(path, "stage1_fc_val_acc_hist.npy"))
        stage2 = np.load(os.path.join(path, "stage2_finetune_val_acc_hist.npy"))
        val_acc = np.concatenate([stage1, stage2])
        plt.plot(epochs, val_acc, label=name)
    else:  # Baseline
        val_acc = np.load(os.path.join(path, "val_acc_hist.npy"))
        plt.plot(epochs, val_acc[:40], label=name)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("四种模型验证集准确率曲线对比（统一40轮）")
plt.legend()
plt.tight_layout()
plt.savefig("val_acc_comparison.png")
plt.close()



# 图二：测试集整体指标柱状图对比
def parse_classification_report(report_path):
    metrics = {}
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("accuracy"):
                parts = line.split()
                metrics["accuracy"] = float(parts[1])
            elif line.startswith("macro avg"):
                parts = line.split()
                metrics["precision"] = float(parts[2])
                metrics["recall"] = float(parts[3])
                metrics["f1"] = float(parts[4])
    return metrics

labels = ["Accuracy", "Macro-F1", "Precision", "Recall"]
x = np.arange(len(labels))
bar_width = 0.2

plt.figure(figsize=(10, 6))
for i, (name, path) in enumerate(models.items()):
    # 先尝试 ResNet 的报告文件
    report_path = os.path.join(path, "测试集_report.txt")
    # 如果不存在，就尝试 Baseline 的报告文件
    if not os.path.exists(report_path):
        report_path = os.path.join(path, "cnn_report.txt")

    if os.path.exists(report_path):
        metrics = parse_classification_report(report_path)
        values = [
            metrics.get("accuracy", 0),
            metrics.get("f1", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0)
        ]
        plt.bar(x + i * bar_width, values, width=bar_width, label=name)


plt.xticks(x + 1.5 * bar_width, labels)
plt.ylabel("Score")
plt.title("四种模型在测试集上的整体指标对比")
plt.legend()
plt.tight_layout()
plt.savefig("test_metrics_comparison.png")   # 保存到当前目录
plt.close()

print("已生成 val_acc_comparison.png 和 test_metrics_comparison.png")
