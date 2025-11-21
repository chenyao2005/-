import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 1. 验证集准确率曲线对比
epochs = range(1, 31)  # 30个epoch

val_acc_none = np.load("./results/resnet18_transfer_none/stage2_finetune_val_acc_hist.npy")
val_acc_basic = np.load("./results/resnet18_transfer_basic/stage2_finetune_val_acc_hist.npy")
val_acc_strong = np.load("./results/resnet18_transfer_strong/stage2_finetune_val_acc_hist.npy")

plt.figure(figsize=(8,6))
plt.plot(epochs, val_acc_none, label="无增强")
plt.plot(epochs, val_acc_basic, label="基础增强")
plt.plot(epochs, val_acc_strong, label="强增强")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("数据增强消融实验对比（验证集）")
plt.legend()
plt.savefig("augmentation_ablation_val.png")
plt.show()


# 2. 测试集整体指标柱状图
def parse_report(report_path):
    """从 report.txt 中解析整体指标 (accuracy + macro avg)"""
    metrics = {}
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("accuracy"):
                # accuracy 行格式: accuracy                           0.59       232
                parts = line.split()
                metrics["accuracy"] = float(parts[1])
            elif line.startswith("macro avg"):
                # macro avg 行格式: macro avg       0.61      0.58      0.58       232
                parts = line.split()
                # parts = ["macro","avg","0.61","0.58","0.58","232"]
                metrics["precision"] = float(parts[2])
                metrics["recall"] = float(parts[3])
                metrics["f1"] = float(parts[4])
    return metrics

# 读取三种增强策略的 report.txt
metrics_none = parse_report("./results/resnet18_transfer_none/测试集_report.txt")
metrics_basic = parse_report("./results/resnet18_transfer_basic/测试集_report.txt")
metrics_strong = parse_report("./results/resnet18_transfer_strong/测试集_report.txt")

# 准备绘图数据
labels = ["Accuracy", "Macro-F1", "Precision", "Recall"]
none_values = [metrics_none["accuracy"], metrics_none["f1"], metrics_none["precision"], metrics_none["recall"]]
basic_values = [metrics_basic["accuracy"], metrics_basic["f1"], metrics_basic["precision"], metrics_basic["recall"]]
strong_values = [metrics_strong["accuracy"], metrics_strong["f1"], metrics_strong["precision"], metrics_strong["recall"]]

x = range(len(labels))
bar_width = 0.25

plt.figure(figsize=(8,6))
plt.bar([i - bar_width for i in x], none_values, width=bar_width, label="无增强")
plt.bar(x, basic_values, width=bar_width, label="基础增强")
plt.bar([i + bar_width for i in x], strong_values, width=bar_width, label="强增强")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("三种数据增强策略在测试集上的表现对比")
plt.legend()
plt.ylim(0, 1)  # 指标范围在 0~1
plt.savefig("augmentation_ablation_test.png")
plt.show()
