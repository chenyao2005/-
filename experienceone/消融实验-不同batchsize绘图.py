import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 要测试的 batch size 列表
batch_sizes = [16, 32, 64]

# 图1：测试集柱状图
labels = ["Accuracy", "Macro-F1", "Precision", "Recall"]
bar_width = 0.25
x = np.arange(len(labels))
plt.figure(figsize=(10, 6))

metrics_dict = {}
offset = 0
for bs in batch_sizes:
    combo = f"batch_size={bs}"
    result_path = f"./results/resnet18_bs{bs}"
    report_path = os.path.join(result_path, "测试集_report.txt")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        metrics = {}
        for line in lines:
            if line.strip().startswith("accuracy"):
                parts = line.split()
                metrics["accuracy"] = float(parts[1])
            elif line.strip().startswith("macro avg"):
                parts = line.split()
                metrics["precision"] = float(parts[2])
                metrics["recall"] = float(parts[3])
                metrics["f1"] = float(parts[4])
        if "accuracy" in metrics and "f1" in metrics:
            values = [metrics["accuracy"], metrics["f1"], metrics["precision"], metrics["recall"]]
            metrics_dict[combo] = values
            plt.bar(x + offset * bar_width, values, width=bar_width, label=combo)
            offset += 1

plt.xticks(x + bar_width * (offset-1)/2, labels)
plt.ylabel("Score")
plt.title("不同 batch size 在测试集上的表现")
plt.ylim(0, 1)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("bs_test_metrics_bar.png")
plt.close()

# 图2：验证集准确率曲线（拼接 stage1 + stage2）
plt.figure(figsize=(10, 6))
for bs in batch_sizes:
    result_path = f"./results/resnet18_bs{bs}"
    acc_path1 = os.path.join(result_path, "stage1_fc_val_acc_hist.npy")
    acc_path2 = os.path.join(result_path, "stage2_finetune_val_acc_hist.npy")
    if os.path.exists(acc_path1) and os.path.exists(acc_path2):
        acc1 = np.load(acc_path1)
        acc2 = np.load(acc_path2)
        val_acc = np.concatenate([acc1, acc2])
        label = f"batch_size={bs}"
        plt.plot(range(1, len(val_acc)+1), val_acc, label=label)

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("完整训练过程验证集准确率曲线（不同 batch size）")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("bs_val_acc_curve.png")
plt.close()

# 图3：验证集 loss 曲线（拼接 stage1 + stage2）
plt.figure(figsize=(10, 6))
for bs in batch_sizes:
    result_path = f"./results/resnet18_bs{bs}"
    loss_path1 = os.path.join(result_path, "stage1_fc_val_loss_hist.npy")
    loss_path2 = os.path.join(result_path, "stage2_finetune_val_loss_hist.npy")
    if os.path.exists(loss_path1) and os.path.exists(loss_path2):
        loss1 = np.load(loss_path1)
        loss2 = np.load(loss_path2)
        val_loss = np.concatenate([loss1, loss2])
        label = f"batch_size={bs}"
        plt.plot(range(1, len(val_loss)+1), val_loss, label=label)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("完整训练过程验证集 Loss 曲线（不同 batch size）")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("bs_val_loss_curve.png")
plt.close()

print("已生成 bs_test_metrics_bar.png、bs_val_acc_curve.png、bs_val_loss_curve.png")
