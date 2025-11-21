import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import resnet34
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号


# 1. 全局参数
data_path = "./data/PlantDoc"
train_path = os.path.join(data_path, "TRAIN")
test_path = os.path.join(data_path, "TEST")
result_path = "./results/resnet34_transfer"
os.makedirs(result_path, exist_ok=True)

num_classes = 27
batch_size = 32
img_size = 224   # ResNet34 期望 224x224
epochs_stage1 = 10  # 冻结阶段
epochs_stage2 = 30  # 微调阶段
use_weighted_sampler = False  # 类别严重不均衡时，设为 True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 数据增强与加载
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

train_dataset_full = datasets.ImageFolder(train_path, transform=train_transform)
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

# 划分验证集 20%
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size
train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

# 类别不均衡处理（基于训练集子集统计权重）
def build_weighted_sampler(dataset_subset):
    # dataset_subset 是 Subset，需要通过 indices 找原始数据标签
    targets = np.array([train_dataset_full.targets[i] for i in dataset_subset.indices])
    class_counts = np.bincount(targets, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

if use_weighted_sampler:
    sampler = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 3. 构建预训练 ResNet34
def build_model(num_classes):
    model = resnet34(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# 4. 训练与验证函数
def evaluate_model(model, loader, classes, prefix=""):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro")

    print("\n" + "="*50)
    print(f"{prefix}评估结果:")
    print("="*50)
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"宏平均F1-score: {f1:.4f}")
    print(f"宏平均精确率 (Precision): {precision:.4f}")
    print(f"宏平均召回率 (Recall): {recall:.4f}")
    print("="*50)

    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print("\n详细分类报告:")
    print(report)

    with open(os.path.join(result_path, f"{prefix.lower()}_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{prefix}混淆矩阵")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(result_path, f"{prefix.lower()}_confusion.png"))
    plt.close()

    return acc, f1, precision, recall

def run_one_stage(model, train_loader, val_loader, epochs, optimizer, criterion, stage_name="Stage"):
    best_val_acc = 0.0
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)
        train_acc_hist.append(train_acc)
        train_loss_hist.append(train_loss)

        # 验证
        model.eval()
        v_total, v_correct = 0, 0
        v_running_loss = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                v_running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_acc = v_correct / v_total
        val_loss = v_running_loss / len(val_loader)
        val_acc_hist.append(val_acc)
        val_loss_hist.append(val_loss)

        print(f"[{stage_name} Epoch {epoch+1}/{epochs}] "
              f"TrainAcc={train_acc:.3f} ValAcc={val_acc:.3f} "
              f"TrainLoss={train_loss:.3f} ValLoss={val_loss:.3f}")

        # 保存最优
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(result_path, f"best_resnet34_{stage_name.lower()}.pth"))

    # 曲线
    plt.figure(figsize=(6,4))
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.legend(); plt.title(f"{stage_name} 准确率"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.savefig(os.path.join(result_path, f"{stage_name.lower()}_acc_curve.png")); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.legend(); plt.title(f"{stage_name} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig(os.path.join(result_path, f"{stage_name.lower()}_loss_curve.png")); plt.close()
    # 保存验证集准确率曲线数据，方便后续对比
    np.save(os.path.join(result_path, f"{stage_name.lower()}_val_acc_hist.npy"), np.array(val_acc_hist))
    np.save(os.path.join(result_path, f"{stage_name.lower()}_train_acc_hist.npy"), np.array(train_acc_hist))
    np.save(os.path.join(result_path, f"{stage_name.lower()}_val_loss_hist.npy"), np.array(val_loss_hist))
    np.save(os.path.join(result_path, f"{stage_name.lower()}_train_loss_hist.npy"), np.array(train_loss_hist))

# 5. 主流程：两阶段训练
if __name__ == "__main__":
    print("开始训练 ResNet34 迁移学习：")
    classes = test_dataset.classes

    # 构建模型
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # 阶段 1：冻结主干，只训练 fc
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer_stage1 = optim.Adam(model.fc.parameters(), lr=1e-3)
    run_one_stage(model, train_loader, val_loader, epochs_stage1, optimizer_stage1, criterion, stage_name="Stage1_FC")

    # 加载阶段 1 的最佳权重
    model.load_state_dict(torch.load(os.path.join(result_path, "best_resnet34_stage1_fc.pth")))

    # 阶段 2：解冻全网络，小学习率微调
    for param in model.parameters():
        param.requires_grad = True

    optimizer_stage2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    run_one_stage(model, train_loader, val_loader, epochs_stage2, optimizer_stage2, criterion, stage_name="Stage2_Finetune")

    # 评估：加载阶段 2 的最佳权重（如果阶段 2 未保存，用阶段 1 权重）
    best_path = os.path.join(result_path, "best_resnet34_stage2_finetune.pth")
    if not os.path.exists(best_path):
        best_path = os.path.join(result_path, "best_resnet34_stage1_fc.pth")
    model.load_state_dict(torch.load(best_path))

    print("在测试集上评估：")
    evaluate_model(model, test_loader, classes, prefix="测试集")
    print("结束，所有结果已保存")
