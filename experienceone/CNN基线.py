import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

#1.参数设置
data_path = "./data/PlantDoc"
train_path = os.path.join(data_path,"TRAIN")
test_path = os.path.join(data_path,"TEST")
result_path = "./results/cnn_baseline1"
os.makedirs(result_path,exist_ok=True)
batch_size = 32 #每次训练使用得图片数量
num_classes = 27 #类别
lr = 1e-3 #学习率，控制参数更新步长
epochs = 50 #轮数
img_size = 128 #统一调整图片尺寸

#2.数据增强与加载 训练集增强，测试集不增强
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),#先统一尺寸
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    #transforms.RandomAffine(degrees=0, translate=(0.1,0.1), shear=10), # 仿射变换
    #transforms.RandomPerspective(distortion_scale=0.3, p=0.5),          # 随机透视
    transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3),
    #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),        # 锐化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
#加载数据集
train_dataset = datasets.ImageFolder(train_path,transform=train_transform)
test_dataset = datasets.ImageFolder(test_path,transform=test_transform)
#划分验证集20%
train_size = int(0.8*len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset,val_dataset = torch.utils.data.random_split(train_dataset,[train_size,val_size])
#数据加载器
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size)
test_loader = DataLoader(test_dataset,batch_size=batch_size)

#3.定义CNN基线模型
class CNNBaseline(nn.Module):
    def __init__(self, num_classes=27):
        super(CNNBaseline,self).__init__()
        # 特征提取（卷积层\三层）
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #nn.Conv2d(128,256,3,padding=1),   # 增加第四层卷积
            #nn.ReLU(),
            #nn.MaxPool2d(2),
        )
        # 分类（全连接层）
        self.classifier = nn.Sequential(
            nn.Flatten(),
             # 第一个全连接层
            nn.Linear(128*(img_size//8)*(img_size//8),256),
            nn.ReLU(),
            #防止过拟合
            nn.Dropout(0.5),#随机丢弃50%的神经元
            nn.Linear(256,num_classes)
        )
    #前向传播
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
#4.训练与验证函数
def train_model(model,criterion,optimizer,train_loader,val_loader,epochs=50, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_acc,val_acc=[],[]
    train_loss,val_loss =[],[]
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()#训练
        correct,total = 0,0
        running_loss = 0.0
        for imgs,labels in train_loader:
            imgs,labels=imgs.to(device),labels.to(device)
            optimizer.zero_grad()#清空优化器中之前计算的梯度
            outputs = model(imgs)#前向传播
            loss = criterion(outputs,labels)
            loss.backward()#反向传播
            optimizer.step()
            running_loss+=loss.item()
            _,predict = torch.max(outputs,1)
            correct+=(predict==labels).sum().item()
            total+=labels.size(0)
        train_accuracy = correct/total
        train_loss1 = running_loss/len(train_loader) #回归率
        train_acc.append(train_accuracy)
        train_loss.append(train_loss1)

        #验证
        model.eval()
        correct,total = 0,0
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs,labels=imgs.to(device),labels.to(device)
                outputs = model(imgs)#只有前向传播
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _,predict = torch.max(outputs,1)
                correct+=(predict==labels).sum().item()
                total+=labels.size(0)
            val_accuracy = correct/total
            val_loss1 = val_running_loss/len(val_loader)
            val_acc.append(val_accuracy)
            val_loss.append(val_loss1)
            print(f"[Epoch {epoch+1}/{epochs}] TrainAcc={train_accuracy:.3f} ValAcc={val_accuracy:.3f} "
      f"TrainLoss={train_loss1:.3f} ValLoss={val_loss1:.3f} ")
            # 保存最优模型
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), os.path.join(result_path, "best_cnn.pth"))
            
    #绘制曲线(准确率和loss曲线)
    plt.figure(figsize=(6, 4))
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title("CNN训练与验证准确率")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(result_path, "cnn_acc_curve.png"))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("CNN训练与验证Loss曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(result_path, "cnn_loss_curve.png"))
    plt.close()   
    #保存训练过程数据，便于后续统一对比绘图
    np.save(os.path.join(result_path, "train_acc_hist.npy"), np.array(train_acc))
    np.save(os.path.join(result_path, "val_acc_hist.npy"), np.array(val_acc))
    np.save(os.path.join(result_path, "train_loss_hist.npy"), np.array(train_loss))
    np.save(os.path.join(result_path, "val_loss_hist.npy"), np.array(val_loss))
     
             
#5测试评估函数
def evaluate_model(model,loader,classes):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_predict,all_labels=[],[]
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predict = torch.max(outputs, 1)
            all_predict.extend(predict.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
     # 计算评估指标
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc = accuracy_score(all_labels, all_predict)
    f1 = f1_score(all_labels, all_predict, average="macro")
    precision = precision_score(all_labels, all_predict, average="macro")
    recall = recall_score(all_labels, all_predict, average="macro")
    
    # 在控制台输出测试集评估结果
    print("\n" + "="*50)
    print("测试集评估结果:")
    print("="*50)
    print(f"测试集准确率 (Accuracy): {acc:.4f}")
    print(f"测试集宏平均F1-score: {f1:.4f}")
    print(f"测试集宏平均精确率 (Precision): {precision:.4f}")
    print(f"测试集宏平均召回率 (Recall): {recall:.4f}")
    print("="*50)

    #保存分类报告到txt文件
    report = classification_report(all_labels, all_predict, target_names=classes)
    print("\n详细分类报告:")
    print(report)
    with open(os.path.join(result_path, "cnn_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    #保存整体准确率和F1-score等结果到单独文件
    with open(os.path.join(result_path, "cnn_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test F1-score: {f1:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predict)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("CNN混淆矩阵")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(result_path, "cnn_confusion.png"))
    plt.close()
    return acc, f1, precision, recall

#主函数
if __name__ == "__main__":
    print("开始训练CNN基线模型：")
    model = CNNBaseline(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    train_model(model,criterion,optimizer,train_loader,val_loader,epochs=epochs)
    print("加载验证集最优模型并在测试集上评估")
    model.load_state_dict(torch.load(os.path.join(result_path,"best_cnn.pth")))
    classes = test_dataset.classes
    evaluate_model(model,test_loader,classes)
    print("结束，所有结果已保存")


