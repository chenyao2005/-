import os, glob, random, csv
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#注释的是网格调参的过程，现在剩下的代码是用最优的参数写的


# 1. 数据加载，遍历数据集目录，返回所有图片路径、对应标签和类别列表
def list_images(root):
    #获取所有类别 并按字母排序
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
    paths,labels = [],[]
    for cidx, c in enumerate(classes):
        for p in glob.glob(os.path.join(root,c,"*")):
            if p.lower().endswith((".jpg",".jpeg",".png")):
                paths.append(p)
                labels.append(c)
    return paths,labels,classes

#划分训练集20%为验证集
def split_train_val(train_root,val_ratio = 0.2, seed = 42):
    paths, labels,classes = list_images(train_root)
    X_train, X_val, y_train, y_val = train_test_split(paths,labels,test_size=val_ratio,random_state=seed,stratify=labels)
    return X_train,X_val,y_train,y_val,classes

#2.特征提取

#读取图片转为灰度，缩放到128
def preprocess_image(path, size=(128,128)):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img,size)

#提取 HOG 特征
def hog_feature(img):
    hog = cv2.HOGDescriptor((128,128),(32,32),(16,16),(8,8),9)
    return hog.compute(img).flatten()
"""# —— 新增：构造 HOG 对象（便于配置） ——
def make_hog(win=(128,128), block=(16,16), stride=(8,8), cell=(8,8), bins=9):
    return cv2.HOGDescriptor(win, block, stride, cell, bins)

# —— 新增：跑 HOG 网格调参并记录结果 ——
def run_hog_gridsearch(Xtr_paths, Xval_paths, Xte_paths, ytr, yval, yte, classes, results_dir):
    # 1) 定义网格
    hog_param_grid = [
        {"win":(128,128), "block":(16,16), "stride":(8,8),  "cell":(8,8),  "bins":9},   # 基线
        {"win":(128,128), "block":(16,16), "stride":(8,8),  "cell":(4,4),  "bins":9},   # 更细粒度纹理
        {"win":(128,128), "block":(32,32), "stride":(16,16),"cell":(8,8),  "bins":9},   # 更多上下文
        {"win":(128,128), "block":(16,16), "stride":(8,8),  "cell":(8,8),  "bins":18},  # 更多方向
        {"win":(96,96),   "block":(16,16), "stride":(8,8),  "cell":(4,4),  "bins":9},   # 缩小窗口减少背景
    ]

    hog_results = []   # 每个配置的 (cfg, val_acc, val_f1)
    best_val_f1 = -1
    best_cfg = None
    best_te_metrics = None  # (test_acc, test_f1, test_cm)

    # 2) 遍历网格
    for cfg in hog_param_grid:
        hog = make_hog(cfg["win"], cfg["block"], cfg["stride"], cfg["cell"], cfg["bins"])

        # 注意：预处理时 size 应与 win 一致
        Xtr = [hog.compute(preprocess_image(p, size=cfg["win"])).flatten() for p in Xtr_paths]
        Xval = [hog.compute(preprocess_image(p, size=cfg["win"])).flatten() for p in Xval_paths]
        Xte  = [hog.compute(preprocess_image(p, size=cfg["win"])).flatten() for p in Xte_paths]

        # 训练与评估（固定 SVM：C=0.1, gamma='scale'）
        # 在验证集上选择最优；为了代码复用，这里用 test 位置传 Xval/yval 来比较
        val_acc, val_f1, val_cm, _ = train_and_eval(np.array(Xtr), ytr, np.array(Xval), yval,
                                                    np.array(Xval), yval, clf_name="SVM-RBF")
        hog_results.append([cfg, val_acc, val_f1])
        print(f"[HOG-Grid] cfg={cfg} | Val Acc={val_acc:.3f}, Val F1={val_f1:.3f}")

        # 记录最佳（以宏 F1 为主）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_cfg = cfg
            # 用同一特征在测试集上评估一次，记录测试指标与混淆矩阵
            test_acc, test_f1, test_cm, _ = train_and_eval(np.array(Xtr), ytr, np.array(Xval), yval,
                                                           np.array(Xte), yte, clf_name="SVM-RBF")
            best_te_metrics = (test_acc, test_f1, test_cm)

    # 3) 保存每个配置的验证结果（便于画图）
    csv_path = os.path.join(results_dir, "hog_gridsearch_val.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["HOG_Params", "Val_Acc", "Val_F1"])
        for cfg, acc, f1 in hog_results:
            writer.writerow([str(cfg), acc, f1])

    # 4) 保存最佳配置的测试混淆矩阵与最终指标
    if best_cfg is not None and best_te_metrics is not None:
        test_acc, test_f1, test_cm = best_te_metrics
        plot_confusion(test_cm, classes, f"HOG+SVM Best cfg {best_cfg}", 
                       os.path.join(results_dir, "HOG_SVM_best_cm.png"))
        print(f"[HOG-Grid] 最佳配置: {best_cfg} | Test Acc={test_acc:.3f}, Test F1={test_f1:.3f}")

        # 返回给主程序做汇总
        return best_cfg, hog_results, (test_acc, test_f1)
    else:
        return None, hog_results, None
    """
#构建bovw词典，detector是SIFI或SURF，聚类中心数默认256，最大图片数量默认500
def build_bovw_dictionary(image_paths,detector = "SIFT",dict_size=256,max_images=500):
    sample_paths = random.sample(image_paths,min(len(image_paths),max_images))#随机取样，保证不超过最大数量
    if detector=="SIFT":
        extractor = cv2.SIFT_create()
    else:
        extractor = cv2.xfeatures2d.SURF_create()
    descs=[]#初始化空列表存储所有描述子
    for p in tqdm(sample_paths,desc=f"Building {detector} dict"):
        img = preprocess_image(p)
        _, d = extractor.detectAndCompute(img,None)#只关心描述子
        if d is not None: descs.append(d)
    all_desc = np.vstack(descs)#将列表中所有描述子矩阵垂直堆叠成一个大的二维数组
    #Kmeans终止条件 最大迭代次数为50次中心点变化小于0.1时停止
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,0.1)
    #执行k-means聚类：将描述子转换为float32类型（OpenCV要求）不使用初始标签重复聚类3次
    _,_,centers=cv2.kmeans(all_desc.astype(np.float32),dict_size,None,criteria,3,cv2.KMEANS_PP_CENTERS)
    return centers

#提取bovw特征，将局部特征分配到最近的词典中心，统计直方图
def bovw_feature(img, centers, detector="SIFT"):
    if detector=="SIFT":
        extractor = cv2.SIFT_create()
    else:
        extractor = cv2.xfeatures2d.SURF_create()
    _, desc = extractor.detectAndCompute(img,None)
    if desc is None: return np.zeros(len(centers))
    dists=np.linalg.norm(desc[:,None,:]-centers[None,:,:],axis=2)#计算特征到所有视觉单词的距离
    idx = np.argmin(dists,axis=1)#找到每个特征的最近视觉单词
    hist,_=np.histogram(idx,bins=np.arange(len(centers)+1))#hist 是长度为 256 的直方图，表示每个视觉单词的出现频率
    return hist/ (np.linalg.norm(hist)+1e-8)#归一化
"""
def run_bovw_gridsearch(Xtr_paths, Xval_paths, Xte_paths, ytr, yval, yte, classes, results_dir,
                        dict_sizes=[128,256,512], max_images=800):
    results = []
    best_f1 = -1
    best_dict = None
    best_test_metrics = None

    for dsize in dict_sizes:
        print(f"[BoVW-Grid] 构建 SIFT 词典, dict_size={dsize}")
        centers = build_bovw_dictionary(Xtr_paths, detector="SIFT", dict_size=dsize, max_images=max_images)

        # 提取 BoVW 特征
        Xtr = [bovw_feature(preprocess_image(p), centers, "SIFT") for p in Xtr_paths]
        Xval = [bovw_feature(preprocess_image(p), centers, "SIFT") for p in Xval_paths]
        Xte  = [bovw_feature(preprocess_image(p), centers, "SIFT") for p in Xte_paths]

        # 训练 + 验证
        val_acc, val_f1, _, _ = train_and_eval(np.array(Xtr), ytr, np.array(Xval), yval,
                                               np.array(Xval), yval, clf_name="SVM-RBF")
        results.append([dsize, val_acc, val_f1])
        print(f"[BoVW-Grid] dict={dsize} | Val Acc={val_acc:.3f}, Val F1={val_f1:.3f}")

        # 记录最佳
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_dict = dsize
            # 在测试集上评估
            test_acc, test_f1, cm, _ = train_and_eval(np.array(Xtr), ytr, np.array(Xval), yval,
                                                      np.array(Xte), yte, clf_name="SVM-RBF")
            best_test_metrics = (test_acc, test_f1, cm)

    # 保存验证集结果
    csv_path = os.path.join(results_dir, "sift_bovw_gridsearch_val.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dict_size", "Val_Acc", "Val_F1"])
        writer.writerows(results)

    # 保存最佳配置的测试结果
    if best_dict and best_test_metrics:
        test_acc, test_f1, cm = best_test_metrics
        plot_confusion(cm, classes, f"SIFT-BoVW+SVM dict={best_dict}",
                       os.path.join(results_dir, "SIFT_BoVW_best_cm.png"))
        print(f"[BoVW-Grid] 最佳字典大小: {best_dict} | Test Acc={test_acc:.3f}, Test F1={test_f1:.3f}")
        return best_dict, (test_acc, test_f1)
    else:
        return None, None
"""    
#3.分类器训练和在测试集上评估
def train_and_eval(Xtr,ytr,Xval,yval,Xte,yte,clf_name="SVM-RBF"):
    scaler = StandardScaler()#数据标准化
    Xtr=scaler.fit_transform(Xtr) #用训练集计算均值和标准差，并转换训练集
    #用训练集的统计量转换验证集和测试集
    Xval=scaler.transform(Xval) 
    Xte = scaler.transform(Xte)
    #选择不同分类器
    if clf_name=="SVM-RBF":
        clf=SVC(kernel="rbf",C=100,gamma=0.0001)
    elif clf_name=="SVM-Linear":
        clf=LinearSVC()
    elif clf_name=="LogReg":
        clf=LogisticRegression(C=1,penalty="l2",solver="lbfgs", max_iter=500)
    elif clf_name=="MLP":
        clf=MLPClassifier(hidden_layer_sizes=(512),max_iter=200,alpha=0.001)
    else:
        raise ValueError("Unknown classifier")
    #训练
    clf.fit(Xtr,ytr)
    #在测试集上预测
    ypred=clf.predict(Xte)
    acc=accuracy_score(yte,ypred)#准确率
    f1=f1_score(yte,ypred,average="macro")#计算f1分数 average="macro"：计算每个类别的F1分数，然后取平均
    cm=confusion_matrix(yte,ypred)#计算混淆矩阵
    return acc,f1,cm,ypred
#绘制并保存混淆矩阵
def plot_confusion(cm, classes, title, save_path):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
"""
def run_svm_gridsearch(Xtr, ytr, Xval, yval, Xte, yte, classes, results_dir):
    # 定义参数网格
    C_values = [0.1, 1, 10, 100]
    gamma_values = ["scale", "auto", 1e-3, 1e-4]

    results = []
    best_f1 = -1
    best_params = None
    best_test_metrics = None

    for C in C_values:
        for gamma in gamma_values:
            clf = SVC(kernel="rbf", C=C, gamma=gamma)
            # 标准化
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xval_s = scaler.transform(Xval)
            Xte_s  = scaler.transform(Xte)

            clf.fit(Xtr_s, ytr)
            ypred_val = clf.predict(Xval_s)

            val_acc = accuracy_score(yval, ypred_val)
            val_f1  = f1_score(yval, ypred_val, average="macro")
            results.append([C, gamma, val_acc, val_f1])
            print(f"[SVM-Grid] C={C}, gamma={gamma} | Val Acc={val_acc:.3f}, Val F1={val_f1:.3f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = (C, gamma)
                # 在测试集上评估
                ypred_test = clf.predict(Xte_s)
                test_acc = accuracy_score(yte, ypred_test)
                test_f1  = f1_score(yte, ypred_test, average="macro")
                cm = confusion_matrix(yte, ypred_test)
                best_test_metrics = (test_acc, test_f1, cm)

    # 保存验证集结果
    csv_path = os.path.join(results_dir, "svm_gridsearch_val.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["C", "gamma", "Val_Acc", "Val_F1"])
        writer.writerows(results)

    # 保存最佳模型的混淆矩阵
    if best_params and best_test_metrics:
        test_acc, test_f1, cm = best_test_metrics
        plot_confusion(cm, classes, f"SVM Best C={best_params[0]}, gamma={best_params[1]}",
                       os.path.join(results_dir, "SVM_best_cm.png"))
        print(f"[SVM-Grid] 最佳参数: C={best_params[0]}, gamma={best_params[1]} | Test Acc={test_acc:.3f}, Test F1={test_f1:.3f}")
        return best_params, (test_acc, test_f1)
    else:
        return None, None

def run_logreg_gridsearch(Xtr, ytr, Xval, yval, Xte, yte, classes, results_dir):
    C_values = [0.01, 0.1, 1, 10, 100]
    penalties = ["l2"]  

    results = []
    best_f1 = -1
    best_params = None
    best_test_metrics = None

    for C in C_values:
        for penalty in penalties:
            clf = LogisticRegression(C=C, penalty=penalty, solver="lbfgs", max_iter=500)
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xval_s = scaler.transform(Xval)
            Xte_s  = scaler.transform(Xte)

            clf.fit(Xtr_s, ytr)
            ypred_val = clf.predict(Xval_s)

            val_acc = accuracy_score(yval, ypred_val)
            val_f1  = f1_score(yval, ypred_val, average="macro")
            results.append([C, penalty, val_acc, val_f1])
            print(f"[LogReg-Grid] C={C}, penalty={penalty} | Val Acc={val_acc:.3f}, Val F1={val_f1:.3f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = (C, penalty)
                ypred_test = clf.predict(Xte_s)
                test_acc = accuracy_score(yte, ypred_test)
                test_f1  = f1_score(yte, ypred_test, average="macro")
                cm = confusion_matrix(yte, ypred_test)
                best_test_metrics = (test_acc, test_f1, cm)

    # 保存验证集结果
    csv_path = os.path.join(results_dir, "logreg_gridsearch_val.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["C", "penalty", "Val_Acc", "Val_F1"])
        writer.writerows(results)

    if best_params and best_test_metrics:
        test_acc, test_f1, cm = best_test_metrics
        plot_confusion(cm, classes, f"LogReg Best C={best_params[0]}, penalty={best_params[1]}",
                       os.path.join(results_dir, "LogReg_best_cm.png"))
        print(f"[LogReg-Grid] 最佳参数: C={best_params[0]}, penalty={best_params[1]} | Test Acc={test_acc:.3f}, Test F1={test_f1:.3f}")
        return best_params, (test_acc, test_f1)
    else:
        return None, None

def run_mlp_gridsearch(Xtr, ytr, Xval, yval, Xte, yte, classes, results_dir):
    hidden_sizes = [(128,), (256,), (512,), (256,128)]
    max_iters = [200, 500]
    alphas = [1e-4, 1e-3]

    results = []
    best_f1 = -1
    best_params = None
    best_test_metrics = None

    for h in hidden_sizes:
        for it in max_iters:
            for a in alphas:
                clf = MLPClassifier(hidden_layer_sizes=h, max_iter=it, alpha=a, activation="relu", solver="adam")
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                Xval_s = scaler.transform(Xval)
                Xte_s  = scaler.transform(Xte)

                clf.fit(Xtr_s, ytr)
                ypred_val = clf.predict(Xval_s)

                val_acc = accuracy_score(yval, ypred_val)
                val_f1  = f1_score(yval, ypred_val, average="macro")
                results.append([h, it, a, val_acc, val_f1])
                print(f"[MLP-Grid] hidden={h}, max_iter={it}, alpha={a} | Val Acc={val_acc:.3f}, Val F1={val_f1:.3f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_params = (h, it, a)
                    ypred_test = clf.predict(Xte_s)
                    test_acc = accuracy_score(yte, ypred_test)
                    test_f1  = f1_score(yte, ypred_test, average="macro")
                    cm = confusion_matrix(yte, ypred_test)
                    best_test_metrics = (test_acc, test_f1, cm)

    # 保存验证集结果
    csv_path = os.path.join(results_dir, "mlp_gridsearch_val.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden_layer_sizes", "max_iter", "alpha", "Val_Acc", "Val_F1"])
        writer.writerows(results)

    if best_params and best_test_metrics:
        test_acc, test_f1, cm = best_test_metrics
        plot_confusion(cm, classes, f"MLP Best {best_params}",
                       os.path.join(results_dir, "MLP_best_cm.png"))
        print(f"[MLP-Grid] 最佳参数: {best_params} | Test Acc={test_acc:.3f}, Test F1={test_f1:.3f}")
        return best_params, (test_acc, test_f1)
    else:
        return None, None
"""
#4主程序
if __name__=="__main__":
    TRAIN_ROOT="data/PlantDoc/TRAIN"
    TEST_ROOT="data/PlantDoc/TEST"
    RESULTS_DIR="data/PlantDoc/传统results"
    os.makedirs(RESULTS_DIR,exist_ok=True)
    #加载数据
    Xtr_paths,Xval_paths,ytr,yval,classes=split_train_val(TRAIN_ROOT)
    Xte_paths,yte,_=list_images(TEST_ROOT)
    results=[]
    """
    print("开始 SIFT-BoVW 网格调参...")
    best_dict, best_metrics = run_bovw_gridsearch(
    Xtr_paths, Xval_paths, Xte_paths, ytr, yval, yte, classes, RESULTS_DIR,
    dict_sizes=[128,256,512]
    )
    if best_dict and best_metrics:
        test_acc, test_f1 = best_metrics
        results.append([f"SIFT-BoVW+SVM dict={best_dict}", test_acc, test_f1])
    """
    """
    print("提取 HOG 特征...")
    Xtr=[hog_feature(preprocess_image(p)) for p in Xtr_paths]
    Xval=[hog_feature(preprocess_image(p)) for p in Xval_paths]
    Xte=[hog_feature(preprocess_image(p)) for p in Xte_paths]
    print("开始 MLP 网格调参...")
    best_params, best_metrics = run_mlp_gridsearch(
    np.array(Xtr), ytr,
    np.array(Xval), yval,
    np.array(Xte), yte,
    classes, RESULTS_DIR
    )
    if best_params and best_metrics:
        test_acc, test_f1 = best_metrics
        results.append([f"MLP Best {best_params}", test_acc, test_f1])

    print("开始 Logistic Regression 网格调参...")
    best_params, best_metrics = run_logreg_gridsearch(
    np.array(Xtr), ytr,
    np.array(Xval), yval,
    np.array(Xte), yte,
    classes, RESULTS_DIR
    )

    if best_params and best_metrics:
        test_acc, test_f1 = best_metrics
        results.append([f"LogReg Best (C={best_params[0]}, penalty={best_params[1]})", test_acc, test_f1])
    """
    """
    print("开始 SVM 网格调参...")
    best_params, best_metrics = run_svm_gridsearch(
    np.array(Xtr), ytr,
    np.array(Xval), yval,
    np.array(Xte), yte,
    classes, RESULTS_DIR
    )
    if best_params and best_metrics:
        test_acc, test_f1 = best_metrics
        results.append([f"SVM-RBF Best (C={best_params[0]}, gamma={best_params[1]})", test_acc, test_f1])
    """

    """print("开始 HOG 网格调参（固定 SVM: C=0.1, gamma='scale'）...")
    best_hog_cfg, hog_results, best_hog_test_metrics = run_hog_gridsearch(
    Xtr_paths, Xval_paths, Xte_paths, ytr, yval, yte, classes, RESULTS_DIR
    )

    # 汇总到 results（用于统一 CSV 输出）
    if best_hog_cfg is not None and best_hog_test_metrics is not None:
       test_acc, test_f1 = best_hog_test_metrics
       results.append([f"HOG+SVM-RBF (Best {best_hog_cfg})", test_acc, test_f1])
    """

    #组合1-3 HOG+三种分类器
    print("提取 HOG 特征...")
    Xtr=[hog_feature(preprocess_image(p)) for p in Xtr_paths]
    Xval=[hog_feature(preprocess_image(p)) for p in Xval_paths]
    Xte=[hog_feature(preprocess_image(p)) for p in Xte_paths]

    for clf in ["SVM-RBF","LogReg","MLP"]:
        acc,f1,cm,ypred=train_and_eval(np.array(Xtr),ytr,np.array(Xval),yval,np.array(Xte),yte,clf_name=clf)
        results.append(["HOG+"+clf,acc,f1])
        plot_confusion(cm, classes, f"HOG+{clf}", os.path.join(RESULTS_DIR,f"HOG_{clf}_cm.png"))
        print(f"HOG+{clf}: Acc={acc:.3f}, F1={f1:.3f}")
    
   #组合4 SIFT-BoVM+SVM
    print("构建 SIFT-BoVW 词典...")
    centers=build_bovw_dictionary(Xtr_paths,detector="SIFT",dict_size=512)
    Xtr=[bovw_feature(preprocess_image(p),centers,"SIFT") for p in Xtr_paths]
    Xval=[bovw_feature(preprocess_image(p),centers,"SIFT") for p in Xval_paths]
    Xte=[bovw_feature(preprocess_image(p),centers,"SIFT") for p in Xte_paths]
    acc,f1,cm,ypred=train_and_eval(np.array(Xtr),ytr,np.array(Xval),yval,np.array(Xte),yte,clf_name="SVM-RBF")
    results.append(["SIFT-BoVW+SVM-RBF",acc,f1])
    plot_confusion(cm, classes, "SIFT-BoVW+SVM-RBF", os.path.join(RESULTS_DIR,"SIFT_SVM_cm.png"))
    print(f"SIFT-BoVW+SVM-RBF: Acc={acc:.3f}, F1={f1:.3f}")
 
    """
    #组合5 SURF-BoVW+SVM
    print("构建 SURF-BoVW 词典...")
    centers_surf = build_bovw_dictionary(Xtr_paths, detector="SURF", dict_size=256)
    Xtr_surf = [bovw_feature(preprocess_image(p), centers_surf, "SURF") for p in Xtr_paths]
    Xval_surf = [bovw_feature(preprocess_image(p), centers_surf, "SURF") for p in Xval_paths]
    Xte_surf = [bovw_feature(preprocess_image(p), centers_surf, "SURF") for p in Xte_paths]
    acc_surf, f1_surf, cm_surf, ypred_surf = train_and_eval( np.array(Xtr_surf), ytr, np.array(Xval_surf), yval, np.array(Xte_surf), yte,clf_name="SVM-RBF")
    results.append(["SURF-BoVW+SVM-RBF", acc_surf, f1_surf])
    plot_confusion(cm_surf, classes, "SURF-BoVW+SVM-RBF",os.path.join(RESULTS_DIR, "SURF_SVM_cm.png"))
    print(f"SURF-BoVW+SVM-RBF: Acc={acc_surf:.3f}, F1={f1_surf:.3f}")
    """
    #保存结果
    csv_path=os.path.join(RESULTS_DIR,"traditional_results.csv")
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["Method","Test Accuracy","Test F1"])
        writer.writerows(results)

    print("实验完成，结果已保存")