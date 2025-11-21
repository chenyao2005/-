
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PlantDocDataset(Dataset):
    """
    PlantDoc 数据集的 PyTorch Dataset 类。
    从 CSV 文件读取图像路径和标签，并应用指定的变换。
    """
    def __init__(self, csv_file, transform=None, label_map=None):
        """
        Args:
            csv_file (string): 包含图像路径和标签的 CSV 文件路径。
            transform (callable, optional): 应用于样本的可选变换。
            label_map (dict, optional): 将字符串标签映射到整数索引的字典。
                                        如果为 None，将根据 CSV 文件自动创建。
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

        # 创建或使用标签映射
        if label_map is None:
            self.labels = sorted(self.data_frame['label'].unique())
            self.label_map = {label: i for i, label in enumerate(self.labels)}
        else:
            self.label_map = label_map
            self.labels = sorted(label_map.keys())

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # CSV 中的路径是相对于项目根目录的，所以我们直接使用
        img_path = self.data_frame.iloc[idx, 0]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path}. Please check the paths in your CSV files.")
            # 返回一个占位符或引发异常
            return None, None
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None


        label_str = self.data_frame.iloc[idx, 1]
        label_idx = self.label_map[label_str]
        label = torch.tensor(label_idx, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms(image_size=224, use_basic_aug=True, use_randaugment=False):
    """
    根据计划，为训练、验证和测试集获取图像变换。

    Args:
        image_size (int): 模型的输入图像尺寸。
        use_basic_aug (bool): 是否在训练时使用基础数据增强。
        use_randaugment (bool): 是否在训练时使用 RandAugment。

    Returns:
        dict: 包含 'train', 'val', 'test' 的变换字典。
    """
    # ImageNet 的均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 训练集变换
    train_transform_list = []
    
    if use_basic_aug:
        train_transform_list.extend([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        print("Using basic augmentations for training.")
    else:
        # 如果没有基础增强，则使用与验证集/测试集相同的调整大小和裁剪策略
        train_transform_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ])
        print("Not using basic augmentations for training.")
    
    if use_randaugment:
        # RandAugment 通常在 ToTensor 之前应用
        train_transform_list.append(transforms.RandAugment())
        print("Using RandAugment for training.")

    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    train_transforms = transforms.Compose(train_transform_list)

    # 验证集和测试集的变换应该相同
    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return {
        'train': train_transforms,
        'val': val_test_transforms,
        'test': val_test_transforms
    }

if __name__ == '__main__':
    # 这是一个简单的使用示例和测试
    
    # 1. 获取变换
    data_transforms = get_transforms(use_randaugment=True)
    
    # 2. 创建数据集实例
    # 假设脚本在 project/ 目录下运行
    train_csv = 'data/splits/train.csv'
    
    try:
        train_dataset = PlantDocDataset(csv_file=train_csv, transform=data_transforms['train'])
        
        # 3. 打印一些信息
        print(f"\nSuccessfully created dataset from '{train_csv}'.")
        print(f"Number of training samples: {len(train_dataset)}")
        
        # 4. 获取一个样本并检查其形状和类型
        image, label = train_dataset[0]
        print(f"Sample 0: Image shape: {image.shape}, Label: {label.item()}")
        print(f"Image dtype: {image.dtype}, Label dtype: {label.dtype}")
        
        # 5. 检查标签映射
        print(f"\nTotal number of classes: {len(train_dataset.label_map)}")
        print("Label map (first 5):")
        for i, (k, v) in enumerate(train_dataset.label_map.items()):
            if i >= 5:
                break
            print(f"  '{k}': {v}")

    except FileNotFoundError as e:
        print(f"\nTest run failed: {e}")
        print("Please make sure you are running this script from the project's root directory ('PlantDoctor/').")
    except Exception as e:
        print(f"An error occurred during the test run: {e}")

