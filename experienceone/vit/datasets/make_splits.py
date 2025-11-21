
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
import glob

def get_image_paths_and_labels(data_dir):
    """
    获取指定目录下所有图片路径及其对应的标签。
    标签是图片所在的子目录名。
    """
    image_paths = []
    labels = []
    
    # 获取所有子目录作为类别标签
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Found {len(class_dirs)} classes in '{data_dir}'.")
    
    for label in tqdm(class_dirs, desc="Processing classes"):
        class_path = os.path.join(data_dir, label)
        # 支持多种常见图片格式
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif'):
            # glob.glob返回的是绝对路径
            paths = glob.glob(os.path.join(class_path, ext))
            for path in paths:
                # 将绝对路径转换为相对于项目根目录的相对路径
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                relative_path = os.path.relpath(path, start=project_root)
                image_paths.append(relative_path.replace('\\', '/')) # 统一使用 / 作为路径分隔符
                labels.append(label)
                
    return image_paths, labels

def make_splits(data_root, output_dir, val_ratio, seed):
    """
    划分数据集并生成 train.csv, val.csv, test.csv 文件。
    - 使用 data_root/TRAIN 进行训练集和验证集的划分。
    - 使用 data_root/TEST 作为测试集。
    """
    train_val_dir = os.path.join(data_root, 'TRAIN')
    test_dir = os.path.join(data_root, 'TEST')

    if not os.path.exists(train_val_dir):
        raise FileNotFoundError(f"TRAIN directory not found at: {train_val_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"TEST directory not found at: {test_dir}")

    # 1. 处理训练集和验证集
    print("Processing TRAIN directory for train/validation splits...")
    train_val_images, train_val_labels = get_image_paths_and_labels(train_val_dir)
    
    if len(train_val_images) == 0:
        print(f"Warning: No images found in {train_val_dir}. Cannot create train/val splits.")
        return

    # 使用分层抽样划分训练集和验证集
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_val_images,
        train_val_labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val_labels  # 保证划分后各类别比例不变
    )

    train_df = pd.DataFrame({'image_path': train_images, 'label': train_labels})
    val_df = pd.DataFrame({'image_path': val_images, 'label': val_labels})

    # 2. 处理测试集
    print("\nProcessing TEST directory for test split...")
    test_images, test_labels = get_image_paths_and_labels(test_dir)
    
    if len(test_images) == 0:
        print(f"Warning: No images found in {test_dir}. Cannot create test split.")
        return
        
    test_df = pd.DataFrame({'image_path': test_images, 'label': test_labels})

    # 3. 保存到 CSV 文件
    os.makedirs(output_dir, exist_ok=True)
    train_csv_path = os.path.join(output_dir, 'train.csv')
    val_csv_path = os.path.join(output_dir, 'val.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print("\n--- Data Split Summary ---")
    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples:       {len(test_df)}")
    print("--------------------------")
    print(f"Splits saved to: {os.path.abspath(output_dir)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create train/val/test splits for PlantDoc dataset.")
    parser.add_argument('--data-root', type=str, default='data/PlantDoc',
                        help='Path to the root directory of the PlantDoc dataset (containing TRAIN and TEST folders).')
    parser.add_argument('--output-dir', type=str, default='data/splits',
                        help='Directory to save the output csv files.')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Proportion of the TRAIN data to use for validation (default 0.15 to align with 70/15/15 guidance).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits.')
    
    args = parser.parse_args()

    make_splits(
        data_root=args.data_root,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
