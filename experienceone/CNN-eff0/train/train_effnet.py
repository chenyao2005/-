import os
from pathlib import Path
import math
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

from datasets.plantdoc_dataset import PlantDocCSVDataset
from models.build_model import build_model
from utils.metrics import compute_metrics, get_confusion_matrix, get_classification_report


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        focal = (self.alpha * (1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


def set_seed(seed: int):
    if seed is None:
        return
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(cfg):
    img_size = cfg['augment'].get('img_size', 224)
    val_resize = cfg['augment'].get('val_resize', img_size + 32)
    val_center = cfg['augment'].get('val_center_crop', img_size)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = []
    if cfg['augment'].get('basic', True):
        train_tfms.append(transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)))
        hflip_p = cfg['augment'].get('hflip', 0.5)
        if hflip_p > 0:
            train_tfms.append(transforms.RandomHorizontalFlip(p=hflip_p))
        cj = cfg['augment'].get('color_jitter', [0.2, 0.2, 0.2, 0.1])
        if any(v > 0 for v in cj):
            train_tfms.append(transforms.ColorJitter(*cj))
    else:
        train_tfms.append(transforms.Resize(val_resize))
        train_tfms.append(transforms.CenterCrop(val_center))

    if cfg['augment'].get('randaugment', False):
        try:
            train_tfms.append(transforms.RandAugment())
        except Exception:
            pass

    train_tfms += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    val_tfms = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(val_center),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transforms.Compose(train_tfms), val_tfms


def _build_dataloaders(cfg, train_tfms, val_tfms):
    data = cfg['data']
    train_ds = PlantDocCSVDataset(data['train_csv'], data['image_root'], data.get('label_map'), transforms=train_tfms, limit=data.get('limit_train'))
    val_ds = PlantDocCSVDataset(data['val_csv'], data['image_root'], data.get('label_map'), transforms=val_tfms, limit=data.get('limit_val'))
    test_ds = PlantDocCSVDataset(data['test_csv'], data['image_root'], data.get('label_map'), transforms=val_tfms, limit=data.get('limit_test'))

    batch_size = data.get('batch_size', 32)
    num_workers = data.get('num_workers', 4)

    if data.get('balanced_sampler', False):
        # compute class weights from train set
        labels = []
        for i in range(len(train_ds)):
            _, y = train_ds[i]
            labels.append(int(y))
        import numpy as np
        labels = np.array(labels)
        class_counts = np.bincount(labels, minlength=data['num_classes'])
        class_counts[class_counts == 0] = 1
        weights = 1.0 / class_counts
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def _select_device(cfg):
    dev = cfg.get('device', 'cuda')
    if dev == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return dev


def _make_criterion(cfg):
    loss_cfg = cfg.get('loss', {})
    loss_name = (loss_cfg.get('name') or 'cross_entropy').lower()
    if loss_name == 'focal':
        gamma = loss_cfg.get('gamma', 2.0)
        alpha = loss_cfg.get('alpha', 0.25)
        return FocalLoss(gamma=gamma, alpha=alpha)
    # cross entropy
    label_smoothing = cfg.get('augment', {}).get('label_smoothing', 0.0) or 0.0
    try:
        return nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    except Exception:
        return nn.CrossEntropyLoss()


def _save_history(exp_dir: Path, history: List[Dict]):
    if not history:
        return
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(exp_dir / 'history.csv', index=False)


def _save_outputs(cfg, exp_dir: Path, y_true, y_pred, metrics: Dict[str, float], test_dataset=None):
    exp_dir.mkdir(parents=True, exist_ok=True)
    # confusion matrix
    import numpy as np
    cm = get_confusion_matrix(y_true, y_pred)
    import pandas as pd
    pd.DataFrame(cm).to_csv(exp_dir / 'confusion_matrix.csv', index=False)
    # classification report
    report = get_classification_report(y_true, y_pred)
    with open(exp_dir / 'classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    # save predictions with image paths
    pred_data = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    # Add image paths if test_dataset is available
    if test_dataset is not None and hasattr(test_dataset, 'df'):
        # Extract image paths from dataset (assuming DataLoader maintains order with shuffle=False)
        image_paths = test_dataset.df[test_dataset.col_image].tolist()
        if len(image_paths) == len(y_true):
            pred_data['image_path'] = image_paths
        else:
            print(f"[WARN] 图像路径数量 ({len(image_paths)}) 与预测结果数量 ({len(y_true)}) 不匹配，跳过图像路径")
    else:
        # Try to read from test CSV as fallback
        try:
            test_csv = Path(cfg['data']['test_csv'])
            if test_csv.exists():
                test_df = pd.read_csv(test_csv)
                # Find image column
                cols = {c.lower(): c for c in test_df.columns}
                col_image = cols.get('image') or cols.get('path') or cols.get('image_path') or test_df.columns[0]
                image_paths = test_df[col_image].tolist()
                if len(image_paths) == len(y_true):
                    pred_data['image_path'] = image_paths
        except Exception as e:
            print(f"[WARN] 无法获取图像路径: {e}")
    
    # Add label names if label_map is available
    if cfg['data'].get('label_map'):
        try:
            import json
            with open(cfg['data']['label_map'], 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            idx_to_label = {v: k for k, v in label_map.items()}
            pred_data['true_label'] = [idx_to_label.get(idx, f'class_{idx}') for idx in y_true]
            pred_data['pred_label'] = [idx_to_label.get(idx, f'class_{idx}') for idx in y_pred]
        except Exception as e:
            print(f"[WARN] 无法加载标签映射: {e}")
    
    pred_df = pd.DataFrame(pred_data)
    pred_df.to_csv(exp_dir / 'test_predictions.csv', index=False, encoding='utf-8')
    print(f"[INFO] 测试集预测结果已保存至: {exp_dir / 'test_predictions.csv'}")
    
    # metrics to stdout
    print(f"[RESULT] {metrics}")

    # append to global csv
    results_csv = Path(cfg['logging'].get('results_csv', 'outputs/experiment_results.csv'))
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {'experiment': cfg['experiment']['name']}
    row.update(metrics)
    df = pd.DataFrame([row])
    if results_csv.exists():
        df.to_csv(results_csv, mode='a', index=False, header=False)
    else:
        df.to_csv(results_csv, index=False)


def _epoch_loop(model, loader, device, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    y_true, y_pred = [], []
    pbar = tqdm(loader, leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast(device_type=device.split(':')[0], enabled=scaler.is_enabled()):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                logits = model(imgs)
                loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred


def _apply_freeze(model, freeze: bool):
    for p in model.parameters():
        p.requires_grad = not freeze


def _apply_partial_unfreeze(model, patterns: List[str]):
    import re
    # default: freeze all
    for p in model.parameters():
        p.requires_grad = False
    if not patterns:
        return
    compiled = [re.compile(pat) for pat in patterns]
    for name, p in model.named_parameters():
        if any(r.search(name) for r in compiled):
            p.requires_grad = True
    # always keep classifier trainable
    for p in model.classifier.parameters():
        p.requires_grad = True


def _build_optimizer(params, name: str, lr: float, weight_decay: float):
    name = (name or 'adamw').lower()
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer, total_epochs, warmup_epochs: int, min_lr: float):
    # simple epoch-level warmup + cosine
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # cosine for remaining epochs
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_training(cfg: Dict):
    import time
    start_time = time.time()
    
    set_seed(cfg.get('experiment', {}).get('seed', 42))
    device = _select_device(cfg)
    print(f"[INFO] Using device: {device.upper()}")
    amp_enabled = bool(cfg.get('amp', True) and device == 'cuda')

    # data
    train_tfms, val_tfms = build_transforms(cfg)
    train_loader, val_loader, test_loader = _build_dataloaders(cfg, train_tfms, val_tfms)

    # model
    mcfg = cfg['model']
    model = build_model(mcfg.get('arch'), mcfg.get('num_classes'), mcfg.get('pretrained', True), mcfg.get('drop_rate', 0.2))
    model.to(device)

    # loss
    criterion = _make_criterion(cfg)

    out_dir = Path(cfg['logging'].get('output_dir', f"outputs/checkpoints/{cfg['experiment']['name']}") )
    out_dir.mkdir(parents=True, exist_ok=True)

    # stage1: head-only
    epochs1 = cfg['train']['epochs'].get('stage1', 0)
    epochs2 = cfg['train']['epochs'].get('stage2', 0)
    stage2_enabled = cfg['train'].get('stage2_enabled', True)
    grad_clip = cfg['train'].get('grad_clip_norm', 0.0)
    optimizer_name = cfg['optimizer'].get('name', 'adamw')
    warmup_epochs = cfg['scheduler'].get('warmup_epochs', 0)
    min_lr = cfg['scheduler'].get('min_lr', 1e-6)

    scaler = torch.amp.GradScaler(device='cuda', enabled=amp_enabled)

    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    history = []

    if epochs1 > 0:
        # freeze backbone
        _apply_freeze(model, True)
        # unfreeze classifier
        for p in model.classifier.parameters():
            p.requires_grad = True
        opt1 = _build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), optimizer_name,
                                lr=cfg['optimizer']['stage1'].get('lr', 5e-4),
                                weight_decay=cfg['optimizer']['stage1'].get('weight_decay', 1e-4))
        sch1 = _build_scheduler(opt1, epochs1, warmup_epochs=min(warmup_epochs, epochs1), min_lr=min_lr)

        for epoch in range(epochs1):
            train_loss, train_metrics, _, _ = _epoch_loop(model, train_loader, device, criterion, optimizer=opt1, scaler=scaler)
            val_loss, val_metrics, y_true, y_pred = _epoch_loop(model, val_loader, device, criterion)
            sch1.step()

            epoch_metrics = {
                'stage': 'stage1', 'epoch': epoch,
                'train_loss': train_loss, 'val_loss': val_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            history.append(epoch_metrics)

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
            print(f"[Stage1][{epoch+1}/{epochs1}] train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}")

    # stage2
    if stage2_enabled and epochs2 > 0:
        if cfg['train'].get('partial_unfreeze', False):
            patterns = cfg['train'].get('unfreeze_regex', [])
            _apply_partial_unfreeze(model, patterns)
        else:
            _apply_freeze(model, False)

        lr2 = cfg['optimizer']['stage2'].get('lr') or cfg['optimizer']['stage2'].get('lr_backbone', 5e-5)
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt2 = _build_optimizer(params, optimizer_name, lr=lr2, weight_decay=cfg['optimizer']['stage2'].get('weight_decay', 1e-4))
        sch2 = _build_scheduler(opt2, epochs2, warmup_epochs=min(warmup_epochs, epochs2), min_lr=min_lr)

        for epoch in range(epochs2):
            train_loss, train_metrics, _, _ = _epoch_loop(model, train_loader, device, criterion, optimizer=opt2, scaler=scaler)
            val_loss, val_metrics, y_true, y_pred = _epoch_loop(model, val_loader, device, criterion)
            sch2.step()

            epoch_metrics = {
                'stage': 'stage2', 'epoch': epoch,
                'train_loss': train_loss, 'val_loss': val_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            history.append(epoch_metrics)

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_epoch = epochs1 + epoch
            print(f"[Stage2][{epoch+1}/{epochs2}] train_loss={train_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}")

    # load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        # save best model checkpoint
        checkpoint_dir = out_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / 'best_model.pth'
        torch.save(best_state, best_model_path)
        print(f"[INFO] Best model saved to: {best_model_path}")
    
    # save last model checkpoint
    checkpoint_dir = out_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_model_path = checkpoint_dir / 'last_model.pth'
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, last_model_path)
    print(f"[INFO] Last model saved to: {last_model_path}")

    test_loss, test_metrics, y_true, y_pred = _epoch_loop(model, test_loader, device, criterion)
    
    # Calculate training time
    end_time = time.time()
    train_time_minutes = (end_time - start_time) / 60.0
    
    # Add timing info to test_metrics
    test_metrics['train_time_minutes'] = train_time_minutes
    test_metrics['best_epoch'] = best_epoch
    
    print(f"[INFO] Training completed in {train_time_minutes:.2f} minutes")
    print(f"[INFO] Best epoch: {best_epoch}")
    
    _save_outputs(cfg, out_dir, y_true, y_pred, test_metrics, test_dataset=test_loader.dataset)
    _save_history(out_dir, history)
