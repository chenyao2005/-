import os
import sys
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Ensure repository root is on sys.path for module imports when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.plantdoc_dataset import PlantDocDataset, get_transforms
from models.vit_model import create_vit_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal ViT training script for PlantDoc")
    parser.add_argument("--train-csv", default="data/splits/train.csv", type=str)
    parser.add_argument("--val-csv", default="data/splits/val.csv", type=str)
    parser.add_argument("--model-name", default="vit_base_patch16_224.augreg_in1k", type=str)
    parser.add_argument('--freeze-mode', type=str, default='full',
                        choices=['full', 'head_only'],
                        help='full: unfreeze all layers; head_only: freeze body and unfreeze head')
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--label-smoothing", default=0.1, type=float, help="Label smoothing factor")
    parser.add_argument("--num-classes", default=27, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument('--no-basic-aug', action='store_true', help='Disable basic augmentation')
    parser.add_argument("--use-randaugment", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--save-dir", default="outputs/checkpoints/min_vit", type=str)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--max-train-batches", default=0, type=int, help="limit number of train batches per epoch (0 = no limit)")
    parser.add_argument("--max-val-batches", default=0, type=int, help="limit number of validation batches (0 = no limit)")
    # Scheduler
    parser.add_argument("--sched", default="cosine", choices=["none", "cosine"], help="LR scheduler type")
    parser.add_argument("--warmup-epochs", default=3, type=int, help="number of warmup epochs")
    parser.add_argument("--min-lr", default=1e-6, type=float, help="min learning rate for cosine annealing")
    return parser.parse_args()


def build_dataloaders(args):
    transforms_dict = get_transforms(
        use_basic_aug=not args.no_basic_aug,
        use_randaugment=args.use_randaugment
    )

    train_ds = PlantDocDataset(csv_file=args.train_csv, transform=transforms_dict['train'])
    val_ds = PlantDocDataset(csv_file=args.val_csv, transform=transforms_dict['val'], label_map=train_ds.label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Build class names from label_map (index -> name ordered)
    inv_map = {v: k for k, v in train_ds.label_map.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]

    return train_loader, val_loader, class_names


def evaluate(model, loader, device, max_batches: int = 0):
    model.eval()
    all_preds = []
    all_labels = []
    top3_correct = 0
    total_samples = 0
    with torch.no_grad():
        for b_idx, (images, labels) in enumerate(loader, start=1):
            if images is None:
                continue
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            # top-3 accuracy
            if logits.ndim == 2 and logits.size(1) >= 3:
                top3 = torch.topk(logits, k=min(3, logits.size(1)), dim=1).indices
                top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += labels.size(0)
            if max_batches and b_idx >= max_batches:
                break
    if not all_preds:
        return 0.0, 0.0, 0.0, 0.0
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    top3_acc = (top3_correct / total_samples) if total_samples else 0.0
    return acc, f1_macro, f1_micro, top3_acc


def evaluate_collect(model, loader, device):
    """Collect all predictions and labels for report/confusion."""
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            if images is None:
                continue
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())
    if not preds_all:
        return np.array([]), np.array([])
    return torch.cat(preds_all).numpy(), torch.cat(labels_all).numpy()


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, max_batches: int = 0):
    model.train()
    total_loss = 0.0
    seen = 0
    for b_idx, (images, labels) in enumerate(loader, start=1):
        if images is None:
            continue
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        seen += bs
        if max_batches and b_idx >= max_batches:
            break
    denom = max(seen, 1)
    return total_loss / denom


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {args.model_name}, Freeze mode: {args.freeze_mode}")

    # Data
    train_loader, val_loader, class_names = build_dataloaders(args)
    args.num_classes = len(class_names)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model, trainable_params = create_vit_model(model_name=args.model_name,
                                               num_classes=args.num_classes,
                                               pretrained=True,
                                               freeze_mode=args.freeze_mode)
    model.to(device)
    print(f"Trainable parameters count: {trainable_params}")

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler (cosine with warmup)
    scheduler = None
    if args.sched == 'cosine':
        warmup_epochs = max(0, int(args.warmup_epochs))
        total_epochs = max(1, int(args.epochs))
        cosine_tmax = max(1, total_epochs - warmup_epochs)

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_tmax, eta_min=args.min_lr)

        if warmup_epochs > 0:
            # Linear warmup followed by cosine decay
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            # No warmup; use cosine directly
            scheduler = cosine_scheduler

    scaler = torch.amp.GradScaler('cuda', enabled=(args.use_amp and torch.cuda.is_available()))

    best_val_acc = -1.0
    best_ckpt_path = None

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.use_amp, args.max_train_batches)
        val_acc, val_f1_macro, val_f1_micro, val_top3 = evaluate(model, val_loader, device, args.max_val_batches)
        elapsed = time.time() - start_t
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{args.epochs} - lr: {current_lr:.6g} | loss: {train_loss:.4f} | val_acc: {val_acc:.4f} | val_f1_macro: {val_f1_macro:.4f} | val_f1_micro: {val_f1_micro:.4f} | top3: {val_top3:.4f} | time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_name = f"best_epoch{epoch}_acc{val_acc:.4f}.pth"
            best_ckpt_path = os.path.join(args.save_dir, ckpt_name)
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc, 'val_f1_macro': val_f1_macro, 'val_f1_micro': val_f1_micro, 'val_top3': val_top3, 'args': vars(args)}, best_ckpt_path)
            print(f"  -> Saved new best checkpoint: {best_ckpt_path}")

            # Generate and save classification report & confusion matrix for best model
            preds_np, labels_np = evaluate_collect(model, val_loader, device)
            if preds_np.size and labels_np.size:
                report = classification_report(labels_np, preds_np, target_names=class_names, digits=4, zero_division=0)
                report_path = Path(args.save_dir) / "classification_report.txt"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                cm = confusion_matrix(labels_np, preds_np)
                cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
                cm_path = Path(args.save_dir) / "confusion_matrix.csv"
                cm_df.to_csv(cm_path, encoding='utf-8')
                print(f"  -> Saved classification_report and confusion_matrix to {args.save_dir}")

        # Step scheduler at epoch end
        if scheduler is not None:
            scheduler.step()

    # Final summary
    metrics_path = Path(args.save_dir) / "run_summary.txt"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f"best_val_acc={best_val_acc}\n")
        f.write(f"best_ckpt={best_ckpt_path}\n")
    print(f"Training finished. Best val_acc={best_val_acc:.4f}. Summary written to {metrics_path}")


if __name__ == "__main__":
    main()
