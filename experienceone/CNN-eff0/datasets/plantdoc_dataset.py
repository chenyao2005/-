from pathlib import Path
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


def _read_label_map(label_map_path):
    with open(label_map_path, 'r', encoding='utf-8') as f:
        lm = json.load(f)
    # accept both {class_name: id} or list of names
    if isinstance(lm, list):
        return {name: i for i, name in enumerate(lm)}
    return lm


class PlantDocCSVDataset(Dataset):
    def __init__(self, csv_path, image_root, label_map_path, transforms=None, limit=None):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.df = pd.read_csv(self.csv_path)
        if limit is not None and limit > 0:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
        self.transforms = transforms
        self.label_map = _read_label_map(label_map_path) if label_map_path else None
        self.label_map_inv = None
        if self.label_map:
            self.label_map_inv = {v: k for k, v in self.label_map.items()}

        # normalize column names
        cols = {c.lower(): c for c in self.df.columns}
        self.col_image = cols.get('image') or cols.get('path') or list(self.df.columns)[0]
        self.col_label = cols.get('label') or cols.get('class')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel = str(row[self.col_image])
        p = Path(rel)
        if not p.is_absolute():
            # avoid double prefix if CSV already includes image_root
            try:
                rel_str = str(p).replace('\\', '/')
                root_str = str(self.image_root).replace('\\', '/')
                if rel_str.startswith(root_str.rstrip('/') + '/'):  # already rooted
                    p = Path(rel_str)
                else:
                    p = self.image_root / rel
            except Exception:
                p = self.image_root / rel
        img = Image.open(p).convert('RGB')

        # resolve label
        label = None
        if self.col_label is not None:
            label_raw = row[self.col_label]
            if isinstance(label_raw, str):
                # try map by name
                if self.label_map and label_raw in self.label_map:
                    label = self.label_map[label_raw]
                else:
                    # maybe numeric string
                    try:
                        label = int(label_raw)
                    except Exception:
                        # fallback to parent folder
                        label_name = p.parent.name
                        label = self.label_map[label_name] if self.label_map else -1
            else:
                # numeric label
                label = int(label_raw)
        else:
            # no label column -> infer from folder name
            label_name = p.parent.name
            label = self.label_map[label_name] if self.label_map else -1

        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(label, dtype=torch.long)
