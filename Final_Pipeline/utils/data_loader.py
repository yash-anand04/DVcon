"""
COCOTasksDataset — loads pre-processed COCO-Tasks samples.

Expected data_root layout:
    <data_root>/
        train/
            images/          (640×640 JPGs, named <task_id>_<image_id>_norm.jpg)
            samples.json     (list of sample dicts, produced by preprocess_dataset.py)
        test/
            images/
            samples.json

Each entry in samples.json must contain:
    image_path : absolute or relative path to the 640×640 image
    task_id    : int 1–14
    boxes      : list of [x, y, w, h] GT bounding boxes (COCO format)
    prefs      : list of 0/1 flags — 1 marks the preferred (GT) box
"""

import json
import os

import cv2
import torch
from torch.utils.data import Dataset


class COCOTasksDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train"):
        """
        Args:
            data_root : path to the preprocessed dataset root (contains train/ and test/)
            split     : "train" or "test"
        """
        split_dir  = os.path.join(data_root, split)
        index_path = os.path.join(split_dir, "samples.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Dataset index not found: {index_path}\n"
                f"Run preprocess_dataset.py first, or check --data-root."
            )

        with open(index_path) as f:
            self.samples = json.load(f)

        print(f"[COCOTasksDataset] {split}: {len(self.samples):,} samples from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = cv2.imread(s["image_path"])
        if img is None:
            raise ValueError(f"Cannot load image: {s['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        boxes = s["boxes"]
        prefs = s["prefs"]

        return {
            "image":      img_tensor,
            "task_id":    torch.tensor(s["task_id"], dtype=torch.long),
            "boxes":      torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "prefs":      torch.tensor(prefs, dtype=torch.long)    if prefs else torch.zeros((0,), dtype=torch.long),
            "image_path": s["image_path"],
        }


def custom_collate(batch):
    """Collate fn that handles variable-length bounding-box lists."""
    return {
        "image":       torch.stack([b["image"]   for b in batch]),
        "task_id":     torch.stack([b["task_id"] for b in batch]),
        "boxes":       [b["boxes"]      for b in batch],
        "prefs":       [b["prefs"]      for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }
