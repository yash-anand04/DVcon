"""
preprocess_dataset.py — TORCA Dataset Preprocessor
====================================================
Converts the raw COCO-Tasks dataset into the format expected by train.py.

Raw dataset layout (--data-root):
    <data_root>/
        train/
            images/        (original COCO images)
            annotations/   (task_1_train.json … task_14_train.json)
        test/
            images/
            annotations/   (task_1_test.json … task_14_test.json)

Output layout (--out-root):
    <out_root>/
        train/
            images/        (640×640 letterboxed JPGs + colour-jitter augmentations)
            samples.json   (index consumed by COCOTasksDataset)
        test/
            images/
            samples.json

Usage:
    python preprocess_dataset.py --data-root /path/to/Data --out-root /path/to/Data_Preprocessed
    python preprocess_dataset.py --data-root /path/to/Data --out-root /path/to/Data_Preprocessed --img-size 640
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

THIS = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────
#  Raw sample loader
# ─────────────────────────────────────────────────────────────────────
def load_raw_samples(data_root: str, split: str):
    img_dir = os.path.join(data_root, split, "images")
    ann_dir = os.path.join(data_root, split, "annotations")

    samples = []
    for task_id in range(1, 15):
        ann_file = os.path.join(ann_dir, f"task_{task_id}_{split}.json")
        if not os.path.exists(ann_file):
            continue

        with open(ann_file) as f:
            data = json.load(f)

        ann_by_img = {}
        for ann in data.get("annotations", []):
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_info in data["images"]:
            img_id   = img_info["id"]
            img_path = os.path.join(img_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
            samples.append({
                "task_id":     task_id,
                "image_path":  img_path,
                "img_id":      img_id,
                "annotations": ann_by_img.get(img_id, []),
            })

    return samples


# ─────────────────────────────────────────────────────────────────────
#  Letterbox resize (preserves aspect ratio, pads with grey)
# ─────────────────────────────────────────────────────────────────────
def letterbox(img, boxes, size=640):
    h, w = img.shape[:2]
    r    = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2

    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top,    bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left,   right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))

    new_boxes = []
    for (x, y, bw, bh) in boxes:
        new_boxes.append([x * r + left, y * r + top, bw * r, bh * r])

    return img, new_boxes


# ─────────────────────────────────────────────────────────────────────
#  Colour-jitter augmentation (train only)
# ─────────────────────────────────────────────────────────────────────
def colour_jitter(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    t = TF.adjust_brightness(t, random.uniform(0.7, 1.3))
    t = TF.adjust_contrast(t,   random.uniform(0.7, 1.3))
    t = TF.adjust_saturation(t, random.uniform(0.7, 1.3))
    t = t.clamp(0, 1)
    img_rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def preprocess(data_root: str, out_root: str, img_size: int = 640):
    print(f"Source  : {data_root}")
    print(f"Output  : {out_root}")
    print(f"Img size: {img_size}×{img_size}")

    for split in ("train", "test"):
        print(f"\n── {split} ──")
        samples = load_raw_samples(data_root, split)
        if not samples:
            print(f"  No annotations found — skipping.")
            continue

        img_out = os.path.join(out_root, split, "images")
        os.makedirs(img_out, exist_ok=True)

        index = []

        for s in tqdm(samples):
            img = cv2.imread(s["image_path"])
            if img is None:
                continue

            raw_boxes = [ann["bbox"]        for ann in s["annotations"]]
            prefs     = [ann["category_id"] for ann in s["annotations"]]

            img_lb, boxes_lb = letterbox(img, raw_boxes, img_size)

            stem      = f"{s['task_id']}_{s['img_id']}"
            save_path = os.path.join(img_out, f"{stem}_norm.jpg")
            cv2.imwrite(save_path, img_lb)

            index.append({
                "task_id":    s["task_id"],
                "image_path": save_path,
                "boxes":      boxes_lb,
                "prefs":      prefs,
            })

            # Colour-jitter augmentation for train only
            if split == "train":
                img_aug      = colour_jitter(img_lb)
                aug_path     = os.path.join(img_out, f"{stem}_aug.jpg")
                cv2.imwrite(aug_path, img_aug)
                index.append({
                    "task_id":    s["task_id"],
                    "image_path": aug_path,
                    "boxes":      boxes_lb,
                    "prefs":      prefs,
                })

        idx_path = os.path.join(out_root, split, "samples.json")
        with open(idx_path, "w") as f:
            json.dump(index, f)
        print(f"  {len(index):,} samples  →  {idx_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess COCO-Tasks for TORCA training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True,
                        help="Root of the raw COCO-Tasks dataset (contains train/ and test/).")
    parser.add_argument("--out-root",  required=True,
                        help="Where to write the preprocessed dataset.")
    parser.add_argument("--img-size",  type=int, default=640,
                        help="Target image size (square).")
    args = parser.parse_args()
    preprocess(args.data_root, args.out_root, args.img_size)
