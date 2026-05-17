"""
train.py — TORCA Training Script
=================================
Trains TORCA (FPN backbone + RoI-Align + task cross-attention scorer).
The YOLO detector stays frozen; the FPN backbone is fine-tuned after
freeze_epochs full epochs.

Sub-epoch validation (--chunk-frac):
  Each full epoch is split into N shards. Validation runs after every shard,
  allowing the best checkpoint to be captured before overfitting diverges.
  Example: --chunk-frac 0.25  →  4 shards per epoch.

Usage:
    python train.py --data-root /path/to/Data_Preprocessed
    python train.py --data-root /path/to/Data_Preprocessed --epochs 15 --batch 8
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Make model and utils importable from this directory
THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from model import DEFAULT_YOLO_PATH, TORCA, YOLODetector
from utils.data_loader import COCOTasksDataset, custom_collate
from utils.plot_metrics import plot_training_losses

DEFAULTS = dict(
    data_root      = os.path.join(THIS, "data"),   # override with --data-root
    weights_dir    = os.path.join(THIS, "weights"),
    yolo_weights   = DEFAULT_YOLO_PATH,
    epochs         = 15,
    batch          = 8,
    lr             = 5e-5,
    backbone_lr    = 5e-6,
    freeze_epochs  = 3,
    num_workers    = 4,
    iou_threshold  = 0.5,
    chunk_frac     = 0.25,   # 1.0 = standard full-epoch mode
)


# ─────────────────────────────────────────────────────────────────────
#  Geometry utilities
# ─────────────────────────────────────────────────────────────────────
def xywh_to_xyxy(boxes):
    out = boxes.clone()
    out[..., 2] = boxes[..., 0] + boxes[..., 2]
    out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def pairwise_iou_xyxy(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-6)


def load_pil_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


# ─────────────────────────────────────────────────────────────────────
#  Target builder — hard 0/1 labels
# ─────────────────────────────────────────────────────────────────────
def build_targets(batch, det_results, mask, iou_threshold, device):
    """
    Returns:
      targets [B, maxN] : 1 if the detected box overlaps a preferred GT at
                          >= iou_threshold, else 0.
      valid   [B, maxN] : True where the position holds a real detection.
    """
    B    = len(det_results)
    maxN = mask.shape[1]
    targets = torch.zeros(B, maxN, device=device)
    valid   = ~mask

    for b in range(B):
        gt_boxes  = batch["boxes"][b].to(device)
        prefs     = batch["prefs"][b].to(device)
        preferred = gt_boxes[prefs == 1]
        if preferred.numel() == 0:
            continue

        preferred_xyxy = xywh_to_xyxy(preferred)
        boxes_b, _, _  = det_results[b]
        if boxes_b.shape[0] == 0:
            continue

        n_b      = min(boxes_b.shape[0], maxN)
        det_xyxy = boxes_b[:n_b].to(device)
        ious     = pairwise_iou_xyxy(det_xyxy, preferred_xyxy)
        targets[b, :n_b] = (ious.max(dim=1).values >= iou_threshold).float()

    return targets, valid


# ─────────────────────────────────────────────────────────────────────
#  Optimizer — separate learning rates for backbone vs scorer heads
# ─────────────────────────────────────────────────────────────────────
def build_optimizer(model, lr, backbone_lr):
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.roi_fusion.parameters()) +
        list(model.scorer.parameters())
    )
    return optim.AdamW(
        [{"params": backbone_params, "lr": backbone_lr},
         {"params": head_params,     "lr": lr}],
        weight_decay=1e-4,
    )


def set_backbone_grad(model, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad_(requires_grad)


# ─────────────────────────────────────────────────────────────────────
#  Single forward pass (train or val)
# ─────────────────────────────────────────────────────────────────────
def run_epoch(model, detector, loader, optimizer, cfg, device, train=True):
    model.train(train)
    total_loss, total_steps = 0.0, 0
    criterion = torch.nn.functional.binary_cross_entropy

    for batch in tqdm(loader, desc="Train" if train else "Valid", leave=False):
        pil_images  = load_pil_images(batch["image_paths"])
        img_tensors = batch["image"].to(device)
        task_ids    = batch["task_id"].to(device)

        with torch.no_grad():
            det_results = detector.detect_batch(pil_images)

        rel_scores, _, mask = model(img_tensors, det_results, task_ids)
        targets, valid = build_targets(batch, det_results, mask, cfg["iou_threshold"], device)

        if not valid.any():
            continue

        loss = criterion(rel_scores[valid], targets[valid])

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss  += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


# ─────────────────────────────────────────────────────────────────────
#  Shard helper for sub-epoch validation
# ─────────────────────────────────────────────────────────────────────
def make_chunks(n_samples, chunk_frac):
    chunk_size = max(1, int(math.floor(n_samples * chunk_frac)))
    perm       = torch.randperm(n_samples).tolist()
    chunks     = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        if n_samples - end < chunk_size * 0.25 and chunks:
            chunks[-1].extend(perm[start:end])
        else:
            chunks.append(perm[start:end])
    return chunks


# ─────────────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────────────
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    print(f"Device       : {device}")
    print(f"Data root    : {cfg['data_root']}")
    print(f"Weights dir  : {cfg['weights_dir']}")

    chunk_frac = float(cfg["chunk_frac"])
    chunked    = chunk_frac < 1.0
    n_chunks   = math.ceil(1.0 / chunk_frac) if chunked else 1

    train_ds = COCOTasksDataset(cfg["data_root"], split="train")
    val_ds   = COCOTasksDataset(cfg["data_root"], split="test")

    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=custom_collate,
    )
    full_train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=custom_collate,
    )

    model    = TORCA(checkpoint=cfg["yolo_weights"]).to(device)
    detector = YOLODetector(yolo_or_checkpoint=model._shared_yolo, device=device)

    optimizer = build_optimizer(model, cfg["lr"], cfg["backbone_lr"])

    latest_path = os.path.join(cfg["weights_dir"], "torca_latest.pt")
    best_path   = os.path.join(cfg["weights_dir"], "torca_best.pt")

    start_epoch   = 0
    best_val      = float("inf")
    train_history = []
    val_history   = []
    step_labels   = []

    if os.path.exists(latest_path):
        print(f"\nResuming from {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("full_epoch", 0)
        best_val    = ckpt.get("val_loss", float("inf"))
        hist_path   = os.path.join(cfg["weights_dir"], "training_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                h = json.load(f)
            train_history = h.get("train", [])
            val_history   = h.get("val", [])
            step_labels   = h.get("labels", [])
            best_val      = h.get("best_val_loss", best_val)
        print(f"Resuming from epoch {start_epoch}, best val {best_val:.4f}")

    freeze_epochs = cfg["freeze_epochs"]
    if start_epoch < freeze_epochs:
        set_backbone_grad(model, False)
        print(f"Backbone frozen for first {freeze_epochs} epochs.")

    if chunked:
        print(f"Sub-epoch mode: {chunk_frac*100:.0f}% shards "
              f"({n_chunks} shards/epoch, validating after each)")
    else:
        print("Full-epoch mode.")

    for full_epoch in range(start_epoch, cfg["epochs"]):

        if full_epoch == freeze_epochs:
            set_backbone_grad(model, True)
            print(f"\nEpoch {full_epoch+1}: backbone unfrozen.")

        shards = make_chunks(len(train_ds), chunk_frac) if chunked else [None]

        for shard_idx, shard_indices in enumerate(shards):
            label = (f"E{full_epoch+1}.{shard_idx+1}/{len(shards)}"
                     if chunked else f"E{full_epoch+1}")
            print(f"\n── {label} ──")

            if chunked:
                loader = DataLoader(
                    Subset(train_ds, shard_indices),
                    batch_size=cfg["batch"], shuffle=True,
                    num_workers=cfg["num_workers"], pin_memory=True,
                    collate_fn=custom_collate,
                )
            else:
                loader = full_train_loader

            train_loss = run_epoch(model, detector, loader, optimizer, cfg, device, train=True)
            val_loss   = run_epoch(model, detector, val_loader, optimizer, cfg, device, train=False)

            train_history.append(train_loss)
            val_history.append(val_loss)
            step_labels.append(label)
            print(f"Train: {train_loss:.4f}  |  Val: {val_loss:.4f}")

            ckpt = {
                "full_epoch": full_epoch + 1,
                "shard_idx":  shard_idx,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   val_loss,
                "cfg":        cfg,
            }
            torch.save(ckpt, latest_path)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, best_path)
                print(f"  -> Best checkpoint saved (val={best_val:.4f})")

            with open(os.path.join(cfg["weights_dir"], "training_history.json"), "w") as f:
                json.dump({
                    "train":         train_history,
                    "val":           val_history,
                    "labels":        step_labels,
                    "best_val_loss": best_val,
                }, f, indent=2)

    plot_path = plot_training_losses(
        train_history, val_history,
        labels=step_labels,
        save_dir=cfg["weights_dir"],
    )
    print(f"\nDone. Best val loss : {best_val:.4f}")
    print(f"Best checkpoint    : {best_path}")
    print(f"Loss curve         : {plot_path}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TORCA on COCO-Tasks data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root",     type=str,   default=DEFAULTS["data_root"],
                        help="Root of the preprocessed COCO-Tasks dataset (contains train/ and test/).")
    parser.add_argument("--weights-dir",   type=str,   default=DEFAULTS["weights_dir"],
                        help="Directory to save checkpoints and training history.")
    parser.add_argument("--yolo-weights",  type=str,   default=DEFAULTS["yolo_weights"],
                        help="Path to yolo11n.pt (detector weights, stays frozen).")
    parser.add_argument("--epochs",        type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch",         type=int,   default=DEFAULTS["batch"])
    parser.add_argument("--lr",            type=float, default=DEFAULTS["lr"],
                        help="Learning rate for scorer heads.")
    parser.add_argument("--backbone-lr",   type=float, default=DEFAULTS["backbone_lr"],
                        help="Learning rate for FPN backbone (lower than head LR).")
    parser.add_argument("--freeze-epochs", type=int,   default=DEFAULTS["freeze_epochs"],
                        help="Number of epochs to keep the FPN backbone frozen.")
    parser.add_argument("--num-workers",   type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--iou-threshold", type=float, default=DEFAULTS["iou_threshold"],
                        help="IoU threshold for assigning positive training labels.")
    parser.add_argument("--chunk-frac",    type=float, default=DEFAULTS["chunk_frac"],
                        help="Fraction of training data per shard (e.g. 0.25 → 4 shards/epoch). "
                             "Set to 1.0 for standard full-epoch mode.")
    args = parser.parse_args()
    train({**DEFAULTS, **{k.replace("-", "_"): v for k, v in vars(args).items()}})
