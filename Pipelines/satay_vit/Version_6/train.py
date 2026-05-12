"""
train.py  –  SATAY-ViT V6 Training Script
==========================================
Trains SATAYViT_V6 (FPN backbone + RoI-Align + task attention) jointly.
YOLO detector stays frozen; FPN backbone is fine-tuned after freeze_epochs.

Usage:
    python train.py [--epochs N] [--batch B] [--lr LR] [--freeze-epochs N]
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import DEFAULT_YOLO_PATH, SATAYViT_V6, YOLODetector
from utils.data_loader import COCOTasksDataset, custom_collate
from utils.plot_metrics import plot_training_losses

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    data_root      = "e:/DVcon/DVcon/Data_Preprocessed",
    weights_dir    = os.path.join(CURRENT_DIR, "weights"),
    yolo_model     = DEFAULT_YOLO_PATH,
    epochs         = 15,
    batch          = 8,
    lr             = 5e-5,
    backbone_lr    = 5e-6,
    freeze_epochs  = 3,
    num_workers    = 4,
    embed_dim      = 256,
    iou_threshold  = 0.5,
)


# ─────────────────────────────────────────────────────────────────────
#  Box utilities
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
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-6)


def load_pil_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


# ─────────────────────────────────────────────────────────────────────
#  Target builder
# ─────────────────────────────────────────────────────────────────────
def build_targets(batch, det_results, mask, iou_threshold, device):
    """
    Returns
      targets [B, maxN] : 1.0 where detected box overlaps preferred GT (IoU >= thresh)
      valid   [B, maxN] : True where position is a real detection (not padding)
    """
    B    = len(det_results)
    maxN = mask.shape[1]
    targets = torch.zeros(B, maxN, device=device)
    valid   = ~mask

    for b in range(B):
        gt_boxes = batch["boxes"][b].to(device)     # [G, 4] xywh
        prefs    = batch["prefs"][b].to(device)     # [G]
        preferred = gt_boxes[prefs == 1]            # preferred GT boxes

        if preferred.numel() == 0:
            continue

        preferred_xyxy = xywh_to_xyxy(preferred)
        boxes_b, _, _  = det_results[b]            # xyxy, already in 640-px space

        if boxes_b.shape[0] == 0:
            continue

        n_b      = min(boxes_b.shape[0], maxN)
        det_xyxy = boxes_b[:n_b].to(device)
        ious     = pairwise_iou_xyxy(det_xyxy, preferred_xyxy)
        max_iou  = ious.max(dim=1).values
        positives = (max_iou >= iou_threshold) & valid[b, :n_b]
        targets[b, :n_b] = positives.float()

    return targets, valid


# ─────────────────────────────────────────────────────────────────────
#  Optimizer with separate backbone / head LRs
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
#  Epoch loop
# ─────────────────────────────────────────────────────────────────────
def run_epoch(model, detector, loader, criterion, optimizer, cfg, device, train=True):
    model.train(train)
    total_loss, total_steps = 0.0, 0
    desc = "Train" if train else "Valid"

    for batch in tqdm(loader, desc=desc):
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
#  Main training function
# ─────────────────────────────────────────────────────────────────────
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    print(f"Device: {device}")

    train_ds = COCOTasksDataset(cfg["data_root"], split="train", grid_size=16)
    val_ds   = COCOTasksDataset(cfg["data_root"], split="test",  grid_size=16)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=custom_collate,
    )

    detector = YOLODetector(checkpoint=cfg["yolo_model"], device=device)
    model    = SATAYViT_V6(
        checkpoint=cfg["yolo_model"],
        embed_dim=cfg["embed_dim"],
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = build_optimizer(model, cfg["lr"], cfg["backbone_lr"])

    latest_path = os.path.join(cfg["weights_dir"], "v6_latest.pt")
    best_path   = os.path.join(cfg["weights_dir"], "v6_best.pt")

    start_epoch   = 0
    best_val      = float("inf")
    train_history = []
    val_history   = []

    if os.path.exists(latest_path):
        print(f"\nResuming from: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val    = ckpt.get("val_loss", float("inf"))
        hist_path   = os.path.join(cfg["weights_dir"], "training_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                h = json.load(f)
            train_history = h.get("train", [])
            val_history   = h.get("val", [])
            best_val      = h.get("best_val_loss", best_val)
        print(f"Epoch {start_epoch}, best val loss {best_val:.4f}")

    # Freeze backbone for first freeze_epochs
    freeze_epochs = cfg["freeze_epochs"]
    if start_epoch < freeze_epochs:
        set_backbone_grad(model, False)
        print(f"Backbone frozen for first {freeze_epochs} epochs.")

    for epoch in range(start_epoch, cfg["epochs"]):
        if epoch == freeze_epochs:
            set_backbone_grad(model, True)
            print(f"\nEpoch {epoch+1}: backbone unfrozen.")

        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")
        train_loss = run_epoch(model, detector, train_loader, criterion, optimizer, cfg, device, train=True)
        val_loss   = run_epoch(model, detector, val_loader,   criterion, optimizer, cfg, device, train=False)

        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"Train: {train_loss:.4f}  |  Val: {val_loss:.4f}")

        ckpt = {
            "epoch":      epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_loss":   val_loss,
            "cfg":        cfg,
        }
        torch.save(ckpt, latest_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_path)
            print(f"  -> Best checkpoint saved ({best_val:.4f})")

        with open(os.path.join(cfg["weights_dir"], "training_history.json"), "w") as f:
            json.dump({"train": train_history, "val": val_history, "best_val_loss": best_val}, f, indent=2)

    plot_path = plot_training_losses(train_history, val_history, save_dir=cfg["weights_dir"])
    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Best checkpoint : {best_path}")
    print(f"Loss curve      : {plot_path}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",     type=str,   default=DEFAULTS["data_root"])
    parser.add_argument("--weights-dir",   type=str,   default=DEFAULTS["weights_dir"])
    parser.add_argument("--yolo-model",    type=str,   default=DEFAULTS["yolo_model"])
    parser.add_argument("--epochs",        type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch",         type=int,   default=DEFAULTS["batch"])
    parser.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    parser.add_argument("--backbone-lr",   type=float, default=DEFAULTS["backbone_lr"])
    parser.add_argument("--freeze-epochs", type=int,   default=DEFAULTS["freeze_epochs"])
    parser.add_argument("--num-workers",   type=int,   default=DEFAULTS["num_workers"])
    parser.add_argument("--embed-dim",     type=int,   default=DEFAULTS["embed_dim"])
    parser.add_argument("--iou-threshold", type=float, default=DEFAULTS["iou_threshold"])
    args = parser.parse_args()
    train({**DEFAULTS, **{k.replace("-", "_"): v for k, v in vars(args).items()}})
