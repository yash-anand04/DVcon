"""
train.py  –  SATAY-ViT V7 Training Script
==========================================
Trains SATAYViT_V7 (FPN backbone + RoI-Align + task attention) jointly.
YOLO detector stays frozen; FPN backbone is fine-tuned after freeze_epochs.

Chunked-epoch mode (--chunk-frac):
  Instead of one full dataset pass per epoch, each full epoch is split into
  N sequential shards.  Validation runs after every shard so the best
  checkpoint is saved at sub-epoch granularity.  This prevents the model from
  memorising the entire training set before validation can catch overfitting.

  Example: --chunk-frac 0.2  →  5 shards per epoch, 5× more val checks.

Usage:
    python train.py [--epochs N] [--batch B] [--lr LR] [--freeze-epochs N]
                    [--chunk-frac F]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import DEFAULT_YOLO_PATH, SATAYViT_V7, YOLODetector
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
    chunk_frac     = 0.2,    # 1.0 = standard full-epoch mode
    focal_gamma    = 2.0,
    focal_alpha    = 0.25,   # restored: the prior recommendation (0.75) starved the model and hurt convergence
    soft_targets   = True,   # restored: hard targets alone didn't compensate for the slowed CLIP projection
    clip_proj_lr   = 5e-5,   # same as head LR — the critical new layer needs full-speed training
    patience       = 6,      # early-stop after this many consecutive non-improving validation steps
    min_delta      = 1e-5,   # val loss must improve by at least this to count as "improvement"
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
def build_targets(batch, det_results, mask, iou_threshold, device, soft=False):
    """
    Returns
      targets [B, maxN] : default — hard 0/1 (1.0 if IoU>=thresh else 0.0)
                          if soft=True, returns the exact IoU score for boxes
                          above the threshold and 0 otherwise.
      valid   [B, maxN] : True where position is a real detection (not padding)

    Hard targets keep the model's argmax sharp (good for Top-1).  Soft targets
    are better-calibrated (good for mAP) but smear the score for borderline
    positives, which can flip the argmax.  V7 defaults to hard targets after
    measuring a Top-1 regression from soft mode.
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
        max_iou  = ious.max(dim=1).values
        above    = (max_iou >= iou_threshold).float()
        targets[b, :n_b] = (max_iou * above) if soft else above

    return targets, valid


# ─────────────────────────────────────────────────────────────────────
#  Focal loss
# ─────────────────────────────────────────────────────────────────────
def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    """
    Binary focal loss.  Reduces the contribution of easy negatives
    (the ~14 clearly-wrong boxes per image) so gradients focus on
    hard / borderline examples.

    alpha  : prior weight for the positive class (0.25 is standard)
    gamma  : focusing exponent; 0 → plain BCE, 2 is standard
    """
    bce   = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
    p_t   = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


# ─────────────────────────────────────────────────────────────────────
#  Optimizer with separate backbone / head LRs
# ─────────────────────────────────────────────────────────────────────
def build_optimizer(model, lr, backbone_lr, clip_proj_lr):
    """
    Three parameter groups:
      backbone  : YOLO FPN layers     (lr = backbone_lr, frozen for freeze_epochs)
      clip_proj : task_proj Linear    (lr = clip_proj_lr — lower, to preserve CLIP structure)
      head      : everything else     (lr = lr — full head LR)
    """
    backbone_params  = list(model.backbone.parameters())
    clip_proj_params = list(model.scorer.task_proj.parameters())
    clip_proj_ids    = {id(p) for p in clip_proj_params}
    head_params = [
        p for p in list(model.roi_fusion.parameters()) + list(model.scorer.parameters())
        if id(p) not in clip_proj_ids
    ]
    return optim.AdamW(
        [{"params": backbone_params,  "lr": backbone_lr},
         {"params": clip_proj_params, "lr": clip_proj_lr},
         {"params": head_params,      "lr": lr}],
        weight_decay=1e-4,
    )


def set_backbone_grad(model, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad_(requires_grad)


# ─────────────────────────────────────────────────────────────────────
#  Single loader pass (train or val)
# ─────────────────────────────────────────────────────────────────────
def run_epoch(model, detector, loader, criterion, optimizer, cfg, device, train=True):
    model.train(train)
    total_loss, total_steps = 0.0, 0
    desc = "Train" if train else "Valid"

    for batch in tqdm(loader, desc=desc, leave=False):
        pil_images  = load_pil_images(batch["image_paths"])
        img_tensors = batch["image"].to(device)
        task_ids    = batch["task_id"].to(device)

        with torch.no_grad():
            det_results = detector.detect_batch(pil_images)

        rel_scores, _, mask = model(img_tensors, det_results, task_ids)
        targets, valid = build_targets(batch, det_results, mask, cfg["iou_threshold"], device,
                                        soft=cfg.get("soft_targets", False))

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
#  Chunk helper: split shuffled indices into N equal shards
# ─────────────────────────────────────────────────────────────────────
def make_chunks(n_samples, chunk_frac):
    """
    Returns a list of index lists.  Each list is ~(chunk_frac * n_samples) long.
    The final shard may be slightly larger to avoid a tiny leftover.
    """
    chunk_size = max(1, int(math.floor(n_samples * chunk_frac)))
    perm       = torch.randperm(n_samples).tolist()
    chunks     = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        # absorb tiny tail into last chunk
        if n_samples - end < chunk_size * 0.25 and chunks:
            chunks[-1].extend(perm[start:end])
        else:
            chunks.append(perm[start:end])
    return chunks


# ─────────────────────────────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────────────────────────────
def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    print(f"Device: {device}")

    chunk_frac  = float(cfg["chunk_frac"])
    chunked     = chunk_frac < 1.0
    n_chunks    = math.ceil(1.0 / chunk_frac) if chunked else 1

    train_ds = COCOTasksDataset(cfg["data_root"], split="train", grid_size=16)
    val_ds   = COCOTasksDataset(cfg["data_root"], split="test",  grid_size=16)

    # Full val loader is reused every shard
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=custom_collate,
    )

    # Standard full-epoch loader (used when chunked=False)
    full_train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=custom_collate,
    )

    detector = YOLODetector(checkpoint=cfg["yolo_model"], device=device)
    model    = SATAYViT_V7(
        checkpoint=cfg["yolo_model"],
        embed_dim=cfg["embed_dim"],
    ).to(device)

    criterion = lambda p, t: focal_loss(p, t, gamma=cfg["focal_gamma"], alpha=cfg["focal_alpha"])
    optimizer = build_optimizer(model, cfg["lr"], cfg["backbone_lr"], cfg["clip_proj_lr"])

    latest_path = os.path.join(cfg["weights_dir"], "v7_latest.pt")
    best_path   = os.path.join(cfg["weights_dir"], "v7_best.pt")

    start_epoch   = 0
    best_val      = float("inf")
    train_history = []   # one entry per shard (or per epoch if not chunked)
    val_history   = []
    step_labels   = []   # e.g. "E1.2/5"

    if os.path.exists(latest_path):
        print(f"\nResuming from: {latest_path}")
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("full_epoch", ckpt.get("epoch", 0))
        best_val    = ckpt.get("val_loss", float("inf"))
        hist_path   = os.path.join(cfg["weights_dir"], "training_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                h = json.load(f)
            train_history = h.get("train", [])
            val_history   = h.get("val",   [])
            step_labels   = h.get("labels", [])
            best_val      = h.get("best_val_loss", best_val)
        print(f"Resuming from full-epoch {start_epoch}, best val loss {best_val:.4f}")

    freeze_epochs = cfg["freeze_epochs"]
    if start_epoch < freeze_epochs:
        set_backbone_grad(model, False)
        print(f"Backbone frozen for first {freeze_epochs} full epochs.")

    if chunked:
        print(f"Chunked mode: {chunk_frac*100:.0f}% shards "
              f"({n_chunks} shards/epoch, validating after each shard)")
    else:
        print("Standard full-epoch mode.")

    # Early-stopping state
    patience    = int(cfg["patience"])
    min_delta   = float(cfg["min_delta"])
    bad_steps   = 0
    stop_early  = False
    print(f"Early stopping: patience={patience} non-improving val steps "
          f"(min Δ={min_delta:.0e})")

    for full_epoch in range(start_epoch, cfg["epochs"]):
        if stop_early:
            break

        if full_epoch == freeze_epochs:
            set_backbone_grad(model, True)
            print(f"\nFull epoch {full_epoch+1}: backbone unfrozen.")

        if chunked:
            shards = make_chunks(len(train_ds), chunk_frac)
        else:
            shards = [None]   # sentinel → use full_train_loader

        for shard_idx, shard_indices in enumerate(shards):
            label = (f"E{full_epoch+1}.{shard_idx+1}/{len(shards)}"
                     if chunked else f"E{full_epoch+1}")
            print(f"\n── {label} ──")

            if chunked:
                shard_loader = DataLoader(
                    Subset(train_ds, shard_indices),
                    batch_size=cfg["batch"], shuffle=True,
                    num_workers=cfg["num_workers"], pin_memory=True,
                    collate_fn=custom_collate,
                )
                train_loss = run_epoch(
                    model, detector, shard_loader, criterion, optimizer,
                    cfg, device, train=True,
                )
            else:
                train_loss = run_epoch(
                    model, detector, full_train_loader, criterion, optimizer,
                    cfg, device, train=True,
                )

            val_loss = run_epoch(
                model, detector, val_loader, criterion, optimizer,
                cfg, device, train=False,
            )

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

            if val_loss < best_val - min_delta:
                best_val  = val_loss
                bad_steps = 0
                torch.save(ckpt, best_path)
                print(f"  -> Best checkpoint saved (val={best_val:.4f})")
            else:
                bad_steps += 1
                print(f"  -> No improvement ({bad_steps}/{patience} bad val steps)")

            with open(os.path.join(cfg["weights_dir"], "training_history.json"), "w") as f:
                json.dump({
                    "train":         train_history,
                    "val":           val_history,
                    "labels":        step_labels,
                    "best_val_loss": best_val,
                }, f, indent=2)

            if bad_steps >= patience:
                print(f"\nEarly stopping: val loss did not improve for "
                      f"{patience} consecutive steps. Best val={best_val:.4f}")
                stop_early = True
                break

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
    parser.add_argument("--chunk-frac",    type=float, default=DEFAULTS["chunk_frac"],
                        help="Fraction of training data per shard (0.1–1.0). "
                             "Values <1 enable sub-epoch validation.")
    parser.add_argument("--focal-gamma",   type=float, default=DEFAULTS["focal_gamma"],
                        help="Focal loss focusing exponent (0=plain BCE, 2=standard).")
    parser.add_argument("--focal-alpha",   type=float, default=DEFAULTS["focal_alpha"],
                        help="Focal loss prior weight for positives (0.75 default for ~7% positive rate).")
    parser.add_argument("--soft-targets",  action="store_true",
                        help="Use soft IoU targets instead of hard 0/1 (better mAP, worse Top-1).")
    parser.add_argument("--clip-proj-lr",  type=float, default=DEFAULTS["clip_proj_lr"],
                        help="LR for the CLIP→embed_dim projection layer (smaller preserves CLIP semantics).")
    parser.add_argument("--patience",      type=int,   default=DEFAULTS["patience"],
                        help="Early-stop after this many consecutive non-improving val steps.")
    parser.add_argument("--min-delta",     type=float, default=DEFAULTS["min_delta"],
                        help="Minimum val-loss improvement to reset the early-stop counter.")
    args = parser.parse_args()
    train({**DEFAULTS, **{k.replace("-", "_"): v for k, v in vars(args).items()}})
