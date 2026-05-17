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

from model import DEFAULT_YOLO_PATH, OpenVocabDetectionFeatureExtractor, TaskAwareRelevanceHead, gather_topk_like_model
from utils.data_loader import COCOTasksDataset, custom_collate
from utils.plot_metrics import plot_training_losses


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    data_root="e:/DVcon/DVcon/Data_Preprocessed",
    weights_dir=os.path.join(CURRENT_DIR, "weights"),
    yolo_model=DEFAULT_YOLO_PATH,
    epochs=10,
    batch=4,
    lr=1e-4,
    num_workers=0,
    top_k=20,
    iou_threshold=0.5,
    pos_weight=5.0,
    score_mode="fused",
    chunk_frac=0.2,   # 1.0 = standard full-epoch mode
)


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
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def load_pil_images(paths):
    return [Image.open(path).convert("RGB") for path in paths]


def build_relevance_targets(batch, topk_boxes, topk_mask, iou_threshold, device):
    targets = torch.zeros(topk_boxes.size(0), topk_boxes.size(1), device=device)
    valid = ~topk_mask

    for b in range(topk_boxes.size(0)):
        gt_boxes = batch["boxes"][b].to(device)
        prefs = batch["prefs"][b].to(device)
        preferred = gt_boxes[prefs == 1]
        if preferred.numel() == 0:
            continue

        preferred_xyxy = xywh_to_xyxy(preferred)
        ious = pairwise_iou_xyxy(topk_boxes[b], preferred_xyxy)
        max_iou = ious.max(dim=1).values if ious.numel() else torch.zeros(topk_boxes.size(1), device=device)
        targets[b] = ((max_iou >= iou_threshold) & valid[b]).float()

    return targets, valid


def run_epoch(model, extractor, loader, criterion, optimizer, cfg, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_steps = 0
    total_pos = 0.0
    total_valid = 0.0
    desc = "Train" if train else "Valid"

    for batch in tqdm(loader, desc=desc):
        images = load_pil_images(batch["image_paths"])

        with torch.no_grad():
            visual_feats, class_ids, det_scores, text_scores, mask, boxes, image_sizes = extractor.extract_with_boxes(images)

        task_ids = batch["task_id"].to(device)
        logits = model(visual_feats, class_ids, det_scores, text_scores, boxes, image_sizes, task_ids, mask)
        topk_boxes = gather_topk_like_model(boxes, det_scores, model.top_k)
        topk_mask = gather_topk_like_model(mask, det_scores, model.top_k)
        targets, valid = build_relevance_targets(batch, topk_boxes, topk_mask, cfg["iou_threshold"], device)

        if not valid.any():
            continue

        loss = criterion(logits[valid], targets[valid])

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1
        total_pos += targets[valid].sum().item()
        total_valid += valid.sum().item()

    return {
        "loss": total_loss / max(total_steps, 1),
        "positive_rate": total_pos / max(total_valid, 1.0),
    }


def make_chunks(n_samples, chunk_frac):
    chunk_size = max(1, int(math.floor(n_samples * chunk_frac)))
    perm = torch.randperm(n_samples).tolist()
    chunks = []
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        if n_samples - end < chunk_size * 0.25 and chunks:
            chunks[-1].extend(perm[start:end])
        else:
            chunks.append(perm[start:end])
    return chunks


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    print(f"Device: {device}")
    print(f"Detector: {cfg['yolo_model']}")

    chunk_frac = float(cfg["chunk_frac"])
    chunked    = chunk_frac < 1.0
    n_chunks   = math.ceil(1.0 / chunk_frac) if chunked else 1

    train_ds = COCOTasksDataset(cfg["data_root"], split="train", grid_size=16)
    val_ds   = COCOTasksDataset(cfg["data_root"], split="test",  grid_size=16)

    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=custom_collate,
    )
    full_train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=custom_collate,
    )

    extractor = OpenVocabDetectionFeatureExtractor(yolo_model=cfg["yolo_model"], device=device)
    extractor.eval()
    model = TaskAwareRelevanceHead(num_classes=512, num_tasks=14, feat_dim=256, top_k=cfg["top_k"]).to(device)

    pos_weight = torch.tensor([cfg["pos_weight"]], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    train_history = []
    val_history   = []
    step_labels   = []
    best_val  = float("inf")
    best_path = os.path.join(cfg["weights_dir"], "v1c_relevance_best.pt")

    if chunked:
        print(f"Sub-epoch mode: {chunk_frac*100:.0f}% shards "
              f"({n_chunks} shards/epoch, validating after each)")
    else:
        print("Full-epoch mode.")

    for epoch in range(cfg["epochs"]):
        shards = make_chunks(len(train_ds), chunk_frac) if chunked else [None]

        for shard_idx, shard_indices in enumerate(shards):
            label = (f"E{epoch+1}.{shard_idx+1}/{len(shards)}"
                     if chunked else f"E{epoch+1}")
            print(f"\n── {label} ──")

            if chunked:
                loader = DataLoader(
                    Subset(train_ds, shard_indices),
                    batch_size=cfg["batch"], shuffle=True,
                    num_workers=cfg["num_workers"], collate_fn=custom_collate,
                )
            else:
                loader = full_train_loader

            train_stats = run_epoch(model, extractor, loader,      criterion, optimizer, cfg, device, train=True)
            val_stats   = run_epoch(model, extractor, val_loader,  criterion, optimizer, cfg, device, train=False)

            train_history.append(train_stats["loss"])
            val_history.append(val_stats["loss"])
            step_labels.append(label)
            print(
                f"Train: {train_stats['loss']:.4f}  |  Val: {val_stats['loss']:.4f}  |  "
                f"Pos rate: {train_stats['positive_rate']:.4f}"
            )

            latest_path = os.path.join(cfg["weights_dir"], "v1c_relevance_latest.pt")
            ckpt = {
                "full_epoch": epoch + 1,
                "shard_idx":  shard_idx,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   val_stats["loss"],
                "cfg":        cfg,
            }
            torch.save(ckpt, latest_path)

            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                torch.save(ckpt, best_path)
                print(f"  -> Best checkpoint saved (val={best_val:.4f})")

            with open(os.path.join(cfg["weights_dir"], "training_history.json"), "w") as f:
                json.dump({
                    "train":         train_history,
                    "val":           val_history,
                    "labels":        step_labels,
                    "best_val_loss": best_val,
                    "pos_weight":    cfg["pos_weight"],
                    "yolo_model":    cfg["yolo_model"],
                }, f, indent=2)

    plot_path = plot_training_losses(train_history, val_history, labels=step_labels, save_dir=cfg["weights_dir"])
    print(f"\nTraining finished. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {best_path}")
    print(f"Loss curve: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULTS["data_root"])
    parser.add_argument("--weights-dir", type=str, default=DEFAULTS["weights_dir"])
    parser.add_argument("--yolo-model", type=str, default=DEFAULTS["yolo_model"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--iou-threshold", type=float, default=DEFAULTS["iou_threshold"])
    parser.add_argument("--pos-weight",  type=float, default=DEFAULTS["pos_weight"])
    parser.add_argument("--chunk-frac",  type=float, default=DEFAULTS["chunk_frac"],
                        help="Fraction of training data per shard (e.g. 0.25 → 4 shards/epoch). "
                             "Set to 1.0 for standard full-epoch mode.")
    args = parser.parse_args()
    train({**DEFAULTS, **{k.replace("-", "_"): v for k, v in vars(args).items()}})
