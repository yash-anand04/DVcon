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

from model import DEFAULT_YOLO_PATH, DetectionFeatureExtractor, TaskDrivenAttentionModel, gather_topk_like_model
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
    aux_loss_weight=0.25,
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


def build_targets(batch, topk_boxes, topk_mask, num_tasks, iou_threshold, device):
    targets = torch.zeros(topk_boxes.size(0), topk_boxes.size(1), num_tasks, device=device)
    valid = ~topk_mask

    for b in range(topk_boxes.size(0)):
        task_idx = int(batch["task_id"][b].item()) - 1
        gt_boxes = batch["boxes"][b].to(device)
        prefs = batch["prefs"][b].to(device)
        preferred = gt_boxes[prefs == 1]

        if preferred.numel() == 0:
            continue

        preferred_xyxy = xywh_to_xyxy(preferred)
        det_boxes = topk_boxes[b]
        ious = pairwise_iou_xyxy(det_boxes, preferred_xyxy)
        max_iou = ious.max(dim=1).values if ious.numel() else torch.zeros(det_boxes.size(0), device=device)
        positives = (max_iou >= iou_threshold) & valid[b]
        targets[b, positives, task_idx] = 1.0

    return targets, valid


def run_epoch(model, extractor, loader, criterion, optimizer, cfg, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_steps = 0
    desc = "Train" if train else "Valid"

    for batch in tqdm(loader, desc=desc):
        images = load_pil_images(batch["image_paths"])

        with torch.no_grad():
            class_onehot, visual_feats, det_scores, mask, boxes = extractor.extract_with_boxes(images)

        p_final, p_context, p_visual = model(class_onehot, visual_feats, det_scores, mask)
        topk_boxes = gather_topk_like_model(boxes, det_scores, model.top_k)
        topk_mask = gather_topk_like_model(mask, det_scores, model.top_k)
        targets, valid = build_targets(
            batch, topk_boxes, topk_mask, model.num_tasks,
            cfg["iou_threshold"], device
        )

        if not valid.any():
            continue

        loss_main = criterion(p_final[valid], targets[valid])
        loss_aux = criterion(p_context[valid], targets[valid]) + criterion(p_visual[valid], targets[valid])
        loss = loss_main + cfg["aux_loss_weight"] * loss_aux

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(total_steps, 1)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["weights_dir"], exist_ok=True)
    print(f"Device: {device}")

    train_ds = COCOTasksDataset(cfg["data_root"], split="train", grid_size=16)
    val_ds = COCOTasksDataset(cfg["data_root"], split="test", grid_size=16)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=custom_collate
    )

    extractor = DetectionFeatureExtractor(yolo_model=cfg["yolo_model"], device=device)
    extractor.eval()
    model = TaskDrivenAttentionModel(num_classes=80, num_tasks=14, top_k=cfg["top_k"]).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    train_history = []
    val_history = []
    best_val = float("inf")
    best_path = os.path.join(cfg["weights_dir"], "v1b_attention_best.pt")

    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")
        train_loss = run_epoch(model, extractor, train_loader, criterion, optimizer, cfg, device, train=True)
        val_loss = run_epoch(model, extractor, val_loader, criterion, optimizer, cfg, device, train=False)

        train_history.append(train_loss)
        val_history.append(val_loss)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        latest_path = os.path.join(cfg["weights_dir"], "v1b_attention_latest.pt")
        ckpt = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "cfg": cfg,
        }
        torch.save(ckpt, latest_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint -> {best_path}")

        with open(os.path.join(cfg["weights_dir"], "training_history.json"), "w") as f:
            json.dump({"train": train_history, "val": val_history, "best_val_loss": best_val}, f, indent=2)

    plot_path = plot_training_losses(train_history, val_history, save_dir=cfg["weights_dir"])
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
    parser.add_argument("--aux-loss-weight", type=float, default=DEFAULTS["aux_loss_weight"])
    args = parser.parse_args()
    train({**DEFAULTS, **vars(args)})
