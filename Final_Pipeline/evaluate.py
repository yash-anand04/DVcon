"""
evaluate.py — TORCA Evaluation
================================
Computes Top-1 Accuracy and mAP@0.5 on the COCO-Tasks test split.
Writes eval_results.json, per_task_accuracy.png, and loss_curve.png.

Usage:
    python evaluate.py --data-root /path/to/Data_Preprocessed
    python evaluate.py --data-root /path/to/Data_Preprocessed --weights weights/torca_best.pt
    python evaluate.py --data-root /path/to/Data_Preprocessed --num-workers 0 --batch 4  # CPU-safe
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from model import DEFAULT_YOLO_PATH, TORCA, YOLODetector
from utils.data_loader import COCOTasksDataset, custom_collate

TASK_NAMES = {
    1:  "Step on something",   2:  "Sit comfortably",
    3:  "Place flowers",       4:  "Get potatoes out of fire",
    5:  "Water plant",         6:  "Get lemon out of tea",
    7:  "Dig hole",            8:  "Open bottle of beer",
    9:  "Open parcel",         10: "Serve wine",
    11: "Pour sugar",          12: "Smear butter",
    13: "Extinguish fire",     14: "Pound carpet",
}

DEFAULTS = dict(
    data_root      = os.path.join(THIS, "data"),
    weights        = os.path.join(THIS, "weights", "torca_best.pt"),
    yolo_weights   = DEFAULT_YOLO_PATH,
    output_dir     = THIS,
    batch          = 8,
    num_workers    = 4,
    iou_thresholds = [0.5],
)


# ─────────────────────────────────────────────────────────────────────
#  Geometry utilities
# ─────────────────────────────────────────────────────────────────────
def xywh_to_xyxy(boxes):
    out = boxes.clone()
    out[..., 2] = boxes[..., 0] + boxes[..., 2]
    out[..., 3] = boxes[..., 1] + boxes[..., 3]
    return out


def pairwise_iou_xyxy(b1, b2):
    if b1.numel() == 0 or b2.numel() == 0:
        return b1.new_zeros((b1.shape[0], b2.shape[0]))
    x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-6)


def ap_at_iou(recalls, precisions):
    recalls    = np.concatenate([[0.0], np.array(recalls),    [1.0]])
    precisions = np.concatenate([[1.0], np.array(precisions), [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def load_pil_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


# ─────────────────────────────────────────────────────────────────────
#  Main evaluation loop
# ─────────────────────────────────────────────────────────────────────
def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print(f"Device      : {device}")
    print(f"Data root   : {cfg['data_root']}")
    print(f"Weights     : {cfg['weights']}")

    ds     = COCOTasksDataset(cfg["data_root"], split="test")
    loader = DataLoader(
        ds, batch_size=cfg["batch"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=custom_collate,
    )

    model    = TORCA(checkpoint=cfg["yolo_weights"]).to(device)
    detector = YOLODetector(yolo_or_checkpoint=model._shared_yolo, device=device)

    if not os.path.exists(cfg["weights"]):
        raise FileNotFoundError(
            f"Model weights not found: {cfg['weights']}\n"
            f"Train first with train.py, or point --weights to an existing checkpoint."
        )
    ckpt = torch.load(cfg["weights"], map_location=device)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    print(f"Loaded weights.")
    model.eval()

    iou_thresholds = cfg["iou_thresholds"]
    if isinstance(iou_thresholds, float):
        iou_thresholds = [iou_thresholds]

    records       = []
    per_task_torca = {t: {"total": 0, "correct": {iou: 0 for iou in iou_thresholds}} for t in range(1, 15)}
    per_task_yolo  = {t: {"total": 0, "correct": {iou: 0 for iou in iou_thresholds}} for t in range(1, 15)}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating TORCA"):
            pil_images  = load_pil_images(batch["image_paths"])
            img_tensors = batch["image"].to(device)
            task_ids    = batch["task_id"].to(device)

            det_results                    = detector.detect_batch(pil_images)
            rel_scores, det_scores_t, mask = model(img_tensors, det_results, task_ids)

            for b in range(img_tensors.shape[0]):
                task_id   = int(batch["task_id"][b].item())
                gt_boxes  = batch["boxes"][b].to(device)
                prefs     = batch["prefs"][b].to(device)
                preferred = gt_boxes[prefs == 1]
                if preferred.numel() == 0:
                    continue

                preferred_xyxy = xywh_to_xyxy(preferred)
                boxes_b, _, _  = det_results[b]

                if boxes_b.shape[0] == 0:
                    hit_dict      = {iou: False for iou in iou_thresholds}
                    yolo_hit_dict = {iou: False for iou in iou_thresholds}
                    records.append((0.0, hit_dict, yolo_hit_dict, task_id))
                    continue

                per_task_torca[task_id]["total"] += 1
                per_task_yolo[task_id]["total"]  += 1

                n_b   = boxes_b.shape[0]
                valid = ~mask[b, :n_b]

                final_scores = rel_scores[b, :n_b] * det_scores_t[b, :n_b]
                final_scores[~valid] = -1.0
                yolo_scores  = det_scores_t[b, :n_b].clone()
                yolo_scores[~valid] = -1.0

                best_torca = int(final_scores.argmax())
                best_yolo  = int(yolo_scores.argmax())

                torca_iou = pairwise_iou_xyxy(boxes_b[best_torca].unsqueeze(0), preferred_xyxy).max().item()
                yolo_iou  = pairwise_iou_xyxy(boxes_b[best_yolo].unsqueeze(0),  preferred_xyxy).max().item()

                hit_dict      = {iou: torca_iou >= iou for iou in iou_thresholds}
                yolo_hit_dict = {iou: yolo_iou  >= iou for iou in iou_thresholds}
                records.append((float(final_scores[best_torca].item()), hit_dict, yolo_hit_dict, task_id))

                for iou in iou_thresholds:
                    if hit_dict[iou]:      per_task_torca[task_id]["correct"][iou] += 1
                    if yolo_hit_dict[iou]: per_task_yolo[task_id]["correct"][iou]  += 1

    # ── Aggregate ─────────────────────────────────────────────────────
    total      = len(records)
    top1_torca = {}
    top1_yolo  = {}
    maps       = {}

    for iou in iou_thresholds:
        tp_t = sum(r[1][iou] for r in records)
        tp_y = sum(r[2][iou] for r in records)
        top1_torca[iou] = tp_t / total if total else 0.0
        top1_yolo[iou]  = tp_y / total if total else 0.0

        sorted_recs = sorted(records, key=lambda r: r[0], reverse=True)
        cum_tp = cum_fp = 0
        prec_l, rec_l = [], []
        for _, hits, _, _ in sorted_recs:
            if hits[iou]: cum_tp += 1
            else:         cum_fp += 1
            prec_l.append(cum_tp / (cum_tp + cum_fp))
            rec_l.append(cum_tp / total)
        maps[iou] = ap_at_iou(rec_l, prec_l)

    per_task_results = {}
    for t in range(1, 15):
        n = per_task_torca[t]["total"]
        per_task_results[t] = {
            "task_name": TASK_NAMES[t],
            "total":     n,
            "torca_acc": {iou: per_task_torca[t]["correct"][iou] / n if n else 0.0 for iou in iou_thresholds},
            "yolo_acc":  {iou: per_task_yolo[t]["correct"][iou]  / n if n else 0.0 for iou in iou_thresholds},
        }

    # ── Print results table ───────────────────────────────────────────
    primary = iou_thresholds[0]
    print("\n" + "=" * 70)
    print("  TORCA EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Samples evaluated      : {total:,}")
    for iou in iou_thresholds:
        print(f"  Top-1 TORCA @ IoU {iou:.2f} : {top1_torca[iou]*100:.2f}%")
    print(f"  Top-1 YOLO  @ IoU {primary:.2f} : {top1_yolo[primary]*100:.2f}%")
    for iou in iou_thresholds:
        print(f"  mAP         @ IoU {iou:.2f} : {maps[iou]*100:.2f}%")
    print("-" * 70)
    print(f"  {'Task':<30} {'TORCA':>8}   {'YOLO':>8}")
    print("-" * 70)
    for t in range(1, 15):
        n     = per_task_results[t]["total"]
        torca = per_task_results[t]["torca_acc"][primary] * 100
        yolo  = per_task_results[t]["yolo_acc"][primary]  * 100
        print(f"  {TASK_NAMES[t]:<30} {torca:>7.1f}%  {yolo:>7.1f}%")
    print("=" * 70)

    # ── Save JSON ──────────────────────────────────────────────────────
    results = {
        "total_samples": total,
        "top1_torca":    top1_torca,
        "top1_yolo":     top1_yolo,
        "map":           maps,
        "per_task":      per_task_results,
    }
    results_path = os.path.join(cfg["output_dir"], "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    # ── Per-task bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    labels   = [TASK_NAMES[t] for t in range(1, 15)]
    x        = np.arange(len(labels))
    width    = 0.38
    torca_accs = [per_task_results[t]["torca_acc"][primary] * 100 for t in range(1, 15)]
    yolo_accs  = [per_task_results[t]["yolo_acc"][primary]  * 100 for t in range(1, 15)]
    ax.bar(x - width / 2, torca_accs, width, label="TORCA",      color="#C84B0E", edgecolor="#7D2D08", linewidth=0.6, zorder=3)
    ax.bar(x + width / 2, yolo_accs,  width, label="YOLO-only",  color="#88BBE8", edgecolor="#3F6F9F", linewidth=0.6, zorder=3)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(
        f"Per-Task Top-1 Accuracy (Overall: TORCA {top1_torca[primary]*100:.1f}%, "
        f"YOLO {top1_yolo[primary]*100:.1f}%, mAP@{primary} {maps[primary]*100:.1f}%)",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    plt.tight_layout()
    chart_path = os.path.join(cfg["output_dir"], "per_task_accuracy.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Chart saved   -> {chart_path}")

    # ── Training loss curve (if history exists) ───────────────────────
    hist_path = os.path.join(cfg["output_dir"], "weights", "training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
        train_l = hist.get("train", [])
        val_l   = hist.get("val",   [])
        hist_labels = hist.get("labels", [])
        if train_l:
            steps = range(1, len(train_l) + 1)
            fig, ax = plt.subplots(figsize=(max(8, len(train_l) * 0.5), 5))
            ax.plot(steps, train_l, "o-",  color="#C84B0E", lw=1.5, markersize=4, label="Train BCE")
            ax.plot(steps, val_l,   "s--", color="#4C9BE8", lw=1.5, markersize=4, label="Val BCE")
            best_i = val_l.index(min(val_l))
            ax.axvline(best_i + 1, color="#2ECC71", lw=1.2, alpha=0.7,
                       label=f"Best ckpt ({hist_labels[best_i] if hist_labels else best_i+1}, val={min(val_l):.3f})")
            if hist_labels:
                ax.set_xticks(list(steps)[::4])
                ax.set_xticklabels([hist_labels[i] for i in range(0, len(hist_labels), 4)],
                                   rotation=45, ha="right", fontsize=7)
            ax.set_xlabel("Sub-Epoch")
            ax.set_ylabel("BCE Loss")
            ax.set_title("TORCA — Training Loss Curve")
            ax.legend()
            ax.yaxis.grid(True, linestyle="--", alpha=0.6)
            ax.set_axisbelow(True)
            plt.tight_layout()
            lc_path = os.path.join(cfg["output_dir"], "loss_curve.png")
            plt.savefig(lc_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Loss curve    -> {lc_path}")

    return results


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TORCA on the COCO-Tasks test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root",      type=str,   default=DEFAULTS["data_root"],
                        help="Root of the preprocessed COCO-Tasks dataset.")
    parser.add_argument("--weights",        type=str,   default=DEFAULTS["weights"],
                        help="Path to trained TORCA checkpoint (.pt).")
    parser.add_argument("--yolo-weights",   type=str,   default=DEFAULTS["yolo_weights"],
                        help="Path to yolo11n.pt.")
    parser.add_argument("--output-dir",     type=str,   default=DEFAULTS["output_dir"],
                        help="Directory to write eval_results.json and charts.")
    parser.add_argument("--batch",          type=int,   default=DEFAULTS["batch"])
    parser.add_argument("--num-workers",    type=int,   default=DEFAULTS["num_workers"],
                        help="Set to 0 for CPU-only evaluation.")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=DEFAULTS["iou_thresholds"])
    args = parser.parse_args()
    evaluate({**DEFAULTS, **{k.replace("-", "_"): v for k, v in vars(args).items()}})
