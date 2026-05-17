"""
run_eval.py — PureYOLO Baseline Evaluation
===========================================
Evaluates YOLOv11n (max-confidence) as a zero-shot baseline on the COCO-Tasks
test split. No trained weights required beyond the standard YOLO detector.

The highest-confidence detection is taken as the answer for every task —
no task conditioning whatsoever. This establishes the floor that any
task-aware model (TORCA) must beat.

Usage:
    python run_eval.py
    python run_eval.py --data-root /path/to/Data_Preprocessed
    python run_eval.py --yolo-model path/to/yolo11n.pt
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

TASK_NAMES = {
    1:  "Step on something",       2:  "Sit comfortably",
    3:  "Place flowers",           4:  "Get potatoes out of fire",
    5:  "Water plant",             6:  "Get lemon out of tea",
    7:  "Dig hole",                8:  "Open bottle of beer",
    9:  "Open parcel",             10: "Serve wine",
    11: "Pour sugar",              12: "Smear butter",
    13: "Extinguish fire",         14: "Pound carpet",
}


def xywh_to_xyxy(boxes):
    """[x, y, w, h] → [x1, y1, x2, y2]"""
    out = np.array(boxes, dtype=np.float32)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def iou(box1, boxes):
    """box1: [4], boxes: [N, 4] — all xyxy. Returns [N] IoUs."""
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1   = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-6)


def evaluate(data_root, yolo_model_path, iou_thresh=0.5, conf_thresh=0.05):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"YOLO model : {yolo_model_path}")
    print(f"Data root  : {data_root}\n")

    index_path = os.path.join(data_root, "test", "samples.json")
    with open(index_path) as f:
        samples = json.load(f)

    model = YOLO(yolo_model_path)

    per_task = {t: {"total": 0, "correct": 0} for t in range(1, 15)}
    records  = []   # (score, hit) for mAP

    for s in tqdm(samples, desc="Evaluating"):
        task_id  = s["task_id"]
        boxes_gt = np.array(s["boxes"], dtype=np.float32)
        prefs    = np.array(s["prefs"], dtype=np.int32)
        preferred = boxes_gt[prefs == 1]

        if len(preferred) == 0:
            continue

        preferred_xyxy = xywh_to_xyxy(preferred)

        result = model.predict(
            s["image_path"],
            conf=conf_thresh,
            verbose=False,
            imgsz=640,
        )[0]

        per_task[task_id]["total"] += 1

        if result.boxes is None or len(result.boxes) == 0:
            records.append((0.0, False))
            continue

        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_confs = result.boxes.conf.cpu().numpy()
        best_idx   = int(np.argmax(pred_confs))
        best_box   = pred_boxes[best_idx]
        best_conf  = float(pred_confs[best_idx])

        max_iou = float(iou(best_box, preferred_xyxy).max())
        hit     = max_iou >= iou_thresh

        per_task[task_id]["correct"] += int(hit)
        records.append((best_conf, hit))

    # ── Overall metrics ───────────────────────────────────────────────
    total   = len(records)
    correct = sum(r[1] for r in records)
    top1    = correct / total if total else 0.0

    # mAP@iou_thresh (11-point interpolation)
    sorted_r = sorted(records, key=lambda r: r[0], reverse=True)
    cum_tp, cum_fp = 0, 0
    prec_l, rec_l  = [], []
    tot_pos = correct or 1
    for score, hit in sorted_r:
        if hit:
            cum_tp += 1
        else:
            cum_fp += 1
        prec_l.append(cum_tp / (cum_tp + cum_fp))
        rec_l.append(cum_tp / tot_pos)
    recalls    = np.concatenate([[0.0], rec_l,  [1.0]])
    precisions = np.concatenate([[1.0], prec_l, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    map_score = float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))

    print("\n" + "=" * 60)
    print("  YOLO Max-Confidence Baseline  —  COCO-Tasks Evaluation")
    print("=" * 60)
    print(f"  Samples evaluated : {total:,}")
    print(f"  Top-1 @ IoU {iou_thresh:.2f}   : {top1 * 100:.2f}%")
    print(f"  mAP  @ IoU {iou_thresh:.2f}   : {map_score * 100:.2f}%")
    print("=" * 60)
    print(f"\n  {'Task':<32} {'N':>5}  {'Top-1':>6}")
    print(f"  {'─'*32}  {'─'*5}  {'─'*6}")
    for t in range(1, 15):
        n   = per_task[t]["total"]
        acc = per_task[t]["correct"] / n * 100 if n else 0.0
        print(f"  {TASK_NAMES[t]:<32} {n:>5}  {acc:>5.1f}%")

    # ── Save results ──────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {
        "total_samples": total,
        "top1": round(top1, 4),
        "map":  round(map_score, 4),
        "iou_threshold": iou_thresh,
        "yolo_model": yolo_model_path,
        "per_task": {
            str(t): {
                "task_name": TASK_NAMES[t],
                "total":     per_task[t]["total"],
                "correct":   per_task[t]["correct"],
                "acc":       round(per_task[t]["correct"] / per_task[t]["total"], 4)
                             if per_task[t]["total"] else 0.0,
            }
            for t in range(1, 15)
        },
    }
    res_path = os.path.join(out_dir, "eval_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {res_path}")

    # ── Per-task bar chart ────────────────────────────────────────────
    labels = [TASK_NAMES[t] for t in range(1, 15)]
    accs   = [per_task[t]["correct"] / per_task[t]["total"] * 100
              if per_task[t]["total"] else 0.0 for t in range(1, 15)]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(14), accs, color="#4C9BE8", zorder=3)
    ax.set_xticks(np.arange(14))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_title(
        f"YOLO Max-Confidence Baseline  —  Per-Task Top-1@{iou_thresh:.2f}\n"
        f"Overall: {top1*100:.1f}%  |  mAP: {map_score*100:.1f}%",
        fontsize=10,
    )
    plt.tight_layout()
    chart_path = os.path.join(out_dir, "per_task_accuracy.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Chart saved   → {chart_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-root",   default="e:/DVcon/DVcon/Data_Preprocessed",
                        help="Root of the preprocessed COCO-Tasks dataset.")
    parser.add_argument("--yolo-model",  default="yolov8s-worldv2.pt",
                        help="YOLO-World checkpoint.")
    parser.add_argument("--iou-thresh",  type=float, default=0.5)
    parser.add_argument("--conf-thresh", type=float, default=0.05)
    args = parser.parse_args()
    evaluate(args.data_root, args.yolo_model, args.iou_thresh, args.conf_thresh)
