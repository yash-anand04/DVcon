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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import DEFAULT_YOLO_PATH, DetectionFeatureExtractor, TaskDrivenAttentionModel, gather_topk_like_model
from train import pairwise_iou_xyxy, xywh_to_xyxy
from utils.data_loader import COCOTasksDataset, custom_collate


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

TASK_NAMES = {
    1: "Step on something", 2: "Sit comfortably", 3: "Place flowers",
    4: "Get potatoes out of fire", 5: "Water plant", 6: "Get lemon out of tea",
    7: "Dig hole", 8: "Open bottle of beer", 9: "Open parcel",
    10: "Serve wine", 11: "Pour sugar", 12: "Smear butter",
    13: "Extinguish fire", 14: "Pound carpet",
}

DEFAULTS = dict(
    data_root="e:/DVcon/DVcon/Data_Preprocessed",
    weights_path=os.path.join(CURRENT_DIR, "weights", "v1b_attention_best.pt"),
    output_dir=CURRENT_DIR,
    yolo_model=DEFAULT_YOLO_PATH,
    batch=1,
    num_workers=0,
    top_k=20,
    iou_thresholds=[0.5],
    score_mode="prob",
)


def ap_at_iou(recalls, precisions):
    recalls = np.concatenate([[0.0], np.array(recalls), [1.0]])
    precisions = np.concatenate([[1.0], np.array(precisions), [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def load_pil_images(paths):
    return [Image.open(path).convert("RGB") for path in paths]


def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    dataset = COCOTasksDataset(cfg["data_root"], split="test", grid_size=16)
    loader = DataLoader(
        dataset, batch_size=cfg["batch"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=custom_collate
    )

    extractor = DetectionFeatureExtractor(yolo_model=cfg["yolo_model"], device=device)
    extractor.eval()

    model = TaskDrivenAttentionModel(num_classes=80, num_tasks=14, top_k=cfg["top_k"]).to(device)
    if os.path.exists(cfg["weights_path"]):
        ckpt = torch.load(cfg["weights_path"], map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"V1B weights not found: {cfg['weights_path']}")
    model.eval()

    iou_thresholds = cfg["iou_thresholds"]
    if isinstance(iou_thresholds, float):
        iou_thresholds = [iou_thresholds]

    records = []
    per_task_v1b = {t: {"total": 0, "correct": {iou: 0 for iou in iou_thresholds}} for t in range(1, 15)}
    per_task_yolo = {t: {"total": 0, "correct": {iou: 0 for iou in iou_thresholds}} for t in range(1, 15)}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating V1B"):
            images = load_pil_images(batch["image_paths"])
            class_onehot, visual_feats, det_scores, mask, boxes = extractor.extract_with_boxes(images)
            p_final, _, _ = model(class_onehot, visual_feats, det_scores, mask)

            topk_boxes = gather_topk_like_model(boxes, det_scores, model.top_k)
            topk_scores = gather_topk_like_model(det_scores, det_scores, model.top_k)
            topk_mask = gather_topk_like_model(mask, det_scores, model.top_k)

            for b in range(topk_boxes.size(0)):
                task_id = int(batch["task_id"][b].item())
                task_idx = task_id - 1
                gt_boxes = batch["boxes"][b].to(device)
                prefs = batch["prefs"][b].to(device)
                preferred = gt_boxes[prefs == 1]
                if preferred.numel() == 0:
                    continue

                valid = ~topk_mask[b]
                if not valid.any():
                    hit_dict = {iou: False for iou in iou_thresholds}
                    yolo_hit_dict = {iou: False for iou in iou_thresholds}
                    records.append((0.0, hit_dict, yolo_hit_dict, task_id))
                    continue

                per_task_v1b[task_id]["total"] += 1
                per_task_yolo[task_id]["total"] += 1

                preferred_xyxy = xywh_to_xyxy(preferred)

                task_probs = p_final[b, :, task_idx]
                if cfg["score_mode"] == "fused":
                    rank_scores = task_probs * topk_scores[b]
                else:
                    rank_scores = task_probs
                rank_scores = rank_scores.masked_fill(~valid, -1.0)

                best_idx = torch.argmax(rank_scores).item()
                v1b_box = topk_boxes[b, best_idx].unsqueeze(0)
                v1b_max_iou = pairwise_iou_xyxy(v1b_box, preferred_xyxy).max().item()
                hit_dict = {iou: v1b_max_iou >= iou for iou in iou_thresholds}

                yolo_scores = topk_scores[b].masked_fill(~valid, -1.0)
                yolo_idx = torch.argmax(yolo_scores).item()
                yolo_box = topk_boxes[b, yolo_idx].unsqueeze(0)
                yolo_max_iou = pairwise_iou_xyxy(yolo_box, preferred_xyxy).max().item()
                yolo_hit_dict = {iou: yolo_max_iou >= iou for iou in iou_thresholds}

                score = float(rank_scores[best_idx].item())
                records.append((score, hit_dict, yolo_hit_dict, task_id))

                for iou in iou_thresholds:
                    if hit_dict[iou]:
                        per_task_v1b[task_id]["correct"][iou] += 1
                    if yolo_hit_dict[iou]:
                        per_task_yolo[task_id]["correct"][iou] += 1

    total = len(records)
    top1_v1b = {}
    top1_yolo = {}
    maps = {}

    for iou in iou_thresholds:
        tp_v1b = sum(r[1][iou] for r in records)
        tp_yolo = sum(r[2][iou] for r in records)
        top1_v1b[iou] = tp_v1b / total if total else 0.0
        top1_yolo[iou] = tp_yolo / total if total else 0.0

        sorted_records = sorted(records, key=lambda r: r[0], reverse=True)
        cum_tp, cum_fp = 0, 0
        prec_list, rec_list = [], []
        tot_pos = tp_v1b or 1
        for _, hits, _, _ in sorted_records:
            if hits[iou]:
                cum_tp += 1
            else:
                cum_fp += 1
            prec_list.append(cum_tp / (cum_tp + cum_fp))
            rec_list.append(cum_tp / tot_pos)
        maps[iou] = ap_at_iou(rec_list, prec_list)

    per_task_results = {}
    for t in range(1, 15):
        n = per_task_v1b[t]["total"]
        per_task_results[t] = {
            "task_name": TASK_NAMES[t],
            "total": n,
            "v1b_acc": {iou: per_task_v1b[t]["correct"][iou] / n if n else 0.0 for iou in iou_thresholds},
            "yolo_acc": {iou: per_task_yolo[t]["correct"][iou] / n if n else 0.0 for iou in iou_thresholds},
        }

    primary_iou = iou_thresholds[0]
    print("\n" + "=" * 70)
    print("  SATAY-ViT V1B EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Total test samples evaluated : {total:,}")
    for iou in iou_thresholds:
        print(f"  Top-1 Task-Aware Acc (V1B@{iou:.2f}) : {top1_v1b[iou] * 100:.2f}%")
    print(f"  Top-1 Confidence Acc (YOLO@{primary_iou:.2f}) : {top1_yolo[primary_iou] * 100:.2f}%")
    for iou in iou_thresholds:
        print(f"  mAP@{iou:.2f} (V1B) : {maps[iou] * 100:.2f}%")
    print("=" * 70)

    results = {
        "total_samples": total,
        "top1_v1b": top1_v1b,
        "top1_yolo": top1_yolo,
        "map": maps,
        "score_mode": cfg["score_mode"],
        "per_task": per_task_results,
    }
    results_path = os.path.join(cfg["output_dir"], "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [TASK_NAMES[t] for t in range(1, 15)]
    x = np.arange(len(labels))
    width = 0.38
    v1b_acc = [per_task_results[t]["v1b_acc"][primary_iou] * 100 for t in range(1, 15)]
    yolo_acc = [per_task_results[t]["yolo_acc"][primary_iou] * 100 for t in range(1, 15)]
    ax.bar(x - width / 2, v1b_acc, width, label="V1B Attention", color="#4C9BE8", zorder=3)
    ax.bar(x + width / 2, yolo_acc, width, label="YOLO-only", color="#E8784C", zorder=3)
    ax.set_xlabel("Task")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("SATAY-ViT V1B vs YOLO-only - Per-Task Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.legend()
    fig.text(
        0.5, 0.97,
        f"Overall: V1B {top1_v1b[primary_iou] * 100:.1f}% | YOLO {top1_yolo[primary_iou] * 100:.1f}% | mAP {maps[primary_iou] * 100:.1f}%",
        ha="center", va="top", fontsize=10, color="#444444"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = os.path.join(cfg["output_dir"], "per_task_accuracy.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Results saved -> {results_path}")
    print(f"Chart saved   -> {chart_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULTS["data_root"])
    parser.add_argument("--weights-path", type=str, default=DEFAULTS["weights_path"])
    parser.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"])
    parser.add_argument("--yolo-model", type=str, default=DEFAULTS["yolo_model"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    parser.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=DEFAULTS["iou_thresholds"])
    parser.add_argument("--score-mode", choices=["prob", "fused"], default=DEFAULTS["score_mode"])
    args = parser.parse_args()
    evaluate({**DEFAULTS, **vars(args)})
