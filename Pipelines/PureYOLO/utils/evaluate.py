import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import sys
import os
import json
import matplotlib.pyplot as plt

# Add parent directory to path to import models easily
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Version_1.model import MEViTReasoner
from utils.inference import fuse_yolo_and_vit
from utils.data_loader import COCOTasksDataset, custom_collate
from torch.utils.data import DataLoader

# Task names from COCO-Tasks paper
TASK_NAMES = {
    1:  "Step on something",         2:  "Sit comfortably",      3:  "Place flowers",    
    4:  "Get potatoes out of fire",  5:  "Water plant",          6:  "Get lemon out of tea",
    7:  "Dig hole",                  8:  "Open bottle of beer",  9:  "Open parcel", 
    10: "Serve wine",                11: "Pour sugar",           12: "Smear butter",
    13: "Extinguish fire",           14: "Pound carpet",
}


def compute_iou(box1, box2):
    """Computes IoU between two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xA = max(x1, x2);  yA = max(y1, y2)
    xB = min(x1+w1, x2+w2); yB = min(y1+h1, y2+h2)
    inter = max(0, xB-xA) * max(0, yB-yA)
    return inter / float(w1*h1 + w2*h2 - inter + 1e-6)


def xyxy_to_xywh(b):
    x_min, y_min, x_max, y_max = b
    return [x_min.item(), y_min.item(), (x_max-x_min).item(), (y_max-y_min).item()]


def ap_at_iou(recalls, precisions):
    """Compute Area Under the P-R Curve (AUC-PR) using trapezoidal rule."""
    recalls    = np.concatenate([[0.], np.array(recalls),    [1.]])
    precisions = np.concatenate([[1.], np.array(precisions), [0.]])
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx+1]-recalls[idx]) * precisions[idx+1]))


def evaluate_best_model(
    data_root,
    weights_path,
    output_dir=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_original_data=False
):
    """
    Full SATAY-ViT evaluation:
      - Top-1 Task-Aware Accuracy (SATAY-ViT vs YOLO-only baseline)
      - Per-task accuracy breakdown
      - mAP@0.5 for SATAY-ViT
      - Precision & Recall @0.5
    Results are saved as a JSON + PNG chart.
    """
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Train the model first!")
        return

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(weights_path))
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ models
    print("Loading Models...")
    yolo_weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "yolo11n.pt")
    yolo_model = YOLO(yolo_weights_path).to(device)
    
    # Auto-detect architecture based on filename or checkpoint content
    if "e2e" in weights_path.lower():
        if "Version_3" in weights_path:
            from Version_3.model_e2e import SATAYViT_E2E
        else:
            from Version_2.model_e2e import SATAYViT_E2E
        vit_model = SATAYViT_E2E(embed_dim=256).to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        vit_model.load_state_dict(checkpoint["state_dict"])
    else:
        vit_model  = MEViTReasoner().to(device)
        checkpoint = torch.load(weights_path, map_location=device)
        if "state_dict" in checkpoint:
            vit_model.load_state_dict(checkpoint["state_dict"])
        else:
            vit_model.load_state_dict(checkpoint)
        
    vit_model.eval()

    # ------------------------------------------------------------------ loader
    records = []
    per_task_satay  = {t: {"correct": 0, "total": 0} for t in range(1, 15)}
    per_task_yolo   = {t: {"correct": 0, "total": 0} for t in range(1, 15)}

    if use_original_data:
        import cv2
        from utils.preprocess_dataset import get_base_samples, letterbox_image_and_boxes
        print("Loading Original Test Dataset...")
        samples = get_base_samples(data_root, split="test")

        # Standard ViT Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        with torch.no_grad():
            for sample in tqdm(samples, desc="Evaluating Original Data"):
                task_id = sample["task_id"]
                img_path = sample["image_path"]

                raw_boxes = []
                prefs = []
                for ann in sample["annotations"]:
                    raw_boxes.append(ann["bbox"])
                    prefs.append(ann["category_id"])

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_lb, new_boxes = letterbox_image_and_boxes(img, raw_boxes, new_shape=(640,640))
                preferred_gt = [b for b, p in zip(new_boxes, prefs) if p == 1]
                if len(preferred_gt) == 0:
                    continue

                per_task_satay[task_id]["total"] += 1
                per_task_yolo[task_id]["total"]  += 1

                # YOLO inference
                yolo_res = yolo_model(img_lb, verbose=False)[0]
                if len(yolo_res.boxes) == 0:
                    records.append((0.0, False, False, task_id))
                    continue

                if len(yolo_res.boxes.xyxy) > 0:
                    y_boxes   = torch.tensor([xyxy_to_xywh(b) for b in yolo_res.boxes.xyxy.cpu()])
                else: # fallback if empty
                    records.append((0.0, False, False, task_id))
                    continue
                    
                y_confs   = yolo_res.boxes.conf.cpu()

                yolo_best_box = y_boxes[torch.argmax(y_confs)]
                yolo_hit = any(compute_iou(yolo_best_box, gt) > 0.5 for gt in preferred_gt)

                # ViT fusion
                img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = (img_tensor.to(device) - mean) / std
                img_tensor = img_tensor.unsqueeze(0)

                task_id_t = torch.tensor([task_id], dtype=torch.long, device=device)

                heatmap  = vit_model(img_tensor, task_id_t)[0].cpu()
                best_idx, fused_scores = fuse_yolo_and_vit(y_boxes, y_confs, heatmap)

                if best_idx is None:
                    records.append((0.0, False, yolo_hit, task_id))
                    continue

                satay_best_box = y_boxes[best_idx]
                fused_conf     = fused_scores[best_idx].item()
                satay_hit = any(compute_iou(satay_best_box, gt) > 0.5 for gt in preferred_gt)

                records.append((fused_conf, satay_hit, yolo_hit, task_id))

                if satay_hit: per_task_satay[task_id]["correct"] += 1
                if yolo_hit:  per_task_yolo[task_id]["correct"]  += 1

    else:
        print("Loading Test Dataset...")
        if "Version_3" in weights_path:
            val_dataset = COCOTasksDataset(data_root, split="test", grid_size=32)
        else:
            val_dataset = COCOTasksDataset(data_root, split="test", grid_size=16)
        val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                 num_workers=3,pin_memory=True, collate_fn=custom_collate)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                img        = batch["image"].to(device)
                task_id_t  = batch["task_id"].to(device)
                task_id    = task_id_t.item()
                prefs      = batch["prefs"][0]
                gt_boxes   = batch["boxes"][0]
                image_path = batch["image_paths"][0]

                preferred_gt = gt_boxes[prefs == 1]
                if len(preferred_gt) == 0:
                    continue

                per_task_satay[task_id]["total"] += 1
                per_task_yolo[task_id]["total"]  += 1

                # --- YOLO inference ---
                yolo_res = yolo_model(image_path, verbose=False)[0]
                if len(yolo_res.boxes) == 0:
                    records.append((0.0, False, False, task_id))
                    continue

                y_boxes   = torch.tensor([xyxy_to_xywh(b) for b in yolo_res.boxes.xyxy.cpu()])
                y_confs   = yolo_res.boxes.conf.cpu()

                # YOLO-only: pick highest confidence box
                yolo_best_box = y_boxes[torch.argmax(y_confs)]
                yolo_hit = any(compute_iou(yolo_best_box, gt) > 0.5 for gt in preferred_gt)

                # --- ViT fusion ---
                heatmap  = vit_model(img, task_id_t)[0].cpu()
                best_idx, fused_scores = fuse_yolo_and_vit(y_boxes, y_confs, heatmap)

                if best_idx is None:
                    records.append((0.0, False, yolo_hit, task_id))
                    continue

                satay_best_box = y_boxes[best_idx]
                fused_conf     = fused_scores[best_idx].item()
                satay_hit = any(compute_iou(satay_best_box, gt) > 0.5 for gt in preferred_gt)

                records.append((fused_conf, satay_hit, yolo_hit, task_id))

                if satay_hit: per_task_satay[task_id]["correct"] += 1
                if yolo_hit:  per_task_yolo[task_id]["correct"]  += 1

    # ------------------------------------------------------------------ metrics
    total   = len(records)
    tp_satay = sum(r[1] for r in records)
    tp_yolo  = sum(r[2] for r in records)

    top1_satay = tp_satay / total if total else 0
    top1_yolo  = tp_yolo  / total if total else 0
    delta      = top1_satay - top1_yolo

    # mAP@0.5 for SATAY-ViT (sort by fused score descending)
    records_sorted = sorted(records, key=lambda r: r[0], reverse=True)
    cum_tp, cum_fp = 0, 0
    prec_list, rec_list = [], []
    tot_pos = tp_satay or 1  # guard zero div
    for _, is_tp, _, _ in records_sorted:
        if is_tp: cum_tp += 1
        else:      cum_fp += 1
        prec_list.append(cum_tp / (cum_tp + cum_fp))
        rec_list.append(cum_tp  / tot_pos)

    map50 = ap_at_iou(rec_list, prec_list)

    # Precision & Recall at operating point (all positive predictions above 0)
    precision = tp_satay / total        if total    else 0
    recall    = tp_satay / (tp_satay+1) if tp_satay else 0  # simplified

    # Per-task accuracy strings
    per_task_results = {}
    for t in range(1, 15):
        n = per_task_satay[t]["total"]
        cs = per_task_satay[t]["correct"]
        cy = per_task_yolo[t]["correct"]
        per_task_results[t] = {
            "task_name":      TASK_NAMES[t],
            "total":          n,
            "satay_correct":  cs,
            "yolo_correct":   cy,
            "satay_acc":      cs/n if n else 0,
            "yolo_acc":       cy/n if n else 0,
        }

    # ------------------------------------------------------------------ print
    print("\n" + "="*60)
    print("  SATAY-ViT EVALUATION RESULTS")
    print("="*60)
    print(f"  Total test samples evaluated    : {total:,}")
    print(f"  Top-1 Task-Aware Acc (SATAY-ViT): {top1_satay*100:.2f}%")
    print(f"  Top-1 Confidence Acc (YOLO-only): {top1_yolo*100:.2f}%")
    print(f"  Delta (SATAY-ViT - YOLO-only)   : {delta*100:+.2f}%")
    print(f"  mAP@0.5 (SATAY-ViT)             : {map50*100:.2f}%")
    print(f"  Precision@0.5                   : {precision*100:.2f}%")
    print("-"*60)
    print(f"  {'Task':<16} {'N':>5}  {'SATAY%':>8}  {'YOLO%':>8}  {'Delta':>8}")
    print("-"*60)
    for t in range(1, 15):
        r = per_task_results[t]
        d = (r["satay_acc"] - r["yolo_acc"]) * 100
        print(f"  {r['task_name']:<16} {r['total']:>5}  {r['satay_acc']*100:>7.2f}%  {r['yolo_acc']*100:>7.2f}%  {d:>+7.2f}%")
    print("="*60)

    # ------------------------------------------------------------------ save results JSON
    results = {
        "total_samples":  total,
        "top1_satay":     top1_satay,
        "top1_yolo":      top1_yolo,
        "delta":          delta,
        "map50":          map50,
        "precision":      precision,
        "per_task":       per_task_results,
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    # ------------------------------------------------------------------ bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    task_labels = [TASK_NAMES[t] for t in range(1, 15)]
    satay_accs  = [per_task_results[t]["satay_acc"]*100 for t in range(1, 15)]
    yolo_accs   = [per_task_results[t]["yolo_acc"]*100  for t in range(1, 15)]
    x = np.arange(len(task_labels))
    w = 0.38

    bars1 = ax.bar(x - w/2, satay_accs, w, label="SATAY-ViT", color="#4C9BE8", zorder=3)
    bars2 = ax.bar(x + w/2, yolo_accs,  w, label="YOLO-only", color="#E8784C", zorder=3)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("SATAY-ViT vs YOLO-only — Per-Task Top-1 Accuracy", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.legend(fontsize=11)
    ax.set_axisbelow(True)

    # Annotate overall accuracy on top
    fig.text(0.5, 0.97,
             f"Overall ▶  SATAY-ViT: {top1_satay*100:.1f}%  |  YOLO-only: {top1_yolo*100:.1f}%  |  mAP@0.5: {map50*100:.1f}%",
             ha="center", va="top", fontsize=10, color="#444444")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    chart_path = os.path.join(output_dir, "per_task_accuracy.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved    -> {chart_path}\n")

    return results


if __name__ == "__main__":
    evaluate_best_model(
        data_root    = "e:/DVcon/DVcon/Data_Preprocessed_32",
        weights_path = "E:/DVcon/DVcon/Pipelines/satay_vit/Version_3/weights/satay_vit_e2e_best.pt",
        output_dir   = "e:/DVcon/DVcon/Pipelines/satay_vit/Version_3",
        use_original_data = False
    )
