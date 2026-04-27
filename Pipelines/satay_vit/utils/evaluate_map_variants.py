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
    # Cast everything to native Python floats immediately to prevent JSON errors
    x1, y1, w1, h1 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
    x2, y2, w2, h2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])
    
    xA = max(x1, x2);  yA = max(y1, y2)
    xB = min(x1+w1, x2+w2); yB = min(y1+h1, y2+h2)
    inter = max(0, xB-xA) * max(0, yB-yA)
    
    return float(inter / (w1*h1 + w2*h2 - inter + 1e-6))


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
    use_original_data=False,
    iou_thresholds=[0.5]  # <--- NEW PARAMETER (defaults to original behavior)
):
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Train the model first!")
        return

    # Ensure iou_thresholds is a list even if a single float is passed
    if isinstance(iou_thresholds, float):
        iou_thresholds = [iou_thresholds]

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(weights_path))
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ models
    print("Loading Models...")
    yolo_weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "yolo11n.pt")
    yolo_model = YOLO(yolo_weights_path).to(device)
    

    if "e2e" in weights_path.lower():
        if "Version_5" in weights_path:
            # Peek at the checkpoint FIRST to detect which V5 architecture was saved:
            #   Old V5 (deprecated): had vit_head.raw_knowledge + vit_head.knowledge_proj.*
            #   New V5 (current):    has vit_head.task_embedding.weight  (same as V2)
            checkpoint  = torch.load(weights_path, map_location=device)
            sd_keys     = set(checkpoint["state_dict"].keys())
            knowledge_path = os.path.join(
                os.path.dirname(weights_path), "raw_knowledge_vectors.pt"
            )
            if "vit_head.raw_knowledge" in sd_keys:
                # ── Old V5: Linear-projection checkpoint ─────────────────────
                # The current model_e2e.py uses the new free-embedding arch,
                # so we fall back to the V2 model class (identical state_dict shape
                # minus the projection params which are absent in V2/new-V5).
                print("  [WARNING] Old V5 checkpoint detected (Linear projection arch).")
                print("  Loading with strict=False — results may be sub-optimal.")
                print("  Retrain V5 with the new CLIP-seeded embedding model for best results.")
                from Version_5.model_e2e import SATAYViT_E2E
                vit_model = SATAYViT_E2E(
                    embed_dim=256, knowledge_path=knowledge_path
                ).to(device)
                missing, unexpected = vit_model.load_state_dict(
                    checkpoint["state_dict"], strict=False
                )
                if missing:
                    print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
                if unexpected:
                    print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
            else:
                # ── New V5: CLIP-seeded free-embedding checkpoint ─────────────
                from Version_5.model_e2e import SATAYViT_E2E
                vit_model = SATAYViT_E2E(
                    embed_dim=256, knowledge_path=knowledge_path
                ).to(device)
                vit_model.load_state_dict(checkpoint["state_dict"])
        elif "Version_4" in weights_path:
            from Version_4.model_e2e import SATAYViT_E2E
            knowledge_path = os.path.join(
                os.path.dirname(weights_path), "task_knowledge_vectors.pt"
            )
            vit_model = SATAYViT_E2E(embed_dim=256, knowledge_path=knowledge_path).to(device)
            checkpoint = torch.load(weights_path, map_location=device)
        elif "Version_3" in weights_path:
            from Version_3.model_e2e import SATAYViT_E2E
            vit_model = SATAYViT_E2E(embed_dim=256).to(device)
            checkpoint = torch.load(weights_path, map_location=device)
        else:
            from Version_2.model_e2e import SATAYViT_E2E
            vit_model = SATAYViT_E2E(embed_dim=256).to(device)
            checkpoint = torch.load(weights_path, map_location=device)
        if "Version_5" not in weights_path:
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
    # Dynamic dictionaries based on the thresholds requested
    per_task_satay = {t: {"total": 0, "correct": {iou: 0 for iou in iou_thresholds}} for t in range(1, 15)}
    per_task_yolo  = {t: {"total": 0, "correct": {iou: 0 for iou in iou_thresholds}} for t in range(1, 15)}

    if use_original_data:
        import cv2
        from utils.preprocess_dataset import get_base_samples, letterbox_image_and_boxes
        print("Loading Original Test Dataset...")
        samples = get_base_samples(data_root, split="test")

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
                if img is None: continue

                img_lb, new_boxes = letterbox_image_and_boxes(img, raw_boxes, new_shape=(640,640))
                preferred_gt = [b for b, p in zip(new_boxes, prefs) if p == 1]
                if len(preferred_gt) == 0: continue

                per_task_satay[task_id]["total"] += 1
                per_task_yolo[task_id]["total"]  += 1

                # YOLO inference
                yolo_res = yolo_model(img_lb, verbose=False)[0]
                if len(yolo_res.boxes) == 0:
                    records.append((0.0, {iou: False for iou in iou_thresholds}, {iou: False for iou in iou_thresholds}, task_id))
                    continue

                if len(yolo_res.boxes.xyxy) > 0:
                    y_boxes = torch.tensor([xyxy_to_xywh(b) for b in yolo_res.boxes.xyxy.cpu()])
                else: 
                    records.append((0.0, {iou: False for iou in iou_thresholds}, {iou: False for iou in iou_thresholds}, task_id))
                    continue
                    
                y_confs = yolo_res.boxes.conf.cpu()

                # Dynamic YOLO Evaluation
                yolo_best_box = y_boxes[torch.argmax(y_confs)]
                yolo_max_iou = max([compute_iou(yolo_best_box, gt) for gt in preferred_gt], default=0.0)
                yolo_hits = {iou: yolo_max_iou >= iou for iou in iou_thresholds}

                # ViT fusion
                img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = (img_tensor.to(device) - mean) / std
                img_tensor = img_tensor.unsqueeze(0)

                task_id_t = torch.tensor([task_id], dtype=torch.long, device=device)

                heatmap  = vit_model(img_tensor, task_id_t)[0].cpu()
                best_idx, fused_scores = fuse_yolo_and_vit(y_boxes, y_confs, heatmap)

                if best_idx is None:
                    records.append((0.0, {iou: False for iou in iou_thresholds}, yolo_hits, task_id))
                    continue

                # Dynamic SATAY Evaluation
                satay_best_box = y_boxes[best_idx]
                fused_conf     = fused_scores[best_idx].item()
                satay_max_iou  = max([compute_iou(satay_best_box, gt) for gt in preferred_gt], default=0.0)
                satay_hits     = {iou: satay_max_iou >= iou for iou in iou_thresholds}

                records.append((fused_conf, satay_hits, yolo_hits, task_id))

                for iou in iou_thresholds:
                    if satay_hits[iou]: per_task_satay[task_id]["correct"][iou] += 1
                    if yolo_hits[iou]:  per_task_yolo[task_id]["correct"][iou]  += 1

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
                if len(preferred_gt) == 0: continue

                per_task_satay[task_id]["total"] += 1
                per_task_yolo[task_id]["total"]  += 1

                # --- YOLO inference ---
                yolo_res = yolo_model(image_path, verbose=False)[0]
                if len(yolo_res.boxes) == 0:
                    records.append((0.0, {iou: False for iou in iou_thresholds}, {iou: False for iou in iou_thresholds}, task_id))
                    continue

                y_boxes = torch.tensor([xyxy_to_xywh(b) for b in yolo_res.boxes.xyxy.cpu()])
                y_confs = yolo_res.boxes.conf.cpu()

                # Dynamic YOLO Evaluation
                yolo_best_box = y_boxes[torch.argmax(y_confs)]
                yolo_max_iou = max([compute_iou(yolo_best_box, gt) for gt in preferred_gt], default=0.0)
                yolo_hits = {iou: yolo_max_iou >= iou for iou in iou_thresholds}

                # --- ViT fusion ---
                heatmap  = vit_model(img, task_id_t)[0].cpu()
                best_idx, fused_scores = fuse_yolo_and_vit(y_boxes, y_confs, heatmap)

                if best_idx is None:
                    records.append((0.0, {iou: False for iou in iou_thresholds}, yolo_hits, task_id))
                    continue

                # Dynamic SATAY Evaluation
                satay_best_box = y_boxes[best_idx]
                fused_conf     = fused_scores[best_idx].item()
                satay_max_iou  = max([compute_iou(satay_best_box, gt) for gt in preferred_gt], default=0.0)
                satay_hits     = {iou: satay_max_iou >= iou for iou in iou_thresholds}

                records.append((fused_conf, satay_hits, yolo_hits, task_id))

                for iou in iou_thresholds:
                    if satay_hits[iou]: per_task_satay[task_id]["correct"][iou] += 1
                    if yolo_hits[iou]:  per_task_yolo[task_id]["correct"][iou]  += 1

    # ------------------------------------------------------------------ metrics
    total = len(records)
    
    # Store aggregated metrics dynamically
    top1_satay = {}
    top1_yolo  = {}
    maps       = {}

    for iou in iou_thresholds:
        tp_satay = sum(r[1][iou] for r in records)
        tp_yolo  = sum(r[2][iou] for r in records)
        
        top1_satay[iou] = tp_satay / total if total else 0
        top1_yolo[iou]  = tp_yolo  / total if total else 0
        
        # mAP calculation per threshold
        records_sorted = sorted(records, key=lambda r: r[0], reverse=True)
        cum_tp, cum_fp = 0, 0
        prec_list, rec_list = [], []
        tot_pos = tp_satay or 1 
        
        for r in records_sorted:
            if r[1][iou]: cum_tp += 1
            else:         cum_fp += 1
            prec_list.append(cum_tp / (cum_tp + cum_fp))
            rec_list.append(cum_tp  / tot_pos)
            
        maps[iou] = ap_at_iou(rec_list, prec_list)

    # Per-task accuracy strings dynamically generated
    per_task_results = {}
    for t in range(1, 15):
        n = per_task_satay[t]["total"]
        per_task_results[t] = {
            "task_name": TASK_NAMES[t],
            "total": n,
            "satay_acc": {iou: (per_task_satay[t]["correct"][iou] / n) if n else 0 for iou in iou_thresholds},
            "yolo_acc": {iou: (per_task_yolo[t]["correct"][iou] / n) if n else 0 for iou in iou_thresholds},
        }

    # ------------------------------------------------------------------ print
    primary_iou = iou_thresholds[0]
    delta = top1_satay[primary_iou] - top1_yolo[primary_iou]
    
    print("\n" + "="*70)
    print("  SATAY-ViT EVALUATION RESULTS")
    print("="*70)
    print(f"  Total test samples evaluated : {total:,}")
    
    for iou in iou_thresholds:
        print(f"  Top-1 Task-Aware Acc (SATAY@{iou:.2f}) : {top1_satay[iou]*100:.2f}%")
    print(f"  Top-1 Confidence Acc (YOLO@{primary_iou:.2f})  : {top1_yolo[primary_iou]*100:.2f}%")
    print(f"  Delta (SATAY@{primary_iou:.2f} - YOLO@{primary_iou:.2f})      : {delta*100:+.2f}%")
    
    for iou in iou_thresholds:
        print(f"  mAP@{iou:.2f} (SATAY)                    : {maps[iou]*100:.2f}%")
    print("-"*70)

    # Dynamic Print Headers
    header = f"  {'Task':<22} {'N':>5}"
    for iou in iou_thresholds:
        header += f"  {'SATAY@'+str(iou):>10}"
    header += f"  {'YOLO@'+str(primary_iou):>10}"
    print(header)
    print("-"*70)
    
    for t in range(1, 15):
        r = per_task_results[t]
        row_str = f"  {r['task_name']:<22} {r['total']:>5}"
        for iou in iou_thresholds:
             row_str += f"  {r['satay_acc'][iou]*100:>9.2f}%"
        row_str += f"  {r['yolo_acc'][primary_iou]*100:>9.2f}%"
        print(row_str)
    print("="*70)

    # ------------------------------------------------------------------ save results JSON
    results = {
        "total_samples": total,
        "top1_satay":    top1_satay,
        "top1_yolo":     top1_yolo,
        "map":           maps,
        "per_task":      per_task_results,
    }
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved -> {results_path}")

    # ------------------------------------------------------------------ bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    task_labels = [TASK_NAMES[t] for t in range(1, 15)]
    x = np.arange(len(task_labels))
    
    # We will plot SATAY for all thresholds, plus YOLO for the primary threshold
    num_bars = len(iou_thresholds) + 1
    total_width = 0.8
    bar_w = total_width / num_bars
    
    # Colors for dynamic generation (blues for SATAY, orange for YOLO)
    colors = ["#4C9BE8", "#2A5A8C", "#88BBE8"] 
    
    for idx, iou in enumerate(iou_thresholds):
        accs = [per_task_results[t]["satay_acc"][iou]*100 for t in range(1, 15)]
        offset = (idx - num_bars/2.0 + 0.5) * bar_w
        color = colors[idx % len(colors)]
        ax.bar(x + offset, accs, bar_w, label=f"SATAY @{iou}", color=color, zorder=3)
        
    # Plot YOLO for the primary threshold
    yolo_accs = [per_task_results[t]["yolo_acc"][primary_iou]*100 for t in range(1, 15)]
    offset = (num_bars - 1 - num_bars/2.0 + 0.5) * bar_w
    ax.bar(x + offset, yolo_accs, bar_w, label=f"YOLO @{primary_iou}", color="#E8784C", zorder=3)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    title_str = "SATAY-ViT vs YOLO-only — Per-Task Accuracy"
    if len(iou_thresholds) > 1: title_str += " (Multi-Threshold)"
    ax.set_title(title_str, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.legend(fontsize=11)
    ax.set_axisbelow(True)

    summary_text = "Overall ▶ " + " | ".join([f"SATAY@{iou}: {top1_satay[iou]*100:.1f}%" for iou in iou_thresholds])
    summary_text += f" | YOLO@{primary_iou}: {top1_yolo[primary_iou]*100:.1f}%"
    fig.text(0.5, 0.97, summary_text, ha="center", va="top", fontsize=10, color="#444444")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    chart_path = os.path.join(output_dir, "per_task_accuracy.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved    -> {chart_path}\n")

    return results

if __name__ == "__main__":
    evaluate_best_model(
        data_root    = "e:/DVcon/DVcon/Data_Preprocessed",
        weights_path = "E:/DVcon/DVcon/Pipelines/satay_vit/Version_5/weights_e2e/satay_vit_e2e_best.pt",
        output_dir   = "e:/DVcon/DVcon/Pipelines/satay_vit/Version_5",
        use_original_data = False,
        iou_thresholds = [0.5],
    )