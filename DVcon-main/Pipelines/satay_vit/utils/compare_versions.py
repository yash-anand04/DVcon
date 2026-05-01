import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader

# Add parent directory to path to import models easily
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.data_loader import COCOTasksDataset, custom_collate
from utils.inference import fuse_yolo_and_vit

# Task names from COCO-Tasks paper
TASK_NAMES = {
    1:  "Step on something",         2:  "Sit comfortably",      3:  "Place flowers",    
    4:  "Get potatoes out of fire",  5:  "Water plant",          6:  "Get lemon out of tea",
    7:  "Dig hole",                  8:  "Open bottle of beer",  9:  "Open parcel", 
    10: "Serve wine",                11: "Pour sugar",           12: "Smear butter",
    13: "Extinguish fire",           14: "Pound carpet",
}

VERSION_REGISTRY = {
    "V1": {
        "module": "Version_1.model",
        "class": "MEViTReasoner",
        "weights": os.path.join(ROOT_DIR, "Version_1", "weights", "mevit_best.pt"),
        "grid_size": 16,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
        "extra_args": {}
    },
    "V2": {
        "module": "Version_2.model_e2e",
        "class": "SATAYViT_E2E",
        "weights": os.path.join(ROOT_DIR, "Version_2", "weights_e2e", "satay_vit_e2e_best.pt"),
        "grid_size": 16,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
        "extra_args": {}
    },
    "V3": {
        "module": "Version_3.model_e2e",
        "class": "SATAYViT_E2E",
        "weights": os.path.join(ROOT_DIR, "Version_3", "weights", "satay_vit_e2e_best.pt"),
        "grid_size": 32,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed_32",
        "extra_args": {}
    },
    "V4": {
        "module": "Version_4.model_e2e",
        "class": "SATAYViT_E2E",
        "weights": os.path.join(ROOT_DIR, "Version_4", "weights_e2e", "satay_vit_e2e_best.pt"),
        "grid_size": 16,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
        "extra_args": {
            "knowledge_path": os.path.join(ROOT_DIR, "Version_4", "weights_e2e", "task_knowledge_vectors.pt")
        }
    },
    "V5": {
        "module": "Version_5.model_e2e",
        "class": "SATAYViT_E2E",
        "weights": os.path.join(ROOT_DIR, "Version_5", "weights_e2e", "satay_vit_e2e_best.pt"),
        "grid_size": 16,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
        "extra_args": {
            "knowledge_path": os.path.join(ROOT_DIR, "Version_5", "weights_e2e", "raw_knowledge_vectors.pt")
        }
    }
}

def compute_iou(box1, box2):
    """Computes IoU between two boxes in [x, y, w, h] format, casting to float for consistency."""
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
    recalls    = np.concatenate([[0.], np.array(recalls),    [1.]])
    precisions = np.concatenate([[1.], np.array(precisions), [0.]])
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx+1]-recalls[idx]) * precisions[idx+1]))

def evaluate_version(v_name, v_cfg, yolo_model, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"\n>>> Evaluating {v_name} ...")
    
    # 1. Load Model
    import importlib
    mod = importlib.import_module(v_cfg["module"])
    model_class = getattr(mod, v_cfg["class"])
    
    if "knowledge_path" in v_cfg["extra_args"]:
        # Simplified for current V5/V4 (no old V5 logic as requested)
        model = model_class(embed_dim=256, knowledge_path=v_cfg["extra_args"]["knowledge_path"]).to(device)
    else:
        model = model_class().to(device)
        
    # 2. Load Weights
    if os.path.exists(v_cfg["weights"]):
        ckpt = torch.load(v_cfg["weights"], map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print(f"  WARNING: Weights not found for {v_name} at {v_cfg['weights']}")
        return None

    model.eval()
    
    # 3. Loader
    dataset = COCOTasksDataset(v_cfg["data_root"], split="test", grid_size=v_cfg["grid_size"])
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=custom_collate)
    
    records = []
    per_task = {t: {"correct": 0, "total": 0} for t in range(1, 15)}
    yolo_perf = {t: {"correct": 0, "total": 0} for t in range(1, 15)}

    iou_thresh = 0.5
    relevance_floor = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Testing {v_name}"):
            img        = batch["image"].to(device)
            task_id_t  = batch["task_id"].to(device)
            task_id    = task_id_t.item()
            prefs      = batch["prefs"][0]
            gt_boxes   = batch["boxes"][0]
            image_path = batch["image_paths"][0]

            preferred_gt = gt_boxes[prefs == 1]
            if len(preferred_gt) == 0: continue

            per_task[task_id]["total"] += 1
            yolo_perf[task_id]["total"] += 1

            # YOLO inference
            yolo_res = yolo_model(image_path, verbose=False)[0]
            if len(yolo_res.boxes) == 0:
                records.append((0.0, False, False, task_id))
                continue

            y_boxes = torch.tensor([xyxy_to_xywh(b) for b in yolo_res.boxes.xyxy.cpu()])
            y_confs = yolo_res.boxes.conf.cpu()

            # YOLO-only baseline (aligned with map_variants logic)
            yolo_best_box = y_boxes[torch.argmax(y_confs)]
            yolo_max_iou = max([compute_iou(yolo_best_box, gt) for gt in preferred_gt], default=0.0)
            yolo_hit = yolo_max_iou >= iou_thresh
            if yolo_hit: yolo_perf[task_id]["correct"] += 1

            # SATAY-ViT fusion
            heatmap = model(img, task_id_t)[0].cpu()
            best_idx, fused_scores = fuse_yolo_and_vit(y_boxes, y_confs, heatmap, relevance_floor=relevance_floor)

            if best_idx is None:
                records.append((0.0, False, yolo_hit, task_id))
                continue

            # SATAY evaluation (aligned with map_variants logic)
            satay_best_box = y_boxes[best_idx]
            fused_conf     = fused_scores[best_idx].item()
            satay_max_iou  = max([compute_iou(satay_best_box, gt) for gt in preferred_gt], default=0.0)
            satay_hit      = satay_max_iou >= iou_thresh
            
            records.append((fused_conf, satay_hit, yolo_hit, task_id))
            if satay_hit: per_task[task_id]["correct"] += 1

    # 4. Metrics
    total = len(records)
    tp_satay = sum(r[1] for r in records)
    tp_yolo  = sum(r[2] for r in records)
    
    top1_satay = tp_satay / total if total > 0 else 0
    top1_yolo  = tp_yolo / total if total > 0 else 0
    
    # Simple mAP calculation
    records_sorted = sorted(records, key=lambda r: r[0], reverse=True)
    cum_tp, cum_fp = 0, 0
    prec_list, rec_list = [], []
    tot_pos = tp_satay or 1
    for _, is_tp, _, _ in records_sorted:
        if is_tp: cum_tp += 1
        else:      cum_fp += 1
        prec_list.append(cum_tp / (cum_tp + cum_fp))
        rec_list.append(cum_tp  / tot_pos)
    map50 = ap_at_iou(rec_list, prec_list)

    result = {
        "top1_satay": top1_satay,
        "top1_yolo":  top1_yolo,
        "map50":      map50,
        "per_task":   {t: per_task[t]["correct"]/per_task[t]["total"] if per_task[t]["total"] > 0 else 0 for t in range(1, 15)}
    }
    return result

def plot_comparison(all_results):
    os.makedirs(os.path.join(ROOT_DIR, "comparison_plots"), exist_ok=True)
    versions = list(all_results.keys())
    
    # Harmonious color palette (curated blues/teals)
    colors = ["#4C9BE8", "#2A5A8C", "#88BBE8", "#1E4B6E", "#5EB1FF"]
    
    # ------------------------------------------------------------------ Overall Top-1 Plot
    # Modified: REMOVED YOLO baseline as requested.
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    satay_accs = [all_results[v]["top1_satay"]*100 for v in versions]
    
    x = np.arange(len(versions))
    bars = plt.bar(x, satay_accs, 0.6, color=colors[:len(versions)], zorder=3)
    
    plt.xticks(x, versions)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('SATAY-ViT Version Comparison: Overall Top-1 Task Accuracy', fontsize=14, fontweight="bold")
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Summary header text
    summary_text = "Overall ▶ " + " | ".join([f"{v}: {all_results[v]['top1_satay']*100:.1f}%" for v in versions])
    plt.text(0.5, 1.05, summary_text, ha="center", va="top", transform=ax.transAxes, fontsize=10, color="#444444")
    
    plt.ylim(0, 100)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(ROOT_DIR, "comparison_plots", "top1_comparison.png"), dpi=200)
    plt.show()

    # ------------------------------------------------------------------ Per-Task Accuracy Plot
    # Adopting clean style from evaluate_map_variants.py
    fig, ax = plt.subplots(figsize=(14, 6))
    task_labels = [TASK_NAMES[t] for t in range(1, 15)]
    x_tasks = np.arange(len(task_labels))
    
    num_versions = len(versions)
    total_width = 0.85
    bar_w = total_width / num_versions
    
    for idx, v in enumerate(versions):
        accs = [all_results[v]["per_task"][t]*100 for t in range(1, 15)]
        offset = (idx - num_versions/2.0 + 0.5) * bar_w
        ax.bar(x_tasks + offset, accs, bar_w, label=v, color=colors[idx % len(colors)], zorder=3)
    
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("SATAY-ViT Architectural Comparison — Per-Task Success Rate", fontsize=14, fontweight="bold")
    ax.set_xticks(x_tasks)
    ax.set_xticklabels(task_labels, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.legend(fontsize=11)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "comparison_plots", "per_task_accuracy.png"), dpi=200)
    plt.show()

    # ------------------------------------------------------------------ mAP Comparison Plot
    plt.figure(figsize=(14, 6))
    maps = [all_results[v]["map50"]*100 for v in versions]
    plt.bar(versions, maps, color=colors[:len(versions)], zorder=3)
    plt.ylabel('mAP @ 0.5 (%)', fontsize=12)
    plt.title('SATAY-ViT Version Comparison: mAP @ 0.5', fontsize=14, fontweight="bold")
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, "comparison_plots", "map_comparison.png"), dpi=200)
    plt.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Comparing versions on {device}")
    
    yolo_weights = os.path.join(ROOT_DIR, "weights", "yolo11n.pt")
    yolo_model = YOLO(yolo_weights).to(device)
    
    all_results = {}
    for v_name, v_cfg in VERSION_REGISTRY.items():
        res = evaluate_version(v_name, v_cfg, yolo_model, device)
        if res:
            all_results[v_name] = res
            
    # Save results
    with open(os.path.join(ROOT_DIR, "comparison_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
        
    # Plot
    plot_comparison(all_results)
    print("\nComparison Complete! Results saved to comparison_results.json and comparison_plots/")

if __name__ == "__main__":
    main()
