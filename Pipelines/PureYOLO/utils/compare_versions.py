"""
Unified comparison of all SATAY-ViT pipeline versions.

mAP@0.5 is computed per-task then averaged across 14 tasks, matching the
Sawatzky et al. reference paper ("What Object Should I Use?", 2019).
"""

import os
import sys
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from torch.utils.data import DataLoader
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.data_loader import COCOTasksDataset, custom_collate
from utils.inference import fuse_yolo_and_vit

TASK_NAMES = {
    1:  "Step on something",         2:  "Sit comfortably",      3:  "Place flowers",
    4:  "Get potatoes out of fire",  5:  "Water plant",          6:  "Get lemon out of tea",
    7:  "Dig hole",                  8:  "Open bottle of beer",  9:  "Open parcel",
    10: "Serve wine",                11: "Pour sugar",           12: "Smear butter",
    13: "Extinguish fire",           14: "Pound carpet",
}

VERSION_REGISTRY = {
    "V1": {
        "type": "grid",
        "module": "Version_1.model",
        "class": "MEViTReasoner",
        "weights": os.path.join(ROOT_DIR, "Version_1", "weights", "mevit_best.pt"),
        "grid_size": 16,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
        "extra_args": {}
    },
    "V2": {
        "type": "grid",
        "module": "Version_2.model_e2e",
        "class": "SATAYViT_E2E",
        "weights": os.path.join(ROOT_DIR, "Version_2", "weights_e2e", "satay_vit_e2e_best.pt"),
        "grid_size": 16,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
        "extra_args": {}
    },
    "V3": {
        "type": "grid",
        "module": "Version_3.model_e2e",
        "class": "SATAYViT_E2E",
        "weights": os.path.join(ROOT_DIR, "Version_3", "weights", "satay_vit_e2e_best.pt"),
        "grid_size": 32,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed_32",
        "extra_args": {}
    },
    "V4": {
        "type": "grid",
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
        "type": "grid",
        "module": "Version_5.model_e2e",
        "class": "SATAYViT_E2E",
        "weights": os.path.join(ROOT_DIR, "Version_5", "weights_e2e", "satay_vit_e2e_best.pt"),
        "grid_size": 16,
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
        "extra_args": {
            "knowledge_path": os.path.join(ROOT_DIR, "Version_5", "weights_e2e", "raw_knowledge_vectors.pt")
        }
    },
    "V1B": {
        "type": "object_centric",
        "weights": os.path.join(ROOT_DIR, "Version_1B", "weights", "v1b_attention_best.pt"),
        "yolo_path": os.path.join(ROOT_DIR, "Version_1B", "yolo11n.pt"),
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
        "top_k": 20,
    },
    "V6": {
        "type": "roi_align",
        "weights": os.path.join(ROOT_DIR, "Version_6", "weights", "v6_best.pt"),
        "yolo_path": os.path.join(ROOT_DIR, "Version_6", "weights", "yolo11n.pt"),
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
    },
    "V7": {
        # V7 = V6 + CLIP-text task embeddings + focal/soft-IoU training
        "type": "roi_align_v7",
        "weights": os.path.join(ROOT_DIR, "Version_7", "weights", "v7_best.pt"),
        "yolo_path": os.path.join(ROOT_DIR, "Version_7", "weights", "yolo11n.pt"),
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
    },
    "V8": {
        # V8 = V7 architecture + CLIP student-teacher targets (CLIP image+text encoder at train time only)
        "type": "roi_align_v8",
        "weights": os.path.join(ROOT_DIR, "Version_8", "weights", "v8_best.pt"),
        "yolo_path": os.path.join(ROOT_DIR, "Version_8", "weights", "yolo11n.pt"),
        "data_root": "e:/DVcon/DVcon/Data_Preprocessed",
    },
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_iou_xywh(box1, box2):
    """IoU between two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = float(box1[0]), float(box1[1]), float(box1[2]), float(box1[3])
    x2, y2, w2, h2 = float(box2[0]), float(box2[1]), float(box2[2]), float(box2[3])
    xA, yA = max(x1, x2), max(y1, y2)
    xB, yB = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, xB-xA) * max(0, yB-yA)
    return float(inter / (w1*h1 + w2*h2 - inter + 1e-6))

def pairwise_iou_xyxy(boxes1, boxes2):
    """Pairwise IoU between N xyxy boxes (tensor) and M xyxy boxes (tensor). Returns [N, M]."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    ix1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    return inter / ((area1[:, None] + area2[None, :] - inter) + 1e-6)

def xywh_to_xyxy_tensor(boxes):
    """Convert [B, 4] xywh tensor to xyxy."""
    return torch.stack([boxes[:, 0], boxes[:, 1],
                        boxes[:, 0] + boxes[:, 2],
                        boxes[:, 1] + boxes[:, 3]], dim=1)

def xyxy_to_xywh(b):
    x_min, y_min, x_max, y_max = b
    return [x_min.item(), y_min.item(), (x_max-x_min).item(), (y_max-y_min).item()]

def ap_at_iou(recalls, precisions):
    """Area under the precision-recall curve (11-point interpolation style)."""
    recalls    = np.concatenate([[0.], np.array(recalls),    [1.]])
    precisions = np.concatenate([[1.], np.array(precisions), [0.]])
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx+1]-recalls[idx]) * precisions[idx+1]))

def compute_map50(records):
    """
    Compute mAP@0.5 matching the Sawatzky et al. reference paper:
      - AP@0.5 computed independently per task (recall normalised by task sample count)
      - mAP = mean over all 14 tasks
    records: list of (confidence, is_tp_bool, is_yolo_tp_bool, task_id)
    """
    task_records = {t: [] for t in range(1, 15)}
    for conf, is_tp, _, task_id in records:
        task_records[task_id].append((conf, is_tp))

    aps = []
    for t in range(1, 15):
        t_recs = task_records[t]
        if not t_recs:
            continue
        t_recs_sorted = sorted(t_recs, key=lambda r: r[0], reverse=True)
        n_pos = len(t_recs_sorted)          # one GT preferred object per sample
        cum_tp, cum_fp = 0, 0
        prec_list, rec_list = [], []
        for _, is_tp in t_recs_sorted:
            if is_tp: cum_tp += 1
            else:     cum_fp += 1
            prec_list.append(cum_tp / (cum_tp + cum_fp))
            rec_list.append(cum_tp / n_pos)
        aps.append(ap_at_iou(rec_list, prec_list))

    return float(np.mean(aps)) if aps else 0.0

# ---------------------------------------------------------------------------
# Grid-based (V1–V5) evaluation
# ---------------------------------------------------------------------------

def evaluate_grid_version(v_name, v_cfg, yolo_model, device):
    print(f"\n>>> Evaluating {v_name} ...")

    import importlib
    mod = importlib.import_module(v_cfg["module"])
    model_class = getattr(mod, v_cfg["class"])

    if "knowledge_path" in v_cfg["extra_args"]:
        model = model_class(embed_dim=256, knowledge_path=v_cfg["extra_args"]["knowledge_path"]).to(device)
    else:
        model = model_class().to(device)

    if not os.path.exists(v_cfg["weights"]):
        print(f"  WARNING: Weights not found for {v_name} at {v_cfg['weights']}")
        return None
    ckpt = torch.load(v_cfg["weights"], map_location=device)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.eval()

    dataset = COCOTasksDataset(v_cfg["data_root"], split="test", grid_size=v_cfg["grid_size"])
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                         pin_memory=True, collate_fn=custom_collate)

    records  = []
    per_task = {t: {"correct": 0, "total": 0} for t in range(1, 15)}
    iou_thresh = 0.5

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {v_name}"):
            img        = batch["image"].to(device)
            task_id_t  = batch["task_id"].to(device)
            task_id    = task_id_t.item()
            prefs      = batch["prefs"][0]
            gt_boxes   = batch["boxes"][0]
            image_path = batch["image_paths"][0]

            preferred_gt = gt_boxes[prefs == 1]
            if len(preferred_gt) == 0:
                continue

            per_task[task_id]["total"] += 1

            yolo_res = yolo_model(image_path, verbose=False)[0]
            if len(yolo_res.boxes) == 0:
                records.append((0.0, False, False, task_id))
                continue

            y_boxes = torch.tensor([xyxy_to_xywh(b) for b in yolo_res.boxes.xyxy.cpu()])
            y_confs = yolo_res.boxes.conf.cpu()

            yolo_best_box = y_boxes[torch.argmax(y_confs)]
            yolo_hit = max([compute_iou_xywh(yolo_best_box, gt) for gt in preferred_gt], default=0.0) >= iou_thresh

            heatmap  = model(img, task_id_t)[0].cpu()
            best_idx, fused_scores = fuse_yolo_and_vit(y_boxes, y_confs, heatmap, relevance_floor=0.0)

            if best_idx is None:
                records.append((0.0, False, yolo_hit, task_id))
                continue

            satay_best_box = y_boxes[best_idx]
            fused_conf     = fused_scores[best_idx].item()
            satay_hit      = max([compute_iou_xywh(satay_best_box, gt) for gt in preferred_gt], default=0.0) >= iou_thresh

            records.append((fused_conf, satay_hit, yolo_hit, task_id))
            if satay_hit:
                per_task[task_id]["correct"] += 1

    total = len(records)
    top1  = sum(r[1] for r in records) / total if total > 0 else 0.0
    map50 = compute_map50(records)

    return {
        "top1_satay": top1,
        "top1_yolo":  sum(r[2] for r in records) / total if total > 0 else 0.0,
        "map50":      map50,
        "per_task":   {t: per_task[t]["correct"] / per_task[t]["total"]
                       if per_task[t]["total"] > 0 else 0.0
                       for t in range(1, 15)}
    }

# ---------------------------------------------------------------------------
# Object-centric (V1B) evaluation
# ---------------------------------------------------------------------------

def evaluate_v1b(v_cfg, device):
    print(f"\n>>> Evaluating V1B ...")

    v1b_dir = os.path.join(ROOT_DIR, "Version_1B")
    sys.path.insert(0, v1b_dir)
    from model import DetectionFeatureExtractor, TaskDrivenAttentionModel, gather_topk_like_model
    sys.path.pop(0)

    extractor = DetectionFeatureExtractor(yolo_model=v_cfg["yolo_path"], device=device)
    extractor.resnet.to(device)
    extractor.resnet.eval()

    model = TaskDrivenAttentionModel(num_classes=80, num_tasks=14, top_k=v_cfg["top_k"]).to(device)
    if not os.path.exists(v_cfg["weights"]):
        print(f"  WARNING: V1B weights not found at {v_cfg['weights']}")
        return None
    ckpt  = torch.load(v_cfg["weights"], map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dataset = COCOTasksDataset(v_cfg["data_root"], split="test", grid_size=16)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=0, collate_fn=custom_collate)

    records  = []
    per_task = {t: {"correct": 0, "total": 0} for t in range(1, 15)}
    iou_thresh = 0.5

    with torch.no_grad():
        for batch in tqdm(loader, desc="  V1B"):
            images = [Image.open(p).convert("RGB") for p in batch["image_paths"]]

            class_onehot, visual_feats, det_scores, mask, boxes = extractor.extract_with_boxes(images)
            p_final, _, _ = model(class_onehot, visual_feats, det_scores, mask)

            topk_boxes  = gather_topk_like_model(boxes,      det_scores, model.top_k)  # xyxy
            topk_scores = gather_topk_like_model(det_scores, det_scores, model.top_k)
            topk_mask   = gather_topk_like_model(mask,       det_scores, model.top_k)

            for b in range(topk_boxes.size(0)):
                task_id  = int(batch["task_id"][b].item())
                gt_boxes = batch["boxes"][b].to(device)       # xywh
                prefs    = batch["prefs"][b].to(device)
                preferred = gt_boxes[prefs == 1]
                if preferred.numel() == 0:
                    continue

                valid = ~topk_mask[b]
                per_task[task_id]["total"] += 1

                if not valid.any():
                    records.append((0.0, False, False, task_id))
                    continue

                preferred_xyxy = xywh_to_xyxy_tensor(preferred)

                task_probs  = p_final[b, :, task_id - 1]
                rank_scores = task_probs.masked_fill(~valid, -1.0)
                best_idx    = torch.argmax(rank_scores).item()

                v1b_box = topk_boxes[b, best_idx].unsqueeze(0)    # xyxy
                v1b_iou = pairwise_iou_xyxy(v1b_box, preferred_xyxy).max().item()
                satay_hit = v1b_iou >= iou_thresh

                yolo_idx  = torch.argmax(topk_scores[b].masked_fill(~valid, -1.0)).item()
                yolo_box  = topk_boxes[b, yolo_idx].unsqueeze(0)
                yolo_iou  = pairwise_iou_xyxy(yolo_box, preferred_xyxy).max().item()
                yolo_hit  = yolo_iou >= iou_thresh

                conf = float(rank_scores[best_idx].item())
                records.append((conf, satay_hit, yolo_hit, task_id))
                if satay_hit:
                    per_task[task_id]["correct"] += 1

    total = len(records)
    top1  = sum(r[1] for r in records) / total if total > 0 else 0.0
    map50 = compute_map50(records)

    return {
        "top1_satay": top1,
        "top1_yolo":  sum(r[2] for r in records) / total if total > 0 else 0.0,
        "map50":      map50,
        "per_task":   {t: per_task[t]["correct"] / per_task[t]["total"]
                       if per_task[t]["total"] > 0 else 0.0
                       for t in range(1, 15)}
    }

# ---------------------------------------------------------------------------
# Object-centric V6 (RoI-Align) evaluation
# ---------------------------------------------------------------------------

def evaluate_v6(v_cfg, device):
    print(f"\n>>> Evaluating V6 ...")

    v6_dir = os.path.join(ROOT_DIR, "Version_6")
    sys.path.insert(0, v6_dir)
    from model import SATAYViT_V6, YOLODetector
    sys.path.pop(0)

    detector = YOLODetector(checkpoint=v_cfg["yolo_path"], device=device)
    model    = SATAYViT_V6(checkpoint=v_cfg["yolo_path"]).to(device)

    if not os.path.exists(v_cfg["weights"]):
        print(f"  WARNING: V6 weights not found at {v_cfg['weights']}")
        return None
    ckpt  = torch.load(v_cfg["weights"], map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dataset = COCOTasksDataset(v_cfg["data_root"], split="test", grid_size=16)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False,
                         num_workers=0, collate_fn=custom_collate)

    records  = []
    per_task = {t: {"correct": 0, "total": 0} for t in range(1, 15)}
    iou_thresh = 0.5

    with torch.no_grad():
        for batch in tqdm(loader, desc="  V6"):
            pil_images  = [Image.open(p).convert("RGB") for p in batch["image_paths"]]
            img_tensors = batch["image"].to(device)
            task_ids    = batch["task_id"].to(device)

            det_results                   = detector.detect_batch(pil_images)
            rel_scores, det_scores_t, mask = model(img_tensors, det_results, task_ids)

            B = img_tensors.shape[0]
            for b in range(B):
                task_id  = int(batch["task_id"][b].item())
                gt_boxes = batch["boxes"][b].to(device)
                prefs    = batch["prefs"][b].to(device)
                preferred = gt_boxes[prefs == 1]

                if preferred.numel() == 0:
                    continue

                preferred_xyxy = xywh_to_xyxy_tensor(preferred)
                boxes_b, scores_b, _ = det_results[b]

                per_task[task_id]["total"] += 1

                if boxes_b.shape[0] == 0:
                    records.append((0.0, False, False, task_id))
                    continue

                n_b   = boxes_b.shape[0]
                valid = ~mask[b, :n_b]

                # SATAY-V6: rel × det_conf
                final_scores = rel_scores[b, :n_b] * det_scores_t[b, :n_b]
                final_scores[~valid] = -1.0

                yolo_scores = det_scores_t[b, :n_b].clone()
                yolo_scores[~valid] = -1.0

                best_satay = int(final_scores.argmax())
                best_yolo  = int(yolo_scores.argmax())

                satay_box = boxes_b[best_satay].unsqueeze(0)
                yolo_box  = boxes_b[best_yolo].unsqueeze(0)

                satay_iou = pairwise_iou_xyxy(satay_box, preferred_xyxy).max().item()
                yolo_iou  = pairwise_iou_xyxy(yolo_box,  preferred_xyxy).max().item()

                satay_hit = satay_iou >= iou_thresh
                yolo_hit  = yolo_iou  >= iou_thresh

                records.append((float(final_scores[best_satay].item()), satay_hit, yolo_hit, task_id))
                if satay_hit:
                    per_task[task_id]["correct"] += 1

    total = len(records)
    top1  = sum(r[1] for r in records) / total if total > 0 else 0.0
    map50 = compute_map50(records)

    return {
        "top1_satay": top1,
        "top1_yolo":  sum(r[2] for r in records) / total if total > 0 else 0.0,
        "map50":      map50,
        "per_task":   {t: per_task[t]["correct"] / per_task[t]["total"]
                       if per_task[t]["total"] > 0 else 0.0
                       for t in range(1, 15)}
    }


def evaluate_v7(v_cfg, device):
    """V7 = V6 architecture + CLIP-text task embeddings.  Same evaluation
    procedure as evaluate_v6 but loads SATAYViT_V7 from Version_7."""
    print(f"\n>>> Evaluating V7 ...")

    v7_dir = os.path.join(ROOT_DIR, "Version_7")
    sys.path.insert(0, v7_dir)
    from model import SATAYViT_V7, YOLODetector
    sys.path.pop(0)

    detector = YOLODetector(checkpoint=v_cfg["yolo_path"], device=device)
    model    = SATAYViT_V7(checkpoint=v_cfg["yolo_path"]).to(device)

    if not os.path.exists(v_cfg["weights"]):
        print(f"  WARNING: V7 weights not found at {v_cfg['weights']}")
        return None
    ckpt  = torch.load(v_cfg["weights"], map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dataset = COCOTasksDataset(v_cfg["data_root"], split="test", grid_size=16)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False,
                         num_workers=0, collate_fn=custom_collate)

    records  = []
    per_task = {t: {"correct": 0, "total": 0} for t in range(1, 15)}
    iou_thresh = 0.5

    with torch.no_grad():
        for batch in tqdm(loader, desc="  V7"):
            pil_images  = [Image.open(p).convert("RGB") for p in batch["image_paths"]]
            img_tensors = batch["image"].to(device)
            task_ids    = batch["task_id"].to(device)

            det_results                   = detector.detect_batch(pil_images)
            rel_scores, det_scores_t, mask = model(img_tensors, det_results, task_ids)

            B = img_tensors.shape[0]
            for b in range(B):
                task_id  = int(batch["task_id"][b].item())
                gt_boxes = batch["boxes"][b].to(device)
                prefs    = batch["prefs"][b].to(device)
                preferred = gt_boxes[prefs == 1]

                if preferred.numel() == 0:
                    continue

                preferred_xyxy = xywh_to_xyxy_tensor(preferred)
                boxes_b, scores_b, _ = det_results[b]

                per_task[task_id]["total"] += 1

                if boxes_b.shape[0] == 0:
                    records.append((0.0, False, False, task_id))
                    continue

                n_b   = boxes_b.shape[0]
                valid = ~mask[b, :n_b]

                final_scores = rel_scores[b, :n_b] * det_scores_t[b, :n_b]
                final_scores[~valid] = -1.0
                yolo_scores  = det_scores_t[b, :n_b].clone()
                yolo_scores[~valid] = -1.0

                best_satay = int(final_scores.argmax())
                best_yolo  = int(yolo_scores.argmax())

                satay_box = boxes_b[best_satay].unsqueeze(0)
                yolo_box  = boxes_b[best_yolo].unsqueeze(0)

                satay_iou = pairwise_iou_xyxy(satay_box, preferred_xyxy).max().item()
                yolo_iou  = pairwise_iou_xyxy(yolo_box,  preferred_xyxy).max().item()

                satay_hit = satay_iou >= iou_thresh
                yolo_hit  = yolo_iou  >= iou_thresh

                records.append((float(final_scores[best_satay].item()), satay_hit, yolo_hit, task_id))
                if satay_hit:
                    per_task[task_id]["correct"] += 1

    total = len(records)
    top1  = sum(r[1] for r in records) / total if total > 0 else 0.0
    map50 = compute_map50(records)

    return {
        "top1_satay": top1,
        "top1_yolo":  sum(r[2] for r in records) / total if total > 0 else 0.0,
        "map50":      map50,
        "per_task":   {t: per_task[t]["correct"] / per_task[t]["total"]
                       if per_task[t]["total"] > 0 else 0.0
                       for t in range(1, 15)}
    }


def evaluate_v8(v_cfg, device):
    """V8 = V7 architecture trained with CLIP student-teacher supervision.
    Inference is identical to V6/V7 — CLIP is only used during training."""
    print(f"\n>>> Evaluating V8 ...")

    v8_dir = os.path.join(ROOT_DIR, "Version_8")
    sys.path.insert(0, v8_dir)
    from model import SATAYViT_V8, YOLODetector
    sys.path.pop(0)

    detector = YOLODetector(checkpoint=v_cfg["yolo_path"], device=device)
    model    = SATAYViT_V8(checkpoint=v_cfg["yolo_path"]).to(device)

    if not os.path.exists(v_cfg["weights"]):
        print(f"  WARNING: V8 weights not found at {v_cfg['weights']}")
        return None
    ckpt  = torch.load(v_cfg["weights"], map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    dataset = COCOTasksDataset(v_cfg["data_root"], split="test", grid_size=16)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False,
                         num_workers=0, collate_fn=custom_collate)

    records  = []
    per_task = {t: {"correct": 0, "total": 0} for t in range(1, 15)}
    iou_thresh = 0.5

    with torch.no_grad():
        for batch in tqdm(loader, desc="  V8"):
            pil_images  = [Image.open(p).convert("RGB") for p in batch["image_paths"]]
            img_tensors = batch["image"].to(device)
            task_ids    = batch["task_id"].to(device)

            det_results                   = detector.detect_batch(pil_images)
            rel_scores, det_scores_t, mask = model(img_tensors, det_results, task_ids)

            B = img_tensors.shape[0]
            for b in range(B):
                task_id  = int(batch["task_id"][b].item())
                gt_boxes = batch["boxes"][b].to(device)
                prefs    = batch["prefs"][b].to(device)
                preferred = gt_boxes[prefs == 1]

                if preferred.numel() == 0:
                    continue

                preferred_xyxy = xywh_to_xyxy_tensor(preferred)
                boxes_b, scores_b, _ = det_results[b]

                per_task[task_id]["total"] += 1

                if boxes_b.shape[0] == 0:
                    records.append((0.0, False, False, task_id))
                    continue

                n_b   = boxes_b.shape[0]
                valid = ~mask[b, :n_b]

                final_scores = rel_scores[b, :n_b] * det_scores_t[b, :n_b]
                final_scores[~valid] = -1.0
                yolo_scores  = det_scores_t[b, :n_b].clone()
                yolo_scores[~valid] = -1.0

                best_satay = int(final_scores.argmax())
                best_yolo  = int(yolo_scores.argmax())

                satay_box = boxes_b[best_satay].unsqueeze(0)
                yolo_box  = boxes_b[best_yolo].unsqueeze(0)

                satay_iou = pairwise_iou_xyxy(satay_box, preferred_xyxy).max().item()
                yolo_iou  = pairwise_iou_xyxy(yolo_box,  preferred_xyxy).max().item()

                satay_hit = satay_iou >= iou_thresh
                yolo_hit  = yolo_iou  >= iou_thresh

                records.append((float(final_scores[best_satay].item()), satay_hit, yolo_hit, task_id))
                if satay_hit:
                    per_task[task_id]["correct"] += 1

    total = len(records)
    top1  = sum(r[1] for r in records) / total if total > 0 else 0.0
    map50 = compute_map50(records)

    return {
        "top1_satay": top1,
        "top1_yolo":  sum(r[2] for r in records) / total if total > 0 else 0.0,
        "map50":      map50,
        "per_task":   {t: per_task[t]["correct"] / per_task[t]["total"]
                       if per_task[t]["total"] > 0 else 0.0
                       for t in range(1, 15)}
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# V1-V5: grid-based blues/teals; V1B/V6: orange family (object-centric)
VERSION_COLORS = {
    "V1":  "#88BBE8",
    "V2":  "#4C9BE8",
    "V3":  "#2A5A8C",
    "V4":  "#5EB1FF",
    "V5":  "#1E4B6E",
    "V1B": "#E8784C",
    "V6":  "#C84B0E",  # deeper orange — object-centric, FPN-based
    "V7":  "#7D3C0A",  # burnt sienna — V6 + CLIP task embeddings
    "V8":  "#4A1F05",  # darkest brown — V7 + CLIP student-teacher supervision
}

def plot_comparison(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    versions = list(all_results.keys())
    colors   = [VERSION_COLORS.get(v, "#999999") for v in versions]

    # ── Top-1 Overall ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    satay_accs = [all_results[v]["top1_satay"] * 100 for v in versions]
    x = np.arange(len(versions))
    bars = ax.bar(x, satay_accs, 0.6, color=colors, zorder=3)
    for bar, val in zip(bars, satay_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("SATAY-ViT Version Comparison: Top-1 Task-Aware Accuracy",
                 fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(satay_accs) * 1.15)
    summary = "Overall ▶ " + " | ".join([f"{v}: {all_results[v]['top1_satay']*100:.1f}%" for v in versions])
    fig.text(0.5, 0.97, summary, ha="center", va="top", fontsize=9, color="#444444")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "top1_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── mAP@0.5 Overall ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    maps = [all_results[v]["map50"] * 100 for v in versions]
    bars = ax.bar(x, maps, 0.6, color=colors, zorder=3)
    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=12)
    ax.set_ylabel("mAP @ 0.5 (%)", fontsize=12)
    ax.set_title("SATAY-ViT Version Comparison: mAP@0.5 (per-task AP, then mean)",
                 fontsize=14, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(maps) * 1.15)
    summary = "mAP@0.5 ▶ " + " | ".join([f"{v}: {all_results[v]['map50']*100:.1f}%" for v in versions])
    fig.text(0.5, 0.97, summary, ha="center", va="top", fontsize=9, color="#444444")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "map_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ── Per-Task Accuracy ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7))
    task_labels = [TASK_NAMES[t] for t in range(1, 15)]
    x_tasks     = np.arange(len(task_labels))
    n_ver       = len(versions)
    bar_w       = 0.85 / n_ver

    for idx, v in enumerate(versions):
        accs   = [all_results[v]["per_task"][t] * 100 for t in range(1, 15)]
        offset = (idx - n_ver/2.0 + 0.5) * bar_w
        ax.bar(x_tasks + offset, accs, bar_w,
               label=v, color=VERSION_COLORS.get(v, "#999999"), zorder=3)

    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("SATAY-ViT Architectural Comparison — Per-Task Success Rate",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x_tasks)
    ax.set_xticklabels(task_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_task_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running comparison on: {device}\n")

    yolo_weights = os.path.join(ROOT_DIR, "weights", "yolo11n.pt")
    yolo_model   = YOLO(yolo_weights).to(device)

    all_results = {}
    for v_name, v_cfg in VERSION_REGISTRY.items():
        if v_cfg["type"] == "grid":
            res = evaluate_grid_version(v_name, v_cfg, yolo_model, device)
        elif v_cfg["type"] == "roi_align":
            res = evaluate_v6(v_cfg, device)
        elif v_cfg["type"] == "roi_align_v7":
            res = evaluate_v7(v_cfg, device)
        elif v_cfg["type"] == "roi_align_v8":
            res = evaluate_v8(v_cfg, device)
        else:
            res = evaluate_v1b(v_cfg, device)

        if res is not None:
            all_results[v_name] = res

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_path = os.path.join(ROOT_DIR, "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    # ── Print table ──────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  {'Version':<8}  {'Top-1 Acc':>12}  {'mAP@0.5':>10}  {'YOLO baseline':>14}")
    print("-"*72)
    for v, r in all_results.items():
        print(f"  {v:<8}  {r['top1_satay']*100:>11.2f}%  {r['map50']*100:>9.2f}%  {r['top1_yolo']*100:>13.2f}%")
    print("="*72)
    print("  (mAP@0.5 = per-task AP averaged over 14 tasks, matching ref. paper)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_dir = os.path.join(ROOT_DIR, "comparison_plots")
    plot_comparison(all_results, plot_dir)
    print(f"\nPlots saved -> {plot_dir}/")

if __name__ == "__main__":
    main()
