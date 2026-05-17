# TORCA — Final Pipeline

**Task-Aware Object Selection via RoI-Align on YOLO FPN Features**
DVCon India 2026 Design Contest — Stage 2A submission by Team CHIPMunks.

---

## Overview

TORCA is a dual-branch system for natural-language object selection:

1. A **frozen YOLOv11n detector** proposes candidate bounding boxes.
2. A **trainable task-aware scorer** re-ranks those boxes conditioned on the task.

Final score: `S_final = S_rel × S_yolo_conf` — the box with the highest score is the answer.

**Results on COCO-Tasks test split (N = 6173):**

| Method | Top-1 @ IoU 0.5 | mAP @ 0.5 | Params |
|---|---|---|---|
| YOLO only (max confidence) | 15.6% | 8.95% | — |
| Sawatzky et al. (ResNet50) | 56.4% | 41.9% | 25.5 M |
| **TORCA (ours)** | **60.1%** | **52.2%** | **2.56 M** |

---

## File Structure

```
TORCA/
├── model.py                  # TORCA model definition
├── preprocess_dataset.py     # Convert raw COCO-Tasks → training-ready format
├── train.py                  # Training script
├── evaluate.py               # Evaluation on test split
├── infer.py                  # Single-image inference (any image)
├── hardware_analysis.py      # FPGA resource estimator
├── utils/
│   ├── data_loader.py        # COCOTasksDataset + custom_collate
│   └── plot_metrics.py       # Loss curve plotter
└── weights/ (GENERATED AFTER TRAINING)
    ├── torca_best.pt         # Best TORCA checkpoint 
    ├── torca_latest.pt       # Latest checkpoint
    └── yolo11n.pt            # YOLOv11n detector weights 
```

---

## Setup

```bash
pip install torch torchvision ultralytics pillow tqdm opencv-python
```

Place the following weight files in `weights/`:
- `yolo11n.pt` — YOLOv11n detector (download from Ultralytics or reuse from training)
- `torca_best.pt` — trained TORCA checkpoint

---

## Usage

### Step 1 — Preprocess the dataset

Converts raw COCO-Tasks annotations into letterboxed 640×640 images with a `samples.json` index. Only needed once before training.

```bash
python preprocess_dataset.py \
    --data-root /path/to/Data \
    --out-root  /path/to/Data_Preprocessed
```

**Raw dataset layout expected:**
```
Data/
  train/
    images/
    annotations/   # task_1_train.json … task_14_train.json
  test/
    images/
    annotations/
```

Train augmentation (colour jitter) is applied automatically.

---

### Step 2 — Train

```bash
python train.py --data-root /path/to/Data_Preprocessed
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--data-root` | `./data` | Preprocessed dataset root |
| `--epochs` | `15` | Total training epochs |
| `--batch` | `8` | Batch size |
| `--lr` | `5e-5` | Scorer head learning rate |
| `--backbone-lr` | `5e-6` | FPN backbone learning rate |
| `--freeze-epochs` | `3` | Epochs to keep backbone frozen |
| `--chunk-frac` | `0.25` | Shard size for sub-epoch validation (1.0 = full-epoch) |

Checkpoints are saved to `weights/torca_best.pt` (lowest validation BCE) and `weights/torca_latest.pt` (most recent). Training history is saved to `weights/training_history.json`.

---

### Step 3 — Evaluate

```bash
python evaluate.py --data-root /path/to/Data_Preprocessed
```

Prints Top-1 accuracy and mAP@0.5 per task and overall. Results saved to `eval_results.json`.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--data-root` | `./data` | Preprocessed dataset root |
| `--weights` | `weights/torca_best.pt` | TORCA checkpoint |
| `--batch` | `8` | Batch size |

---

### Step 4 — Single-image inference

Runs TORCA on any image (no preprocessing required). Saves an annotated PNG with the top-ranked box in green and all other detections in grey.

```bash
python infer.py path/to/image.jpg --task 7
python infer.py path/to/image.jpg --task 2 --out result.png
```

**Task IDs:**

| ID | Task | ID | Task |
|---|---|---|---|
| 1 | Step on something | 8 | Open bottle of beer |
| 2 | Sit comfortably | 9 | Open parcel |
| 3 | Place flowers | 10 | Serve wine |
| 4 | Get potatoes out of fire | 11 | Pour sugar |
| 5 | Water plant | 12 | Smear butter |
| 6 | Get lemon out of tea | 13 | Extinguish fire |
| 7 | Dig hole | 14 | Pound carpet |

Key options:

| Flag | Default | Description |
|---|---|---|
| `--task` | `7` | Task ID (1–14) |
| `--weights` | `weights/torca_best.pt` | TORCA checkpoint |
| `--yolo-weights` | `weights/yolo11n.pt` | YOLOv11n weights |
| `--out` | `<image>_torca_task<N>.png` | Output path |

---

### Hardware analysis

Estimates parameter count, weight memory, activation memory, MAC count, and BRAM utilisation on the Genesys-2 (Kintex-7, 2 MB BRAM) target.

```bash
python hardware_analysis.py
python hardware_analysis.py --n-detections 15
```

Prints a full breakdown and saves `hardware_analysis.json`.

**Key figures:** 2.56 M params · 2.4 MB INT8 · ~3.4 GMACs/image · 1.91 MB estimated BRAM (95.3% of 2 MB budget).

---

## Architecture Summary

```
Image → YOLOv11n (layers 0-8) → P3 / P4 / P5 FPN maps
                                        │
                         Multi-scale RoI-Align (7×7 grid)
                         + Class one-hot projection
                         + Fuse MLP → 256-D object tokens
                                        │
                         2× Self-attention (contextualise objects)
                                        │
                         Task embedding cross-attends → S_rel per box
                                        │
               S_final = S_rel × S_yolo_conf  →  argmax = answer
```
