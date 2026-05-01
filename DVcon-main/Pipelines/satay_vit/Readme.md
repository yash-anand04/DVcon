# SATAY-ViT: Semantic-Attention Task-Aware YOLO

## 📌 Description

**SATAY-ViT** (Semantic-Attention Task-Aware YOLO) is a specialized vision pipeline designed for **Task-Aware Object Detection** on edge devices. While traditional object detectors (like YOLO) identify all instances of a class, SATAY-ViT reasons about the specific **context** of a user's task to select the most relevant object.

For example, given the task *"Open bottle of beer,"* SATAY-ViT selects the specific bottle intended for the action rather than simply detecting all visible bottles in the scene.

---

## ⚙️ Unified Pipeline Architecture

All versions of SATAY-ViT operate as a **parallel dual-branch architecture**, ensuring that task-specific reasoning does not bottleneck the initial object localization.

1. **Localization Branch**: A standard **YOLOv11** model runs in parallel on the input image to generate a set of candidate bounding boxes and objectness scores.
2. **Reasoning Branch**: The SATAY-ViT Reasoning Head (Versions 1-5) processes the image and a **Task ID** to generate a 2D **Task Relevance Heatmap**.
3. **Spatial Fusion**: The center of each YOLO candidate box is mapped to the corresponding grid cell in the heatmap.
4. **Score Weighting**: The final confidence score $S_{final}$ is calculated as:
    $$S_{final} = S_{yolo} \times S_{relevance}$$

---

## 📊 Comparative Model Analysis

The following table summarizes the evolution of the SATAY-ViT architecture through five major versions:

| Version | Backbone | Grid Size | Task Embedding | Reasoning Head | Key Innovation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | Linear Projection | 16x16 | Learned (Lookup) | Standard ViT | Initial prototype with fixed patches |
| **V1B** | YOLO + ResNet50 Crop Features | Top-K Objects | Class One-Hot + Visual Token | Object Self-Attention | Alternate notebook architecture for detection-level task scoring |
| **V2** | YOLO11 (Layers 0-8) | 16x16 | Learned (Lookup) | ME-ViT Encoder | E2E feature reuse via FPN Aggregator |
| **V3** | YOLO11 (Layers 0-8) | **32x32** | Learned (Lookup) | **Sparse Top-K** | High-res reasoning with sparse attention |
| **V4** | YOLO11 (Layers 0-8) | 16x16 | **Frozen MLCoT** | ME-ViT Encoder | LLM-distilled visual attribute grounding |
| **V5** | YOLO11 (Layers 0-8) | 16x16 | **CLIP-Seeded** | ME-ViT Encoder | Semantic init + Relevance Floor (0.3) |

---

## 🧠 Version-Specific Architecture & Data Flow

### Version 1: The ViT Prototype

* **Reasoning Flow**: Image $\rightarrow$ 40x40 Linear Patch Projection $\rightarrow$ 16x16 Grid $\rightarrow$ 2-Layer ViT Encoder $\rightarrow$ Dot-product w/ Task Embedding $\rightarrow$ Heatmap.
* **Transition**: Effectively treats the image as a set of flat patches. No deep convolutional features are used.
* **Observation**: Strong baseline, but limited spatial context due to the lack of a deep feature backbone.

### Version 1B: Object-Token Attention Prototype

* **Reasoning Flow**: Image $\rightarrow$ YOLO detections $\rightarrow$ ResNet50 crop features + class one-hot tokens $\rightarrow$ top-k object self-attention $\rightarrow$ per-object task probabilities.
* **Transition**: Uses the alternate notebook architecture directly, shifting from dense heatmap prediction to detection-level task scoring.
* **Training Adapter**: Top-k detections are supervised by matching them against preferred COCO-Tasks boxes for the active task.

### Version 2: End-to-End Feature Reuse

* **Reasoning Flow**: Image $\rightarrow$ **YOLO11 Backbone** $\rightarrow$ **FPN Aggregator** $\rightarrow$ 16x16 Grid $\rightarrow$ ViT Encoder $\rightarrow$ Heatmap.
* **Transition**: Uses standard detection layers (P3, P4, P5) as inputs to the reasoner. The backbone learns to extract features that are simultaneously good for detection and reasoning.
* **Observation**: Significant improvement in heatmap accuracy by leveraging pre-trained detection weights.

### Version 3: High-Resolution Sparse Attention

* **Reasoning Flow**: Image $\rightarrow$ YOLO11 Backbone $\rightarrow$ 32x32 Aggregator $\rightarrow$ **Top-K Sparse Transformer** $\rightarrow$ 32x32 Heatmap.
* **Transition**: Increases the reasoning grid to 32x32 (1024 tokens). To maintain edge-level speed, it uses sparse attention to process only the top 256 most salient spatial interactions.
* **Observation**: Best for small objects and fine-grained spatial reasoning where 16x16 is too coarse.

### Version 4: Knowledge-Graph Grounding

* **Reasoning Flow**: Same architectural flow as V2 + **Frozen MLCoT Embeddings**.
* **Transition**: Task vectors are pre-computed using LLM-distilled visual attributes (e.g., "sitting" $\rightarrow$ "chair, leg, backrest"). These are fixed during training.
* **Observation**: Proved that rigid knowledge constraints can sometimes hurt performance if the linguistic priors don't perfectly match the visual dataset bias.

### Version 5: Semantic Initialization (Optimal)

* **Reasoning Flow**: Same architectural flow as V2 + **CLIP-Seeded Trainable Embeddings**.
* **Transition**: Task vectors are initialized from OpenAI **CLIP** text embeddings (providing a semantic starting point) but remain fully trainable to adapt to the specific COCO-Tasks domain.
* **Inference Innovation**: Implementation of a **Relevance Floor (0.3)** in the fusion logic:
    $$S_{final} = S_{yolo} \times \max(S_{relevance}, 0.3)$$
* **Observation**: Most robust version; semantic initialization speeds up convergence and improves per-task discriminability.

---

## 📂 Components

* `utils/data_loader.py` → Dynamic COCO-Tasks dataset loader (supports 16x16 and 32x32).
* `utils/evaluate.py` → Comparative accuracy benchmarking against YOLO baseline.
* `utils/inference.py` → Scoring fusion and visualization.
* `Version_X/model_e2e.py` → Specific architecture implementation.
* `Version_X/train_e2e.py` → Training scripts with multi-phase (frozen/unfrozen) logic.

---

## 📊 Results

| Metric | Value |
| :--- | :--- |
| Overall Top-1 Accuracy | - |
| Baseline YOLO Acc | - |
| mAP @ 0.5 | - |
| Avg Latency (ms) | - |

---

## ⚡ Hardware Estimation

(Placeholder)

---

## ▶️ Running This Pipeline

To train the most recent version (V5):

```bash
python Version_5/train_e2e.py --epochs 15 --batch 16
```

To evaluate performance:

```bash
python utils/evaluate.py --weights Version_5/weights_e2e/satay_vit_e2e_best.pt
```

---

## 🚀 Future Improvements

* Integration of Temporal Context (ME-ViT-T).
* Quantization-Aware Training (QAT) for FPGA deployment.
* Dynamic grid resolution based on task complexity.
