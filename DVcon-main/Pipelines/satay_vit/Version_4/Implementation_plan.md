# SATAY-ViT Version 4 — MLCoT Implementation Plan

## Context: What V4 Upgrades from V2

Version 4 is built on the **Version 2 architecture** (16×16 heatmap grid, 256 spatial tokens, standard `TransformerEncoder`) — *not* Version 3. The single, targeted upgrade is the **task embedding source**: replacing the plain learned `nn.Embedding` (a lookup table initialized from scratch) with offline, LLM-generated **knowledge vectors** that encode functional, affordance-level attributes for each of the 14 COCO-Tasks.

### What is KEPT from V2 (unchanged)
| Component | V2 Spec |
|-----------|---------|
| YOLO Backbone | YOLOv11n, layers 0–8, hooks at layers 4/6/8 |
| FPN Aggregator | 16×16 grid, `embed_dim=256` |
| ME-ViT Reasoner | Standard `TransformerEncoder` (2 layers, 4 heads), 256 spatial tokens |
| Score Fusion | $S_{final,i} = C_i \times R_{box,i}$ on a 16×16 coordinate map |
| Training Setup | BCE loss, AdamW, cosine LR schedule, 3-epoch backbone freeze warm-up |

### What CHANGES in V4
| Component | V2 | V4 |
|-----------|----|----|
| Task Embedding | `nn.Embedding(15, 256)` — learned from scratch | Pre-computed, fixed 256-dim vector from a CLIP/sentence-transformer text encoder, encoding LLM-derived visual attributes |
| Embedding source | Random init, optimized during training | Offline: generated once, stored as `task_knowledge_vectors.pt` (14 × 256 tensor) |

> **Why V2 and not V3?** V3's 32×32 grid (1,024 tokens) is being tested as the main architectural upgrade. V4's MLCoT is a *semantic* upgrade and is kept at 16×16 to isolate the effect of the knowledge embeddings. A 32×32 MLCoT variant will follow in V5.

---

## Phase 0: Offline Knowledge Vector Generation (`generate_knowledge_embeddings.py`)

This is the prerequisite script that must be run **once** before training to produce the 14 knowledge vectors.

1. **LLM Prompting (3-Level MLCoT Chain)**:
   * **Level 1 — Object Brainstorm**: For each of the 14 COCO-Tasks, prompt an LLM to list all objects that afford the task (beyond the primary COCO categories). For "Task 9: Open parcel," this yields: *knife, scissors, pen, screwdriver, letter opener*.
   * **Level 2 — Affordance Rationales**: Generate reasoning for *why* each object qualifies. E.g., *"a knife can cut through plastic due to its sharp blade edge"*.
   * **Level 3 — Visual Attribute Distillation**: Compress rationales into concise, visually-grounded descriptors. E.g., *"sharp blade, handle, pointed tip"*.

2. **Text Encoding**: Pass each task's Level 3 attribute string through a frozen CLIP text encoder (or `sentence-transformers/all-MiniLM-L6-v2` projected to 256D) to produce a `(14, 256)` tensor.

3. **Save**: Serialize the tensor to `Version_4/weights_e2e/task_knowledge_vectors.pt`.

---

## Phase 1: Model Architecture Changes (`model_e2e.py`)

The `MEViTHead` class is modified to load the pre-computed knowledge vectors instead of using a trainable embedding layer.

* **Remove**: `self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim)`
* **Add**: Load `task_knowledge_vectors.pt` as a **frozen** `nn.Embedding.from_pretrained(knowledge_tensor, freeze=True)`
* The cross-attention query/key mechanism (`q_proj`, `k_proj`) and heatmap output path remain **identical to V2**.
* **Grid**: 16×16, **Tokens**: 256, **embed_dim**: 256 — all unchanged.

The resulting heatmap shape remains `(B, 16, 16)`.

---

## Phase 2: Training Changes (`train_e2e.py`)

Minimal changes are needed to the training script:

* Pass `knowledge_path="weights_e2e/task_knowledge_vectors.pt"` as a config argument when constructing `SATAYViT_E2E`.
* Since the knowledge embeddings are frozen, the optimizer only needs to cover backbone + FPN aggregator + ViT head projections — the `task_embedding` parameters are excluded. This slightly reduces the trainable parameter count.
* All other hyperparameters (LR schedule, BCE loss, freeze epochs) remain the same as V2.

---

## Phase 3: VEGA Core Integration (Score Fusion)

The Task Fusion logic on the VEGA AS1061 RISC-V core is **unchanged in structure** but gains semantic interpretability:

* **Coordinate Mapping**: Bounding box $B_i$ coordinates are mapped onto the 16×16 heatmap grid (same as V2).
* **Relevance Integration**:
  $$R_{box, i} = \frac{1}{N_{patches}} \sum_{(x,y) \in B_i} R_{x,y}$$
* **Final Scoring**:
  $$S_{final, i} = C_i \times R_{box, i}$$

The improvement over V2 is that $R_{box,i}$ is now computed from a heatmap driven by **part-level affordance attributes** (e.g., "stem," "rim" for "Serve wine") rather than a generic task-index embedding. The VEGA core logic itself is unchanged.

**Example**: For "Serve wine," the knowledge vector encodes *stem, round bowl, transparent glass* → the heatmap will activate on wine glass parts ($R_{wine}=0.98$) far more distinctly than on a mug ($R_{cup}=0.40$), even when both have similar YOLO confidence ($C \approx 0.95$–$0.98$).

---

## Expected Outcome

By replacing the semantically-empty learned embedding with LLM-distilled attribute vectors, the cross-attention head has a **grounded query signal** from epoch 1. This should:
- Reduce the overfitting onset (validation loss divergence was observed from Epoch 4 in V3's 32×32 setup).
- Push Top-1 Fusion Accuracy beyond the 51.5% ceiling observed in V3, without the added complexity of sparse attention or a larger grid.
- Provide a clean ablation baseline: **MLCoT @ 16×16** (V4) vs. **MLCoT @ 32×32 + Sparse Attention** (V5).