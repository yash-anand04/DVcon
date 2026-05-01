# Vision-Based Reasoning Pipeline

## Objective

Develop a CPU-based pipeline that performs:

* Object Detection
* Query-based Reasoning
* Evaluation across 14 queries

---

## Pipelines Implemented

EXAMPLE:

| Pipeline Name     | Model Used   | Reasoning Type    | Accuracy | Latency (ms) | Notes               |
| ----------------- | ------------ | ----------------- | -------- | ------------ | ------------------- |
| YOLO Baseline     | YOLOv5n      | None              | -        | -            | Detection only      |
| YOLO + Similarity | YOLOv5n      | Cosine Similarity | -        | -            | Lightweight         |
| Mask R-CNN        | ResNet50-FPN | Rule-based        | -        | -            | Better segmentation |
| Transformer Lite  | ViT (Lite)   | Attention-based   | -        | -            | Expensive           |

Update this table after experiments.

---

### Report folder

```
report/
├── figures/          # Plots, pipeline diagrams, architecture visuals
├── screenshots/      # Output images from pipelines
├── tables/           # Metric tables (optional as images or CSV)
└── final_report.pdf  # Final compiled report
```

---

### Guidelines

* Save all **pipeline outputs** (annotated images, query results) in:

  ```
  pipelines/<pipeline_name>/results/
  ```

  Then copy the **best examples** into:

  ```
  report/screenshots/
  ```

* Save all **architecture diagrams** (pipeline flow, accelerator blocks) in:

  ```
  report/figures/
  ```

* Use clear naming:

  ```
  fig_pipeline_overview.png
  fig_yolo_results.png
  fig_reasoning_example.png
  ```

---


## Evaluation Metrics

* Detection Accuracy
* Query Accuracy (Correct Answer Rate)
* Latency (CPU execution time)
* Memory Usage

---

## How to Run


```bash
pip install -r requirements.txt

cd pipelines/yolo_baseline
python pipeline.py
```

---

## Final Selected Pipeline


(To be filled after evaluation, add to the Final pipeline folder)

---

