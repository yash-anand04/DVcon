# EXAMPLE :

# Pipeline: YOLO + Similarity

## 📌 Description

This pipeline performs:

1. Object detection using YOLOv5n
2. Feature extraction from detected objects
3. Query-based reasoning using cosine similarity

---

## ⚙️ Components

* `pipeline.py` → Main execution
* `similarity.py` → Similarity computation
* `results/` → Output images + logs

---

## 🧠 Reasoning Approach

* Convert query into embedding
* Match with detected object features
* Select highest similarity score

---

## 📊 Results

| Metric             | Value |
| ------------------ | ----- |
| Detection Accuracy | -     |
| Query Accuracy     | -     |
| Avg Latency (ms)   | -     |

---

## ▶️ Run This Pipeline

```bash
python pipeline.py
```

---

## 📌 Observations

* Fast on CPU
* Struggles with complex relational queries

---

## 🚀 Future Improvements

* Add spatial reasoning
* Improve embedding quality
