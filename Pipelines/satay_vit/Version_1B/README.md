# SATAY-ViT Version 1B

V1B integrates the alternate notebook architecture:

- YOLO detects candidate objects.
- ResNet50 extracts visual features from each detected crop.
- Class one-hot features and crop features are projected into a shared embedding.
- Detection-score-weighted object tokens pass through self-attention.
- Context and visual heads predict per-object probabilities for the 14 COCO-Tasks.

The model architecture in `model.py` follows the notebook. The surrounding
training and evaluation code adapts it to this repository's preprocessed
COCO-Tasks format.

## Train

```bash
python train.py --data-root e:/DVcon/DVcon/Data_Preprocessed --epochs 10 --batch 4
```

The trainer freezes YOLO and ResNet feature extraction, then trains only the
task-driven attention model. For each sample, top-k YOLO detections are labeled
positive when they overlap a preferred ground-truth box for that task.

## Evaluate

```bash
python evaluate.py --data-root e:/DVcon/DVcon/Data_Preprocessed --weights-path weights/v1b_attention_best.pt
```

Evaluation writes:

- `eval_results.json`
- `per_task_accuracy.png`

Use `--score-mode fused` to rank detections by `task_probability * detection_confidence`
instead of task probability alone.
