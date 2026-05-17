# SATAY-ViT Version 1C

V1C is the open-vocabulary detection-level relevance experiment.

It keeps the detector's normal box/class/objectness outputs unchanged, then
trains a separate task-aware relevance head over the top-k detections.

## Architecture

- YOLO-World/YOLOE-style detector proposes candidate boxes.
- If supported by the Ultralytics checkpoint, `set_classes()` installs a
  task/affordance vocabulary.
- A parallel suitability head samples YOLO feature maps at each candidate box
  center.
- Class id, detector confidence, open-vocab score, normalized box geometry, and
  YOLO-native visual features are fused into object tokens.
- Object self-attention predicts one relevance logit per candidate.

Inference ranks boxes by:

```text
final_score = detector_confidence * task_relevance
```

## Train

```bash
python train.py --data-root e:/DVcon/DVcon/Data_Preprocessed --epochs 10 --batch 4
```

The default detector is:

```text
yolov8s-worldv2.pt
```

Override it with:

```bash
python train.py --yolo-model path/to/model.pt
```

or:

```bash
set SATAY_OPEN_VOCAB_YOLO_PATH=path/to/model.pt
python train.py
```

## Evaluate

```bash
python evaluate.py --data-root e:/DVcon/DVcon/Data_Preprocessed --weights-path weights/v1c_relevance_best.pt
```

Use `--score-mode prob` to rank only by task relevance, or `--score-mode fused`
to rank by relevance times detector confidence.

## Notes

V1C is intentionally conservative: it does not fine-tune the detector at first.
The regular YOLO heads remain unchanged; only the extra suitability head is
trained.
