"""
infer.py — TORCA Single-Image Inference
=========================================
Runs TORCA on one image for a given task and saves an annotated overlay PNG.
Works on any image — no preprocessed dataset required.

Usage:
    python infer.py path/to/image.jpg --task 7
    python infer.py path/to/image.jpg --task 2 --out result.png
    python infer.py path/to/image.jpg --task 7 --weights weights/torca_best.pt

Tasks:
     1  Step on something      2  Sit comfortably       3  Place flowers
     4  Get potatoes from fire 5  Water plant           6  Get lemon from tea
     7  Dig hole               8  Open bottle of beer   9  Open parcel
    10  Serve wine            11  Pour sugar            12  Smear butter
    13  Extinguish fire       14  Pound carpet
"""

import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)

from model import DEFAULT_YOLO_PATH, TORCA, YOLODetector

TASKS = {
    1:  "Step on something",   2:  "Sit comfortably",
    3:  "Place flowers",       4:  "Get potatoes out of fire",
    5:  "Water plant",         6:  "Get lemon out of tea",
    7:  "Dig hole",            8:  "Open bottle of beer",
    9:  "Open parcel",         10: "Serve wine",
    11: "Pour sugar",          12: "Smear butter",
    13: "Extinguish fire",     14: "Pound carpet",
}

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


def run(image_path: str, task_id: int, weights: str, yolo_weights: str, out_path: str):
    if task_id not in TASKS:
        raise ValueError(f"task_id must be 1–14, got {task_id}")

    device   = "cpu"
    # Load original image — passed to YOLO as-is so it can letterbox correctly
    pil_orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_orig.size
    # 640×640 version used as TORCA input tensor and for drawing
    pil_640  = pil_orig.resize((640, 640))

    model    = TORCA(checkpoint=yolo_weights).to(device)
    detector = YOLODetector(yolo_or_checkpoint=model._shared_yolo, device=device)

    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Model weights not found: {weights}\n"
            f"Train first (train.py) or supply --weights."
        )
    ckpt  = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    import torchvision.transforms as T
    with torch.no_grad():
        # YOLO detects on the original image (returns boxes in original coords)
        det_raw = detector.detect_batch([pil_orig])

        # Scale boxes from original image space → 640×640 space for RoI-Align
        raw_boxes, raw_scores, raw_classes = det_raw[0]
        if raw_boxes.shape[0] > 0:
            scaled = raw_boxes.clone().float()
            scaled[:, 0] *= 640 / orig_w   # x1
            scaled[:, 1] *= 640 / orig_h   # y1
            scaled[:, 2] *= 640 / orig_w   # x2
            scaled[:, 3] *= 640 / orig_h   # y2
            scaled = scaled.clamp(0, 640)
        else:
            scaled = raw_boxes
        det = [(scaled, raw_scores, raw_classes)]

        img_t   = T.ToTensor()(pil_640).unsqueeze(0)
        task_t  = torch.tensor([task_id])
        rel, conf, mask = model(img_t, det, task_t)

        boxes   = det[0][0]   # [N, 4] xyxy in 640×640 space
        classes = det[0][2]   # [N] class ids
        n       = boxes.shape[0]

        if n == 0:
            print("No objects detected in this image.")
            return

        scores  = (rel[0, :n] * conf[0, :n]).cpu()
        best    = int(scores.argmax())
        s_final = float(scores[best])
        bx1, by1, bx2, by2 = boxes[best].tolist()
        cls_name = COCO_NAMES[int(classes[best])] if int(classes[best]) < len(COCO_NAMES) else "object"

    # ── Draw overlay ──────────────────────────────────────────────────
    draw = ImageDraw.Draw(pil_640, "RGBA")
    try:
        font_sm = ImageFont.truetype("arial.ttf", 14)
        font_lg = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font_sm = ImageFont.load_default()
        font_lg = font_sm

    # Grey boxes for lower-ranked detections
    for i in range(n):
        if i == best:
            continue
        x1, y1, x2, y2 = boxes[i].tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(160, 160, 160, 200), width=2)
        cn = COCO_NAMES[int(classes[i])] if int(classes[i]) < len(COCO_NAMES) else ""
        draw.text((x1 + 3, y1 + 2), cn, fill=(200, 200, 200, 230), font=font_sm)

    # Green box for best prediction
    draw.rectangle([bx1, by1, bx2, by2], outline=(46, 204, 113, 255), width=4)
    label   = f"{cls_name}  S={s_final:.2f}"
    tw      = len(label) * 9
    draw.rectangle([bx1, by1 - 22, bx1 + tw, by1], fill=(46, 204, 113, 200))
    draw.text((bx1 + 3, by1 - 20), label, fill=(0, 0, 0, 255), font=font_lg)

    # Task banner
    task_str = f'Task: "{TASKS[task_id]}"'
    draw.rectangle([0, 610, 640, 640], fill=(0, 0, 0, 180))
    draw.text((8, 614), task_str, fill=(255, 255, 255, 255), font=font_lg)

    pil_640.convert("RGB").save(out_path)
    print(f"Saved          : {out_path}")
    print(f"Task           : {TASKS[task_id]}")
    print(f"Best detection : {cls_name}  (S_final={s_final:.3f})")
    print(f"Bounding box   : [{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}]")
    print(f"Detections     : {n} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TORCA inference on a single image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("image",           help="Path to input image (any common format).")
    parser.add_argument("--task",          type=int, default=7,
                        help="Task ID (1–14). See module docstring for the full list.")
    parser.add_argument("--weights",       default=os.path.join(THIS, "weights", "torca_best.pt"),
                        help="Path to trained TORCA checkpoint.")
    parser.add_argument("--yolo-weights",  default=DEFAULT_YOLO_PATH,
                        help="Path to yolo11n.pt.")
    parser.add_argument("--out",           default=None,
                        help="Output PNG path. Defaults to <image_stem>_torca_task<N>.png.")
    args = parser.parse_args()

    out_path = args.out or (
        os.path.splitext(args.image)[0] + f"_torca_task{args.task}.png"
    )
    run(args.image, args.task, args.weights, args.yolo_weights, out_path)
