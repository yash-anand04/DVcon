"""
single_image_demo.py — runs TORCA on one image and saves an annotated overlay.

Usage:
    python single_image_demo.py path/to/image.jpg --task 2
"""
import argparse, os, sys, torch
from PIL import Image, ImageDraw, ImageFont

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from model import SATAYViT_V6, YOLODetector, DEFAULT_YOLO_PATH

TASKS = {
    1: "Step on something",  2: "Sit comfortably",   3: "Place flowers",
    4: "Get potatoes out of fire", 5: "Water plant",  6: "Get lemon out of tea",
    7: "Dig hole",           8: "Open bottle of beer", 9: "Open parcel",
    10: "Serve wine",       11: "Pour sugar",        12: "Smear butter",
    13: "Extinguish fire",  14: "Pound carpet",
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

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--task",    type=int, default=2)
ap.add_argument("--weights", default=os.path.join(THIS, "weights", "v6_best.pt"))
ap.add_argument("--out",     default=None)
args = ap.parse_args()

device   = "cpu"
pil      = Image.open(args.image).convert("RGB").resize((640, 640))
detector = YOLODetector(checkpoint=DEFAULT_YOLO_PATH, device=device)
model    = SATAYViT_V6(checkpoint=DEFAULT_YOLO_PATH).to(device)
ckpt     = torch.load(args.weights, map_location=device)
state    = ckpt.get("state_dict", ckpt)
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    det = detector.detect_batch([pil])
    import torchvision.transforms as T
    img_t  = T.ToTensor()(pil).unsqueeze(0)
    task_t = torch.tensor([args.task])
    rel, conf, mask = model(img_t, det, task_t)
    boxes   = det[0][0]          # [N, 4] xyxy
    classes = det[0][2]          # [N] class ids
    n       = boxes.shape[0]
    scores  = (rel[0, :n] * conf[0, :n]).cpu()
    best    = int(scores.argmax())
    s_final = float(scores[best])

# ── draw ──────────────────────────────────────────────────────────────
draw = ImageDraw.Draw(pil, "RGBA")

try:
    font_sm = ImageFont.truetype("arial.ttf", 14)
    font_lg = ImageFont.truetype("arial.ttf", 17)
except OSError:
    font_sm = ImageFont.load_default()
    font_lg = font_sm

# grey semi-transparent boxes for all non-best detections
for i in range(n):
    if i == best:
        continue
    x1, y1, x2, y2 = boxes[i].tolist()
    draw.rectangle([x1, y1, x2, y2], outline=(160, 160, 160, 220), width=2)
    cls_name = COCO_NAMES[int(classes[i])] if int(classes[i]) < len(COCO_NAMES) else str(int(classes[i]))
    draw.text((x1 + 3, y1 + 2), cls_name, fill=(200, 200, 200, 230), font=font_sm)

# green box for the best detection
bx1, by1, bx2, by2 = boxes[best].tolist()
draw.rectangle([bx1, by1, bx2, by2], outline=(46, 204, 113, 255), width=4)
cls_name = COCO_NAMES[int(classes[best])] if int(classes[best]) < len(COCO_NAMES) else str(int(classes[best]))
label = f"{cls_name}  S={s_final:.2f}"
# label background
tw = len(label) * 9
draw.rectangle([bx1, by1 - 22, bx1 + tw, by1], fill=(46, 204, 113, 200))
draw.text((bx1 + 3, by1 - 20), label, fill=(0, 0, 0, 255), font=font_lg)

# task banner at bottom
task_str = f'Task: "{TASKS[args.task]}"'
draw.rectangle([0, 610, 640, 640], fill=(0, 0, 0, 180))
draw.text((8, 614), task_str, fill=(255, 255, 255, 255), font=font_lg)

out_path = args.out or args.image.rsplit(".", 1)[0] + f"_torca_task{args.task}.png"
pil.convert("RGB").save(out_path)
print(f"Saved: {out_path}")
print(f"Best detection: {cls_name}  S_final={s_final:.3f}  box=[{bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f}]")
print(f"Total detections drawn: {n}")
