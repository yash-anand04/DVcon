import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_YOLO_PATH = os.environ.get("SATAY_OPEN_VOCAB_YOLO_PATH", "yolov8s-worldv2.pt")

TASK_NAMES = {
    1: "Step on something",
    2: "Sit comfortably",
    3: "Place flowers",
    4: "Get potatoes out of fire",
    5: "Water plant",
    6: "Get lemon out of tea",
    7: "Dig hole",
    8: "Open bottle of beer",
    9: "Open parcel",
    10: "Serve wine",
    11: "Pour sugar",
    12: "Smear butter",
    13: "Extinguish fire",
    14: "Pound carpet",
}

OPEN_VOCAB_CLASSES = [
    "person", "chair", "sofa", "bench", "stool", "table", "bottle", "beer bottle",
    "wine glass", "glass", "cup", "mug", "vase", "flower", "potted plant", "plant",
    "pot", "lemon", "tea cup", "spoon", "knife", "fork", "scissors", "shovel",
    "trowel", "tool", "parcel", "package", "box", "envelope", "paper", "sugar",
    "sugar bowl", "butter", "bread", "toast", "carpet", "rug", "fire",
    "fire extinguisher", "bucket", "water bottle", "watering can", "potato",
    "oven mitt", "glove", "shoe", "footstool", "ground", "dirt", "soil",
    "plate", "bowl", "tray", "napkin", "corkscrew", "opener",
]


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))


class TaskAwareRelevanceHead(nn.Module):
    """
    V1C relevance scorer.

    The detector remains frozen and unchanged. This head receives top-k object
    tokens plus the active task id, then predicts one relevance logit per object.
    """
    def __init__(
        self,
        num_classes,
        num_tasks,
        feat_dim=256,
        embed_dim=256,
        top_k=20,
        max_classes=512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.top_k = top_k
        self.max_classes = max_classes

        self.visual_proj = nn.Linear(feat_dim, embed_dim)
        self.class_embed = nn.Embedding(max_classes, embed_dim)
        self.task_embed = nn.Embedding(num_tasks + 1, embed_dim)
        self.score_proj = nn.Linear(2, embed_dim)
        self.box_proj = nn.Linear(4, embed_dim)
        self.combine = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )
        self.attn = SelfAttentionBlock(embed_dim)
        self.relevance_head = nn.Linear(embed_dim, 1)

    def forward(self, visual_feats, class_ids, det_scores, text_scores, boxes_xyxy, image_sizes, task_id, mask=None):
        k = min(self.top_k, det_scores.size(1))
        _, topk_idx = torch.topk(det_scores, k=k, dim=1)

        def gather_last(x):
            return torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        visual_feats = gather_last(visual_feats)
        boxes_xyxy = gather_last(boxes_xyxy)
        det_scores = torch.gather(det_scores, 1, topk_idx)
        text_scores = torch.gather(text_scores, 1, topk_idx)
        class_ids = torch.gather(class_ids, 1, topk_idx).clamp(min=0, max=self.max_classes - 1)
        if mask is not None:
            mask = torch.gather(mask, 1, topk_idx)

        box_feats = normalize_boxes_xyxy(boxes_xyxy, image_sizes)
        visual_token = F.relu(self.visual_proj(visual_feats))
        class_token = self.class_embed(class_ids)
        task_token = self.task_embed(task_id).unsqueeze(1).expand(-1, k, -1)
        score_token = F.relu(self.score_proj(torch.stack([det_scores, text_scores], dim=-1)))
        box_token = F.relu(self.box_proj(box_feats))

        tokens = self.combine(visual_token + class_token + task_token + score_token + box_token)
        tokens = tokens * det_scores.unsqueeze(-1)
        tokens = self.attn(tokens, mask)
        logits = self.relevance_head(tokens).squeeze(-1)
        return logits


class YOLOBackboneFeatureTap(nn.Module):
    HOOK_LAYERS = {4: "p3", 6: "p4", 8: "p5"}

    def __init__(self, yolo):
        super().__init__()
        full_model = yolo.model.model
        self.backbone = nn.Sequential(*list(full_model.children())[:9])
        self._feats = {}
        self._register_hooks()

    def _register_hooks(self):
        layers = list(self.backbone.children())
        for idx, name in self.HOOK_LAYERS.items():
            if idx < len(layers):
                layers[idx].register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._feats[name] = out
        return hook

    def forward(self, x):
        self._feats = {}
        _ = self.backbone(x)
        missing = [name for name in ("p3", "p4", "p5") if name not in self._feats]
        if missing:
            raise RuntimeError(f"Could not tap YOLO feature maps: missing {missing}")
        return self._feats["p3"], self._feats["p4"], self._feats["p5"]


class OpenVocabDetectionFeatureExtractor(nn.Module):
    """
    YOLO-native candidate generator for V1C.

    The regular YOLO heads produce boxes/classes/confidences. A parallel feature
    tap reads YOLO backbone/neck maps, and the suitability head samples those
    maps at each candidate box center. No extra CNN crop encoder is used.
    """
    def __init__(
        self,
        yolo_model=DEFAULT_YOLO_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        vocab=None,
    ):
        super().__init__()
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics YOLO is not installed") from exc

        self.device = device
        self.vocab = list(vocab or OPEN_VOCAB_CLASSES)
        # Store YOLO outside PyTorch's module registry — Ultralytics YOLO
        # overrides .train() to start a training run, so letting PyTorch's
        # Module.eval() traverse into it triggers an unwanted training launch.
        object.__setattr__(self, 'yolo', YOLO(yolo_model))
        self._configure_open_vocab_classes()
        self.feature_tap = YOLOBackboneFeatureTap(self.yolo).to(device)
        self.feature_tap.eval()

    def _configure_open_vocab_classes(self):
        if hasattr(self.yolo, "set_classes"):
            try:
                self.yolo.set_classes(self.vocab)
            except Exception as exc:
                print(f"[V1C] WARNING: could not set open-vocab classes: {exc}")

    @torch.no_grad()
    def forward(self, images):
        visual_feats, class_ids, det_scores, text_scores, mask, boxes, image_sizes = self.extract_with_boxes(images)
        return visual_feats, class_ids, det_scores, text_scores, mask, boxes, image_sizes

    @torch.no_grad()
    def extract_with_boxes(self, images):
        tensors = self._images_to_tensor(images)
        p3, p4, p5 = self.feature_tap(tensors)
        feature_maps = {"p3": p3, "p4": p4, "p5": p5}

        batch_size = len(images)
        feat_list, class_list, score_list, text_score_list, box_list = [], [], [], [], []
        image_sizes = []

        for batch_idx, img in enumerate(images):
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_sizes.append((img.width, img.height))

            results = self.yolo(img, verbose=False)[0]
            boxes = results.boxes.xyxy.to(self.device)
            scores = results.boxes.conf.to(self.device)
            classes = results.boxes.cls.long().to(self.device)
            feats = self._sample_box_features(feature_maps, boxes, batch_idx)

            feat_list.append(feats)
            class_list.append(classes)
            score_list.append(scores)
            text_score_list.append(scores)
            box_list.append(boxes)

        max_n = max([x.shape[0] for x in box_list] + [1])
        feat_batch = torch.zeros(batch_size, max_n, 256, device=self.device)
        class_batch = torch.zeros(batch_size, max_n, dtype=torch.long, device=self.device)
        score_batch = torch.zeros(batch_size, max_n, device=self.device)
        text_score_batch = torch.zeros(batch_size, max_n, device=self.device)
        box_batch = torch.zeros(batch_size, max_n, 4, device=self.device)
        mask = torch.ones(batch_size, max_n, dtype=torch.bool, device=self.device)

        for i in range(batch_size):
            n = box_list[i].shape[0]
            if n == 0:
                mask[i, 0] = False
                continue
            feat_batch[i, :n] = feat_list[i]
            class_batch[i, :n] = class_list[i]
            score_batch[i, :n] = score_list[i]
            text_score_batch[i, :n] = text_score_list[i]
            box_batch[i, :n] = box_list[i]
            mask[i, :n] = False

        return feat_batch, class_batch, score_batch, text_score_batch, mask, box_batch, image_sizes

    def _images_to_tensor(self, images):
        import numpy as np

        arrays = []
        for img in images:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            arrays.append(torch.from_numpy(arr).permute(2, 0, 1))
        return torch.stack(arrays).to(self.device)

    def _sample_box_features(self, feature_maps, boxes, batch_idx):
        if boxes.numel() == 0:
            return torch.zeros(0, 256, device=self.device)

        feats = []
        for box in boxes:
            fmap_name, stride = self._select_feature_map(box)
            fmap = feature_maps[fmap_name][batch_idx]
            feat = self._local_average_at_box_center(fmap, box, stride)
            feats.append(self._pad_or_trim(feat))
        return torch.stack(feats, dim=0)

    def _select_feature_map(self, box):
        x1, y1, x2, y2 = box
        max_side = torch.max(x2 - x1, y2 - y1).item()
        if max_side < 96:
            return "p3", 8
        if max_side < 224:
            return "p4", 16
        return "p5", 32

    def _local_average_at_box_center(self, fmap, box, stride, radius=1):
        _, h, w = fmap.shape
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) * 0.5 / stride).round().long().item()
        cy = ((y1 + y2) * 0.5 / stride).round().long().item()
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        x_start = max(0, cx - radius)
        x_end = min(w, cx + radius + 1)
        y_start = max(0, cy - radius)
        y_end = min(h, cy + radius + 1)
        return fmap[:, y_start:y_end, x_start:x_end].mean(dim=(1, 2))

    def _pad_or_trim(self, feat):
        if feat.numel() == 256:
            return feat
        if feat.numel() > 256:
            return feat[:256]
        return torch.cat([feat, feat.new_zeros(256 - feat.numel())], dim=0)


def normalize_boxes_xyxy(boxes, image_sizes):
    out = boxes.clone()
    for i, (width, height) in enumerate(image_sizes):
        scale = boxes.new_tensor([width, height, width, height]).clamp(min=1.0)
        out[i] = out[i] / scale
    return out.clamp(0.0, 1.0)


def gather_topk_like_model(tensor, det_scores, top_k):
    k = min(top_k, det_scores.size(1))
    _, topk_idx = torch.topk(det_scores, k=k, dim=1)
    if tensor.dim() == 3:
        return torch.gather(tensor, 1, topk_idx.unsqueeze(-1).expand(-1, -1, tensor.size(-1)))
    return torch.gather(tensor, 1, topk_idx)
