import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_PATH = os.environ.get("SATAY_YOLO_PATH", "yolo11n.pt")


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x: [B, N, D]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm(x + attn_out)
        return x


class TaskDrivenAttentionModel(nn.Module):
    def __init__(self, num_classes, num_tasks, feat_dim=2048, embed_dim=256, top_k=20):
        super().__init__()
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.top_k = top_k

        self.class_embed = nn.Linear(num_classes, embed_dim)
        self.visual_proj = nn.Linear(feat_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

        self.attn = SelfAttentionBlock(embed_dim)

        self.context_head = nn.Linear(embed_dim, num_tasks)
        self.visual_head = nn.Linear(feat_dim, num_tasks)

    def forward(self, class_onehot, visual_feats, det_scores, mask=None):
        B, N, _ = visual_feats.shape

        # Top-K by detection score
        k = min(self.top_k, N)
        _, topk_idx = torch.topk(det_scores, k=k, dim=1)

        def gather(x):
            return torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

        class_onehot = gather(class_onehot)
        visual_feats = gather(visual_feats)
        det_scores = torch.gather(det_scores, 1, topk_idx)

        if mask is not None:
            mask = torch.gather(mask, 1, topk_idx)

        # h0
        class_emb = F.relu(self.class_embed(class_onehot))
        visual_emb = F.relu(self.visual_proj(visual_feats))
        h0 = self.combine(class_emb * visual_emb)

        # weight by detection confidence (your idea B)
        h0 = h0 * det_scores.unsqueeze(-1)

        # attention
        h_context = self.attn(h0, mask)

        # heads
        p_context = torch.sigmoid(self.context_head(h_context))
        p_visual = torch.sigmoid(self.visual_head(visual_feats))

        p_final = (p_context + p_visual) / 2
        return p_final, p_context, p_visual


class DetectionFeatureExtractor(nn.Module):
    def __init__(self, yolo_model=DEFAULT_YOLO_PATH, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device

        # YOLO
        if YOLO_AVAILABLE:
            self.yolo = YOLO(yolo_model)
        else:
            self.yolo = None

        # ResNet (remove final FC)
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        self.resnet = nn.Sequential(*list(backbone.children())[:-1]).to(device)
        self.resnet.eval()

        # preprocessing
        self.transform = weights.transforms()

    @torch.no_grad()
    def forward(self, images):
        """
        images: list of PIL images
        Returns:
            class_onehot: [B, N, C]
            visual_feats: [B, N, 2048]
            det_scores:   [B, N]
        """
        class_batch, feat_batch, score_batch, mask, _ = self.extract_with_boxes(images)
        return class_batch, feat_batch, score_batch, mask

    @torch.no_grad()
    def extract_with_boxes(self, images):
        """
        Same feature path as the notebook forward pass, with padded YOLO boxes
        returned for training and evaluation target matching.
        """
        B = len(images)

        class_list = []
        feat_list = []
        score_list = []
        box_list = []

        for img in images:
            if self.yolo is None:
                raise RuntimeError("Ultralytics YOLO not installed")

            results = self.yolo(img, verbose=False)[0]

            boxes = results.boxes.xyxy.to(self.device)
            scores = results.boxes.conf.to(self.device)
            classes = results.boxes.cls.long().to(self.device)

            N = boxes.shape[0]

            # one-hot classes
            num_classes = int(self.yolo.model.model[-1].nc)
            onehot = torch.zeros(N, num_classes, device=self.device)
            if N > 0:
                onehot[torch.arange(N, device=self.device), classes] = 1.0

            crops = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = max(x1 + 1, x2)
                y2 = max(y1 + 1, y2)
                crop = img.crop((x1, y1, x2, y2))
                crop = self.transform(crop).unsqueeze(0).to(self.device)
                crops.append(crop)

            if len(crops) > 0:
                crops = torch.cat(crops, dim=0)
                feats = self.resnet(crops).squeeze(-1).squeeze(-1)
            else:
                feats = torch.zeros(0, 2048, device=self.device)

            class_list.append(onehot)
            feat_list.append(feats)
            score_list.append(scores)
            box_list.append(boxes)

        # pad to max N; keep one dummy masked token if a whole batch has no detections
        maxN = max([c.shape[0] for c in class_list] + [1])
        C = int(self.yolo.model.model[-1].nc)

        class_batch = torch.zeros(B, maxN, C, device=self.device)
        feat_batch = torch.zeros(B, maxN, 2048, device=self.device)
        score_batch = torch.zeros(B, maxN, device=self.device)
        box_batch = torch.zeros(B, maxN, 4, device=self.device)
        mask = torch.ones(B, maxN, dtype=torch.bool, device=self.device)

        for i in range(B):
            n = class_list[i].shape[0]
            if n == 0:
                # MultiheadAttention cannot receive a row where every token is masked.
                # Keep one zero-valued dummy token; downstream IoU matching treats it as negative.
                mask[i, 0] = False
                continue
            class_batch[i, :n] = class_list[i]
            feat_batch[i, :n] = feat_list[i]
            score_batch[i, :n] = score_list[i]
            box_batch[i, :n] = box_list[i]
            mask[i, :n] = False

        return class_batch, feat_batch, score_batch, mask, box_batch


class FullTaskDrivenPredictor(nn.Module):
    def __init__(self, num_classes=80, num_tasks=14):
        super().__init__()
        self.extractor = DetectionFeatureExtractor()
        self.model = TaskDrivenAttentionModel(num_classes, num_tasks)

    def forward(self, images):
        class_onehot, visual_feats, det_scores, mask = self.extractor(images)
        return self.model(class_onehot, visual_feats, det_scores, mask)


def gather_topk_like_model(tensor, det_scores, top_k):
    k = min(top_k, det_scores.size(1))
    _, topk_idx = torch.topk(det_scores, k=k, dim=1)
    if tensor.dim() == 3:
        return torch.gather(tensor, 1, topk_idx.unsqueeze(-1).expand(-1, -1, tensor.size(-1)))
    return torch.gather(tensor, 1, topk_idx)
