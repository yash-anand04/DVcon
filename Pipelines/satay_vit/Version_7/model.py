"""
SATAY-ViT V7: V6 + CLIP-text task embeddings + focal/soft-IoU training
=============================================
Architecture:
  - YOLOv11n backbone (layers 0-8) extracts multi-scale FPN features:
      P3 (stride  8): (B, 128, 80, 80)
      P4 (stride 16): (B, 128, 40, 40)
      P5 (stride 32): (B, 256, 20, 20)

  - For each YOLO-detected object, RoI-Align extracts 7x7 features at
    all three scales, GAP-pools and fuses into a 256-D per-object token.
    No ResNet crops, no serial per-object passes.

  - Class one-hot (80-D) projected and added for semantic context.

  - SelfAttentionBlock stack: objects share context across the scene.

  - TaskObjectScorer: task embedding cross-attends to object tokens,
    yielding a per-object task-relevance score in [0,1].

  - Final score: S_yolo_conf * S_task_relevance → best box = argmax.

FPGA advantage over V1B (29M ResNet50):
  ~3-4M total params, single FPN pass, deterministic latency.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align as tv_roi_align

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import clip as openai_clip
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False

# Natural-language descriptions used to build CLIP task embeddings.
# Written as short imperative phrases to match CLIP's training distribution.
_TASK_DESCRIPTIONS = [
    "",                             # index 0 — unused (tasks are 1-indexed)
    "step on something",
    "sit comfortably",
    "place flowers in a vase",
    "get potatoes out of a fire",
    "water a plant",
    "get a lemon out of tea",
    "dig a hole in the ground",
    "open a bottle of beer",
    "open a parcel or box",
    "serve wine in a glass",
    "pour sugar",
    "smear butter on bread",
    "extinguish a fire",
    "pound a carpet",
]


def _build_clip_task_embeddings(descriptions, clip_dim: int = 512) -> torch.Tensor:
    """
    Encodes task descriptions with a frozen CLIP ViT-B/32 text encoder.
    Returns a [len(descriptions), clip_dim] float32 tensor (L2-normalised).
    Falls back to random unit vectors if CLIP is unavailable.
    """
    n = len(descriptions)
    if not CLIP_AVAILABLE:
        embs = F.normalize(torch.randn(n, clip_dim), dim=-1)
        return embs

    clip_model, _ = openai_clip.load("ViT-B/32", device="cpu")
    clip_model.eval()
    embs = torch.zeros(n, clip_dim)
    with torch.no_grad():
        for i, desc in enumerate(descriptions):
            if not desc:
                continue  # leave index 0 as zeros
            tokens = openai_clip.tokenize([desc])
            feat   = clip_model.encode_text(tokens).float()
            embs[i] = F.normalize(feat, dim=-1)
    del clip_model  # free CLIP weights — not needed after init
    return embs

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_PATH = os.environ.get(
    "SATAY_YOLO_PATH",
    os.path.join(CURRENT_DIR, "weights", "yolo11n.pt"),
)


# ─────────────────────────────────────────────────────────────────────
#  FPN Backbone (YOLO layers 0-8, trainable)
# ─────────────────────────────────────────────────────────────────────
class FPNBackbone(nn.Module):
    """YOLO layers 0-8 with forward hooks to expose P3/P4/P5."""

    HOOK_LAYERS = {4: "p3", 6: "p4", 8: "p5"}

    def __init__(self, checkpoint=DEFAULT_YOLO_PATH):
        super().__init__()
        chk_dir = os.path.dirname(checkpoint)
        if chk_dir:
            os.makedirs(chk_dir, exist_ok=True)
        yolo = YOLO(checkpoint)
        self.backbone = nn.Sequential(*list(yolo.model.model.children())[:9])
        self.out_channels = {"p3": 128, "p4": 128, "p5": 256}
        self._feats: dict = {}
        self._register_hooks()

    def _register_hooks(self):
        layers = list(self.backbone.children())
        for idx, name in self.HOOK_LAYERS.items():
            layers[idx].register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._feats[name] = out
        return hook

    def forward(self, x):
        self._feats = {}
        _ = self.backbone(x)
        return self._feats["p3"], self._feats["p4"], self._feats["p5"]


# ─────────────────────────────────────────────────────────────────────
#  Multi-Scale RoI Fusion
# ─────────────────────────────────────────────────────────────────────
class MultiScaleRoIFusion(nn.Module):
    """
    For each detected box: RoI-Align on P3/P4/P5 + class one-hot,
    fused via MLP into a 256-D per-object embedding.
    """

    def __init__(self, in_channels: dict, embed_dim: int = 256,
                 roi_size: int = 7, num_classes: int = 80):
        super().__init__()
        self.embed_dim = embed_dim
        self.roi_size = roi_size
        self.proj_p3 = nn.Conv2d(in_channels["p3"], embed_dim, 1)
        self.proj_p4 = nn.Conv2d(in_channels["p4"], embed_dim, 1)
        self.proj_p5 = nn.Conv2d(in_channels["p5"], embed_dim, 1)
        self.class_proj = nn.Linear(num_classes, embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, p3, p4, p5, det_results):
        """
        det_results : list of (boxes_xyxy [N_i,4], scores [N_i], classes [N_i])
                      one tuple per image, boxes in 640-px xyxy space.
        Returns
          obj_feats  : [B, maxN, embed_dim]
          det_scores : [B, maxN]  — YOLO detection confidence
          mask       : [B, maxN]  — True = padding token (no real object)
        """
        B = p3.shape[0]
        device = p3.device
        num_classes = self.class_proj.in_features

        p3 = self.proj_p3(p3)   # [B, E, 80, 80]
        p4 = self.proj_p4(p4)   # [B, E, 40, 40]
        p5 = self.proj_p5(p5)   # [B, E, 20, 20]

        rois_list, n_per = [], []
        all_scores, all_onehot = [], []

        for i, (boxes, scores, classes) in enumerate(det_results):
            n = boxes.shape[0]
            n_per.append(n)
            if n > 0:
                bidx = torch.full((n, 1), float(i), device=device)
                rois_list.append(torch.cat([bidx, boxes.to(device)], dim=1))
                all_scores.append(scores.to(device))
                oh = torch.zeros(n, num_classes, device=device)
                oh[torch.arange(n, device=device), classes.to(device)] = 1.0
                all_onehot.append(oh)

        maxN = max(max(n_per, default=0), 1)
        obj_feats   = torch.zeros(B, maxN, self.embed_dim, device=device)
        det_scores_ = torch.zeros(B, maxN, device=device)
        mask        = torch.ones(B, maxN, dtype=torch.bool, device=device)

        total = sum(n_per)
        if total > 0:
            rois           = torch.cat(rois_list, dim=0)            # [total, 5]
            all_scores_cat = torch.cat(all_scores, dim=0)           # [total]
            all_onehot_cat = torch.cat(all_onehot, dim=0)           # [total, C]

            f3 = tv_roi_align(p3, rois, self.roi_size, spatial_scale=1/8,  aligned=True).mean([-2,-1])
            f4 = tv_roi_align(p4, rois, self.roi_size, spatial_scale=1/16, aligned=True).mean([-2,-1])
            f5 = tv_roi_align(p5, rois, self.roi_size, spatial_scale=1/32, aligned=True).mean([-2,-1])
            fc = F.relu(self.class_proj(all_onehot_cat))             # [total, E]
            fused = self.fuse(torch.cat([f3, f4, f5, fc], dim=1))   # [total, E]

            ptr = 0
            for i, n in enumerate(n_per):
                if n > 0:
                    obj_feats[i, :n]   = fused[ptr:ptr+n]
                    det_scores_[i, :n] = all_scores_cat[ptr:ptr+n]
                    mask[i, :n]        = False
                else:
                    mask[i, 0] = False  # keep one dummy visible token
                ptr += n
        else:
            for i in range(B):
                mask[i, 0] = False

        return obj_feats, det_scores_, mask


# ─────────────────────────────────────────────────────────────────────
#  Self-Attention Block (Pre-LN, with FFN)
# ─────────────────────────────────────────────────────────────────────
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ─────────────────────────────────────────────────────────────────────
#  Task Object Scorer
# ─────────────────────────────────────────────────────────────────────
class TaskObjectScorer(nn.Module):
    """Self-attention + task cross-attention → per-object relevance in [0,1].

    Task queries are built from frozen CLIP ViT-B/32 text embeddings of the
    14 task descriptions, projected to embed_dim by a trainable linear layer.
    This gives semantically structured task representations from the start,
    replacing the random-init nn.Embedding used in earlier versions.
    """

    def __init__(self, embed_dim: int = 256,
                 nhead: int = 4, num_layers: int = 2, clip_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        # Frozen CLIP text features, shape [num_tasks+1, clip_dim]
        clip_embs = _build_clip_task_embeddings(_TASK_DESCRIPTIONS, clip_dim)
        self.register_buffer("task_clip_embs", clip_embs)   # saved in state_dict, not trained

        # Trainable projection: CLIP space → model embed space
        self.task_proj  = nn.Linear(clip_dim, embed_dim)

        self.self_attn  = nn.ModuleList([
            SelfAttentionBlock(embed_dim, nhead) for _ in range(num_layers)
        ])
        self.q_proj     = nn.Linear(embed_dim, embed_dim)
        self.k_proj     = nn.Linear(embed_dim, embed_dim)
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, obj_feats, task_id, mask=None):
        """
        obj_feats : [B, N, E]
        task_id   : [B,] int, 1-indexed
        mask      : [B, N] bool — True = padding
        Returns   : [B, N] per-object task-relevance scores in [0,1]
        """
        x = obj_feats
        for layer in self.self_attn:
            x = layer(x, mask)

        # Look up frozen CLIP features, project to embed_dim
        clip_feat = self.task_clip_embs[task_id]                    # [B, clip_dim]
        task_emb  = self.task_proj(clip_feat).unsqueeze(1)          # [B, 1, E]

        Q = self.q_proj(task_emb)                                   # [B, 1, E]
        K = self.k_proj(x)                                          # [B, N, E]
        attn = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)  # [B,1,N]

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))

        weights  = torch.softmax(attn, dim=-1)                      # [B, 1, N]
        attended = weights.transpose(1, 2) * x                     # [B, N, E]
        return torch.sigmoid(self.score_head(attended).squeeze(-1)) # [B, N]


# ─────────────────────────────────────────────────────────────────────
#  Full V7 Model
# ─────────────────────────────────────────────────────────────────────
class SATAYViT_V7(nn.Module):
    """
    V7 = V6 + CLIP-text task embeddings + focal-loss/soft-IoU training.

    Architecture is identical to V6 except TaskObjectScorer uses frozen
    CLIP ViT-B/32 text features (instead of nn.Embedding) projected to
    embed_dim by a trainable linear layer.

    Input : image tensor (B,3,640,640) + det_results list + task_id (B,)
    Output: rel_scores [B,maxN], det_scores [B,maxN], mask [B,maxN]
    """

    def __init__(self, checkpoint=DEFAULT_YOLO_PATH,
                 embed_dim: int = 256, num_tasks: int = 14, num_classes: int = 80):
        super().__init__()
        self.backbone   = FPNBackbone(checkpoint)
        self.roi_fusion = MultiScaleRoIFusion(
            self.backbone.out_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )
        self.scorer   = TaskObjectScorer(embed_dim=embed_dim)
        self.num_tasks = num_tasks

    def forward(self, img_tensor, det_results, task_id):
        p3, p4, p5                  = self.backbone(img_tensor)
        obj_feats, det_scores, mask = self.roi_fusion(p3, p4, p5, det_results)
        rel_scores                  = self.scorer(obj_feats, task_id, mask)
        return rel_scores, det_scores, mask


# ─────────────────────────────────────────────────────────────────────
#  Frozen YOLO Detector Helper
# ─────────────────────────────────────────────────────────────────────
class YOLODetector:
    """Frozen YOLO wrapper for object detection. Not an nn.Module."""

    def __init__(self, checkpoint=DEFAULT_YOLO_PATH, device="cpu"):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        self.yolo        = YOLO(checkpoint)
        self.device      = device
        self.num_classes = int(self.yolo.model.model[-1].nc)

    @torch.no_grad()
    def detect_batch(self, pil_images):
        """Returns list of (boxes_xyxy [N,4], scores [N], classes [N]) per image."""
        out = []
        for img in pil_images:
            r       = self.yolo(img, verbose=False)[0]
            boxes   = r.boxes.xyxy.to(self.device)
            scores  = r.boxes.conf.to(self.device)
            classes = r.boxes.cls.long().to(self.device)
            out.append((boxes, scores, classes))
        return out


# ─────────────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SATAYViT_V7().to(device)
    total  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total:,}")

    B = 2
    img   = torch.randn(B, 3, 640, 640).to(device)
    task  = torch.tensor([1, 7]).to(device)
    dets  = [
        (torch.rand(5, 4).to(device) * 640, torch.rand(5).to(device), torch.randint(0, 80, (5,)).to(device)),
        (torch.rand(3, 4).to(device) * 640, torch.rand(3).to(device), torch.randint(0, 80, (3,)).to(device)),
    ]
    rel, det, mask = model(img, dets, task)
    print(f"rel_scores: {rel.shape}, det_scores: {det.shape}, mask: {mask.shape}")
    print("Smoke test passed!")
