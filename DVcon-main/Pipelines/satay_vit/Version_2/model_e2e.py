"""
SATAY-ViT End-to-End Model
==========================
Architecture:
  - YOLOv11n backbone (layers 0-8) as a trainable feature extractor.
    Feature maps are extracted at three scales:
      P3  @ layer 4  -> (B,  64, 80, 80) -> pooled to 16x16
      P4  @ layer 6  -> (B, 128, 40, 40) -> pooled to 16x16
      P5  @ layer 8  -> (B, 256, 20, 20) -> pooled to 16x16

  - A lightweight FPN-style aggregator projects and fuses these three
    feature maps into a single (B, embed_dim, 16, 16) representation.

  - ME-ViT Reasoning Head performs cross-attention between the 256
    spatial tokens and a learned Task Embedding, outputting a
    (B, 16, 16) relevance heatmap.

Loss: BCELoss on the predicted heatmap vs. the ground-truth preference map.

Gradients flow all the way back through the backbone, so the feature
extractor learns which spatial patterns matter for each task.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_PATH = os.path.join(CURRENT_DIR, "weights_e2e", "yolo11n.pt")


# ─────────────────────────────────────────────────────────────────────
#  Helper: Adaptive Average Pool to a fixed HxW
# ─────────────────────────────────────────────────────────────────────
class AdaptivePool2d(nn.Module):
    def __init__(self, out_hw):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(out_hw)

    def forward(self, x):
        return self.pool(x)


# ─────────────────────────────────────────────────────────────────────
#  YOLO Feature Extractor (backbone only, gradient-enabled)
# ─────────────────────────────────────────────────────────────────────
class YOLOBackbone(nn.Module):
    """
    Wraps the first 9 layers of YOLOv11n and exposes multi-scale
    feature maps from layers 4, 6, and 8 via forward hooks.
    """
    HOOK_LAYERS = {4: "p3", 6: "p4", 8: "p5"}   # layer index -> name

    def __init__(self, checkpoint=DEFAULT_YOLO_PATH):
        super().__init__()
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        yolo = YOLO(checkpoint)
        full_model = yolo.model.model           # nn.Sequential of 24 modules
        # Keep only the backbone (up to + including layer 8 / SPPF = layer 9
        # We stop at layer 8 so we have raw spatial maps before SPPF/attention)
        self.backbone = nn.Sequential(*list(full_model.children())[:9])

        # Feature channels at each hook point (YOLOv11n actual):
        #  layer 4 (C3k2, stride 8)  -> 128 ch
        #  layer 6 (C3k2, stride 16) -> 128 ch
        #  layer 8 (C3k2, stride 32) -> 256 ch
        self.out_channels = {"p3": 128, "p4": 128, "p5": 256}
        self._feats = {}
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
        _ = self.backbone(x)          # run backbone; hooks fill _feats
        return (
            self._feats["p3"],        # (B,  64, 80, 80)
            self._feats["p4"],        # (B, 128, 40, 40)
            self._feats["p5"],        # (B, 256, 20, 20)
        )


# ─────────────────────────────────────────────────────────────────────
#  Feature Pyramid Aggregator
# ─────────────────────────────────────────────────────────────────────
class FPNAggregator(nn.Module):
    """
    Projects each multi-scale feature map to embed_dim channels,
    pools each to GRID_SIZE x GRID_SIZE, then sums them.
    """
    GRID = 16

    def __init__(self, in_channels: dict, embed_dim: int = 256):
        super().__init__()
        self.proj_p3 = nn.Conv2d(in_channels["p3"], embed_dim, 1)
        self.proj_p4 = nn.Conv2d(in_channels["p4"], embed_dim, 1)
        self.proj_p5 = nn.Conv2d(in_channels["p5"], embed_dim, 1)
        self.pool    = AdaptivePool2d((self.GRID, self.GRID))
        self.norm    = nn.BatchNorm2d(embed_dim)

    def forward(self, p3, p4, p5):
        f3 = self.pool(self.proj_p3(p3))   # (B, E, 16, 16)
        f4 = self.pool(self.proj_p4(p4))   # (B, E, 16, 16)
        f5 = self.pool(self.proj_p5(p5))   # (B, E, 16, 16)
        fused = f3 + f4 + f5               # element-wise sum (skip-connection style)
        return self.norm(fused)            # (B, E, 16, 16)


# ─────────────────────────────────────────────────────────────────────
#  ME-ViT Reasoning Head  (same logic, now fed YOLO features)
# ─────────────────────────────────────────────────────────────────────
class MEViTHead(nn.Module):
    """
    Cross-attention between 256 spatial tokens (from FPN) and a
    learned task embedding, producing a (B, 16, 16) relevance heatmap.
    """
    def __init__(self, embed_dim: int = 256, num_tasks: int = 14,
                 nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_patches = 16 * 16        # 256 tokens

        self.pos_embed      = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim)   # 1-indexed

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, feat: torch.Tensor, task_id: torch.Tensor):
        """
        feat    : (B, embed_dim, 16, 16)  from FPNAggregator
        task_id : (B,)
        returns : (B, 16, 16) in [0, 1]
        """
        B = feat.size(0)
        # (B, E, 16, 16) -> (B, 256, E)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed

        tokens   = self.transformer(tokens)                            # self-attn
        task_emb = self.task_embedding(task_id).unsqueeze(1)           # (B, 1, E)

        Q      = self.q_proj(tokens)                                   # (B, 256, E)
        K      = self.k_proj(task_emb)                                 # (B, 1, E)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (B,256,1)

        heatmap = torch.sigmoid(scores.squeeze(2)).view(B, 16, 16)
        return heatmap


# ─────────────────────────────────────────────────────────────────────
#  Full End-to-End Model
# ─────────────────────────────────────────────────────────────────────
class SATAYViT_E2E(nn.Module):
    """
    Full end-to-end SATAY-ViT model.
    Input : image (B,3,640,640)  +  task_id (B,)
    Output: relevance heatmap (B,16,16)
    """
    def __init__(self, yolo_checkpoint=DEFAULT_YOLO_PATH, embed_dim=256, num_tasks=14):
        super().__init__()
        self.backbone   = YOLOBackbone(yolo_checkpoint)
        self.aggregator = FPNAggregator(self.backbone.out_channels, embed_dim)
        self.vit_head   = MEViTHead(embed_dim=embed_dim, num_tasks=num_tasks)

    def forward(self, x, task_id):
        p3, p4, p5 = self.backbone(x)
        feat        = self.aggregator(p3, p4, p5)
        heatmap     = self.vit_head(feat, task_id)
        return heatmap


# ─────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SATAYViT_E2E().to(device)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")

    dummy_img  = torch.randn(2, 3, 640, 640).to(device)
    dummy_task = torch.tensor([1, 7]).to(device)
    out = model(dummy_img, dummy_task)
    print(f"Output heatmap shape: {out.shape}")   # Expected: [2, 16, 16]
    print("Smoke test passed!")
