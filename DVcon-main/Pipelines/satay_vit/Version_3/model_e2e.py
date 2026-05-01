"""
SATAY-ViT End-to-End Model (Updated)
==========================
Architecture:
  - YOLOv11n backbone (layers 0-8) as a trainable feature extractor.
    Feature maps are extracted at three scales:
      P3  @ layer 4  -> (B,  64, 80, 80) -> pooled to 32x32
      P4  @ layer 6  -> (B, 128, 40, 40) -> pooled to 32x32
      P5  @ layer 8  -> (B, 256, 20, 20) -> pooled to 32x32

  - A lightweight FPN-style aggregator projects and fuses these three
    feature maps into a single (B, embed_dim, 32, 32) representation.

  - ME-ViT Reasoning Head performs Top-N Sparse self-attention 
    and cross-attention between the 1024 spatial tokens and a 
    learned Task Embedding, outputting a (B, 32, 32) relevance heatmap.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_PATH = os.path.join(CURRENT_DIR, "weights", "yolo11n.pt")


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
    HOOK_LAYERS = {4: "p3", 6: "p4", 8: "p5"}

    def __init__(self, checkpoint=DEFAULT_YOLO_PATH):
        super().__init__()
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        yolo = YOLO(checkpoint)
        full_model = yolo.model.model
        
        self.backbone = nn.Sequential(*list(full_model.children())[:9])
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
        _ = self.backbone(x)
        return (
            self._feats["p3"],
            self._feats["p4"],
            self._feats["p5"],
        )


# ─────────────────────────────────────────────────────────────────────
#  Feature Pyramid Aggregator (UPDATED to 32x32 Grid)
# ─────────────────────────────────────────────────────────────────────
class FPNAggregator(nn.Module):
    GRID = 32  # Changed from 16 to 32

    def __init__(self, in_channels: dict, embed_dim: int = 256):
        super().__init__()
        self.proj_p3 = nn.Conv2d(in_channels["p3"], embed_dim, 1)
        self.proj_p4 = nn.Conv2d(in_channels["p4"], embed_dim, 1)
        self.proj_p5 = nn.Conv2d(in_channels["p5"], embed_dim, 1)
        self.pool    = AdaptivePool2d((self.GRID, self.GRID))
        self.norm    = nn.BatchNorm2d(embed_dim)

    def forward(self, p3, p4, p5):
        f3 = self.pool(self.proj_p3(p3))   # (B, E, 32, 32)
        f4 = self.pool(self.proj_p4(p4))   # (B, E, 32, 32)
        f5 = self.pool(self.proj_p5(p5))   # (B, E, 32, 32)
        fused = f3 + f4 + f5               
        return self.norm(fused)            


# ─────────────────────────────────────────────────────────────────────
#  Top-N Sparse Transformer Layer (NEW)
# ─────────────────────────────────────────────────────────────────────
class TopKSparseTransformerLayer(nn.Module):
    """
    Applies Top-N Sparse Self-Attention to focus processing resources 
    on the most relevant spatial token interactions.
    """
    def __init__(self, embed_dim, nhead, dim_feedforward, top_k, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.nhead = nhead
        self.head_dim = embed_dim // nhead

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape
        res = x
        x = self.norm1(x)

        # Multi-head Projections
        q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        # Attention Scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Top-K Sparse Masking
        if self.top_k < N:
            threshold, _ = torch.topk(scores, self.top_k, dim=-1)
            threshold = threshold[..., -1, None]
            mask = scores < threshold
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Recombine
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, E)
        out = self.out_proj(out)

        # Residuals & FFN
        x = res + self.dropout(out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────
#  ME-ViT Reasoning Head (UPDATED: 32x32 Grid & Sparse Attention)
# ─────────────────────────────────────────────────────────────────────
class MEViTHead(nn.Module):
    def __init__(self, embed_dim: int = 256, num_tasks: int = 14,
                 nhead: int = 4, num_layers: int = 2, top_k: int = 256):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_patches = 32 * 32        # 1024 tokens

        self.pos_embed      = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim)

        # Replace standard TransformerEncoder with our Sparse version
        self.layers = nn.ModuleList([
            TopKSparseTransformerLayer(embed_dim, nhead, embed_dim * 2, top_k)
            for _ in range(num_layers)
        ])
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, feat: torch.Tensor, task_id: torch.Tensor):
        B = feat.size(0)
        # (B, E, 32, 32) -> (B, 1024, E)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed

        # Sparse Self-Attention
        for layer in self.layers:
            tokens = layer(tokens)

        # Task-Guided Cross-Attention
        task_emb = self.task_embedding(task_id).unsqueeze(1)           
        Q = self.q_proj(tokens)                                   
        K = self.k_proj(task_emb)                                 
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5) 

        # Output Reshape to 32x32
        heatmap = torch.sigmoid(scores.squeeze(2)).view(B, 32, 32)
        return heatmap


# ─────────────────────────────────────────────────────────────────────
#  Full End-to-End Model
# ─────────────────────────────────────────────────────────────────────
class SATAYViT_E2E(nn.Module):
    def __init__(self, yolo_checkpoint=DEFAULT_YOLO_PATH, embed_dim=256, num_tasks=14, top_k=256):
        super().__init__()
        self.backbone   = YOLOBackbone(yolo_checkpoint)
        self.aggregator = FPNAggregator(self.backbone.out_channels, embed_dim)
        self.vit_head   = MEViTHead(embed_dim=embed_dim, num_tasks=num_tasks, top_k=top_k)

    def forward(self, x, task_id):
        p3, p4, p5  = self.backbone(x)
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
    print(f"Output heatmap shape: {out.shape}")   # Expected: [2, 32, 32]
    print("Smoke test passed!")