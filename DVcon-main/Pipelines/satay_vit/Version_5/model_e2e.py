"""
SATAY-ViT End-to-End Model — Version 5 (CLIP-Seeded Free Embedding)
====================================================================
What is new vs V2 (and what was wrong with V4/V5 attempts):

  V2 (best so far):  nn.Embedding(15, 256) randomly initialized, fully trainable.
                     Result: task vectors became nearly orthogonal (mean sim=0.02).
                     Limitation: random init, no semantic starting point.

  V4 (failed):       Frozen DistilBERT + frozen random Linear(768->256).
                     Result: all 14 vectors near-identical (mean sim=0.84). Useless.

  V5 v1/v2 (failed): CLIP + trainable Linear(512->256).
                     Problem: single linear map cannot untangle CLIP vectors with
                     max cosine sim=0.99. Some task pairs remain indistinguishable.

  V5 FINAL (this):   nn.Embedding(15, 256) INITIALIZED from CLIP projections,
                     then FULLY TRAINABLE — same as V2, but starts in a semantically
                     meaningful position rather than random.

  Why this works:
    - Full per-task freedom: all 3,584 embedding params can move anywhere
    - Better starting point: CLIP semantics guide the initial spatial arrangement
    - Backprop can push everything apart without constraint
    - Convergence should be faster and reach a better minimum than random V2 init

Grid size, FPN, TransformerEncoder: UNCHANGED from V2.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

CURRENT_DIR       = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_PATH = os.path.join(CURRENT_DIR, "weights_e2e", "yolo11n.pt")
DEFAULT_KNOW_PATH = os.path.join(CURRENT_DIR, "weights_e2e", "raw_knowledge_vectors.pt")


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
#  YOLO Feature Extractor  (unchanged from V2)
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
        return self._feats["p3"], self._feats["p4"], self._feats["p5"]


# ─────────────────────────────────────────────────────────────────────
#  FPN Aggregator  (unchanged from V2 — GRID=16)
# ─────────────────────────────────────────────────────────────────────
class FPNAggregator(nn.Module):
    GRID = 16

    def __init__(self, in_channels: dict, embed_dim: int = 256):
        super().__init__()
        self.proj_p3 = nn.Conv2d(in_channels["p3"], embed_dim, 1)
        self.proj_p4 = nn.Conv2d(in_channels["p4"], embed_dim, 1)
        self.proj_p5 = nn.Conv2d(in_channels["p5"], embed_dim, 1)
        self.pool    = AdaptivePool2d((self.GRID, self.GRID))
        self.norm    = nn.BatchNorm2d(embed_dim)

    def forward(self, p3, p4, p5):
        f3 = self.pool(self.proj_p3(p3))
        f4 = self.pool(self.proj_p4(p4))
        f5 = self.pool(self.proj_p5(p5))
        return self.norm(f3 + f4 + f5)


# ─────────────────────────────────────────────────────────────────────
#  ME-ViT Reasoning Head — V5 Final (CLIP-seeded free embedding)
# ─────────────────────────────────────────────────────────────────────
class MEViTHead(nn.Module):
    """
    Same cross-attention architecture as V2, but task_embedding is
    initialized from a PCA-projected CLIP embedding instead of random.

    The embedding remains fully trainable — backprop has complete freedom
    to push all 14 vectors wherever BCE loss requires, just like V2.
    The CLIP initialization simply provides a better starting position.
    """
    def __init__(
        self,
        embed_dim:      int = 256,
        num_tasks:      int = 14,
        nhead:          int = 4,
        num_layers:     int = 2,
        knowledge_path: str = DEFAULT_KNOW_PATH,
    ):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_patches = 16 * 16   # 256 tokens

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # ── Task embedding: V2-style free embedding, CLIP-seeded ─────
        # Build a (num_tasks+1, embed_dim) weight tensor to init from.
        # Index 0 = padding (zeros). Indices 1..14 = CLIP-projected seeds.
        if os.path.exists(knowledge_path):
            raw = torch.load(knowledge_path, map_location="cpu", weights_only=True)
            # raw: (14, 512) L2-normalised CLIP text embeddings
            # Project to embed_dim with a fixed PCA-like random matrix
            # (deterministic seed so it is reproducible)
            torch.manual_seed(42)
            proj = torch.randn(raw.shape[1], embed_dim) * 0.02  # (512, 256)
            seeds = raw @ proj                                   # (14, 256)
            seeds = F.normalize(seeds, dim=-1)                   # unit sphere
            # Scale to match typical embedding init magnitude
            seeds = seeds * (embed_dim ** -0.5)
            init_weight = torch.cat([torch.zeros(1, embed_dim), seeds], dim=0)  # (15, 256)
            self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim,
                                               _weight=init_weight.clone())
            print(f"  [MEViTHead] task_embedding seeded from CLIP ({knowledge_path})")
        else:
            # Fallback: random init, identical to V2
            self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim)
            print(f"  [MEViTHead] WARNING: knowledge_path not found, using random init (same as V2)")
        # ─────────────────────────────────────────────────────────────

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=embed_dim * 2,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, feat: torch.Tensor, task_id: torch.Tensor):
        """
        feat    : (B, embed_dim, 16, 16)  from FPNAggregator
        task_id : (B,) task indices in [1, 14]
        returns : (B, 16, 16) relevance heatmap in [0, 1]
        """
        B = feat.size(0)
        tokens   = feat.flatten(2).transpose(1, 2) + self.pos_embed   # (B, 256, E)
        tokens   = self.transformer(tokens)                            # self-attn
        task_emb = self.task_embedding(task_id).unsqueeze(1)           # (B, 1, E)

        Q      = self.q_proj(tokens)
        K      = self.k_proj(task_emb)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)

        heatmap = torch.sigmoid(scores.squeeze(2)).view(B, 16, 16)
        return heatmap


# ─────────────────────────────────────────────────────────────────────
#  Full End-to-End Model
# ─────────────────────────────────────────────────────────────────────
class SATAYViT_E2E(nn.Module):
    """
    Full end-to-end SATAY-ViT V5 (Final).
    Input : image (B, 3, 640, 640)  +  task_id (B,)
    Output: relevance heatmap (B, 16, 16)
    """
    def __init__(
        self,
        yolo_checkpoint: str = DEFAULT_YOLO_PATH,
        embed_dim:       int = 256,
        num_tasks:       int = 14,
        knowledge_path:  str = DEFAULT_KNOW_PATH,
    ):
        super().__init__()
        self.backbone   = YOLOBackbone(yolo_checkpoint)
        self.aggregator = FPNAggregator(self.backbone.out_channels, embed_dim)
        self.vit_head   = MEViTHead(
            embed_dim=embed_dim,
            num_tasks=num_tasks,
            knowledge_path=knowledge_path,
        )

    def forward(self, x, task_id):
        p3, p4, p5 = self.backbone(x)
        feat        = self.aggregator(p3, p4, p5)
        return self.vit_head(feat, task_id)


# ─────────────────────────────────────────────────────────────────────
#  Smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SATAYViT_E2E().to(device)

    # Check initial embedding geometry (should be better than random)
    emb = model.vit_head.task_embedding.weight[1:]  # (14, 256)
    emb_n = F.normalize(emb, dim=-1)
    sim = emb_n @ emb_n.T
    mask = ~torch.eye(14, dtype=torch.bool)
    print(f"Initial task_embedding cosine sim: mean={sim[mask].mean():.4f} "
          f"(V2 random ~0.0, V5-prev CLIP-proj ~0.30)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable : {trainable:,}  |  Total: {total:,}")

    dummy_img  = torch.randn(2, 3, 640, 640).to(device)
    dummy_task = torch.tensor([1, 7]).to(device)
    out = model(dummy_img, dummy_task)
    print(f"Output: {out.shape}  Smoke test PASSED")
