"""
SATAY-ViT End-to-End Model — Version 4 (MLCoT)
===============================================
Architecture: identical to Version 2, with ONE targeted change —
  the task embedding is no longer a learned nn.Embedding but is instead
  replaced by a FROZEN embedding loaded from LLM-distilled MLCoT
  knowledge vectors (see generate_knowledge_embeddings.py).

  V2 baseline components (UNCHANGED):
    - YOLOv11n backbone (layers 0-8), hooked at layers 4, 6, 8
    - FPN aggregator → 16×16 grid, embed_dim=256
    - ME-ViT Reasoner: standard TransformerEncoder (2L, 4H), 256 tokens
    - Score fusion: S_final = C_i × R_box_i on the 16×16 heatmap
    - BCE loss, AdamW, cosine LR schedule, 3-epoch freeze warm-up

  V4 delta:
    - task_embedding: nn.Embedding(15, 256) [random init, trainable]
      → nn.Embedding.from_pretrained(vectors, freeze=True)
         where vectors = (14, 256) tensor from task_knowledge_vectors.pt
         (LLM-derived visual attribute strings → DistilBERT → proj 256D)

Output: (B, 16, 16) relevance heatmap  ← unchanged from V2.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

CURRENT_DIR      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YOLO_PATH = os.path.join(CURRENT_DIR, "weights_e2e", "yolo11n.pt")
DEFAULT_KNOW_PATH = os.path.join(CURRENT_DIR, "weights_e2e", "task_knowledge_vectors.pt")


# ─────────────────────────────────────────────────────────────────────
#  Helper: Adaptive Average Pool to a fixed HxW  (unchanged from V2)
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
    """
    Wraps the first 9 layers of YOLOv11n and exposes multi-scale
    feature maps from layers 4, 6, and 8 via forward hooks.
    """
    HOOK_LAYERS = {4: "p3", 6: "p4", 8: "p5"}   # layer index → name

    def __init__(self, checkpoint=DEFAULT_YOLO_PATH):
        super().__init__()
        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        yolo = YOLO(checkpoint)
        full_model = yolo.model.model           # nn.Sequential of 24 modules
        self.backbone = nn.Sequential(*list(full_model.children())[:9])

        # Feature channels at each hook point (YOLOv11n actual):
        #  layer 4 (C3k2, stride 8)  → 128 ch
        #  layer 6 (C3k2, stride 16) → 128 ch
        #  layer 8 (C3k2, stride 32) → 256 ch
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
            self._feats["p3"],        # (B, 128, 80, 80)
            self._feats["p4"],        # (B, 128, 40, 40)
            self._feats["p5"],        # (B, 256, 20, 20)
        )


# ─────────────────────────────────────────────────────────────────────
#  Feature Pyramid Aggregator  (unchanged from V2 — GRID=16)
# ─────────────────────────────────────────────────────────────────────
class FPNAggregator(nn.Module):
    """
    Projects each multi-scale feature map to embed_dim channels,
    pools each to 16×16, then sums (skip-connection style).
    Grid is identical to V2: 16×16 → 256 tokens.
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
        fused = f3 + f4 + f5               # element-wise sum
        return self.norm(fused)            # (B, E, 16, 16)


# ─────────────────────────────────────────────────────────────────────
#  ME-ViT Reasoning Head — V4 MLCoT variant
# ─────────────────────────────────────────────────────────────────────
class MEViTHead(nn.Module):
    """
    Cross-attention between 256 spatial tokens (from FPN) and a
    FROZEN knowledge-aware task embedding, producing a (B, 16, 16)
    relevance heatmap.

    Key difference from V2:
      - task_embedding is loaded from task_knowledge_vectors.pt
        (14 × embed_dim LLM-distilled attribute vectors) and is FROZEN.
      - All other components (pos_embed, transformer, q_proj, k_proj)
        are trainable, exactly as in V2.
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
        self.num_patches = 16 * 16        # 256 tokens (same as V2)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # ── V4 change: frozen knowledge embedding ───────────────────
        if not os.path.exists(knowledge_path):
            raise FileNotFoundError(
                f"Knowledge vectors not found at:\n  {knowledge_path}\n"
                f"Run generate_knowledge_embeddings.py first."
            )
        knowledge = torch.load(knowledge_path, map_location="cpu", weights_only=True)
        # knowledge: (14, embed_dim) — tasks are 1-indexed, so pad index 0
        if knowledge.shape != (num_tasks, embed_dim):
            raise ValueError(
                f"Expected knowledge tensor of shape ({num_tasks}, {embed_dim}), "
                f"got {tuple(knowledge.shape)}."
            )
        padding = torch.zeros(1, embed_dim)                     # index 0 (unused)
        padded  = torch.cat([padding, knowledge], dim=0)        # (15, embed_dim)
        self.task_embedding = nn.Embedding.from_pretrained(padded, freeze=True)
        # ─────────────────────────────────────────────────────────────

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
        task_id : (B,)  task indices in [1, 14]
        returns : (B, 16, 16) relevance heatmap in [0, 1]
        """
        B = feat.size(0)
        # (B, E, 16, 16) → (B, 256, E)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed

        tokens   = self.transformer(tokens)                            # self-attn
        task_emb = self.task_embedding(task_id).unsqueeze(1)           # (B, 1, E)  frozen

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
    Full end-to-end SATAY-ViT V4 model.
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
        heatmap     = self.vit_head(feat, task_id)
        return heatmap


# ─────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SATAYViT_E2E().to(device)

    total     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters : {total:,}")
    print(f"Total parameters     : {total_all:,}  (frozen task_embedding excluded from grad)")

    dummy_img  = torch.randn(2, 3, 640, 640).to(device)
    dummy_task = torch.tensor([1, 7]).to(device)
    out = model(dummy_img, dummy_task)
    print(f"Output heatmap shape : {out.shape}")   # Expected: [2, 16, 16]
    print("Smoke test passed!")
