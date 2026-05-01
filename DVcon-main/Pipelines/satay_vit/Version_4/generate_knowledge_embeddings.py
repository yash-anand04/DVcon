"""
generate_knowledge_embeddings.py  —  SATAY-ViT V4 (Phase 0)
============================================================
Runs ONCE before training to produce the 14 MLCoT knowledge vectors
(one per COCO-Task) and saves them as a (14, 256) float32 tensor to:
    Version_4/weights_e2e/task_knowledge_vectors.pt

The vectors are derived from a 3-level Chain-of-Thought distillation:
  Level 1 — Object brainstorm  (which objects afford this task?)
  Level 2 — Affordance rationales (why are they suitable?)
  Level 3 — Visual attribute condensation (what does it look like?)

The Level-3 string for each task is encoded via a frozen DistilBERT
text encoder and projected to embed_dim=256.

Usage:
    python generate_knowledge_embeddings.py
"""

import os
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────
#  MLCoT Level-3 Visual Attribute Strings (one per COCO-Task, 1-indexed)
# ─────────────────────────────────────────────────────────────────────
# Each string was derived by the 3-level chain:
#   L1: what objects afford the task?
#   L2: why are they suitable (rationale)?
#   L3: distilled into concrete, visually-grounded descriptors.

TASK_KNOWLEDGE = {
    1:  "flat stable surface, rigid structural body, elevated platform, "
        "wide load-bearing base, non-slip horizontal top",

    2:  "padded cushioned seat, ergonomic curved backrest, armrests, "
        "four stable legs, soft upholstered surface",

    3:  "deep cylindrical opening, water-holding vessel, wide mouth rim, "
        "stable flat base, narrow neck or vase silhouette",

    4:  "long heat-resistant handle, metal tong ends, fork prongs, "
        "rigid structural body, gripping claw tip",

    5:  "elongated curved spout, handle for pouring, enclosed liquid container, "
        "wide water reservoir body, nozzle tip opening",

    6:  "slotted perforated bowl head, fine mesh holes, long thin handle, "
        "strainer basket shape, small circular perforations",

    7:  "sharp pointed tip, long leveraging handle, flat blade edge, "
        "rigid metal body, shovel or trowel head",

    8:  "hard rigid lever edge, circular bottle opener head, "
        "fulcrum notch for cap grip, compact metal body, prying flat tip",

    9:  "sharp cutting blade edge, pointed fine tip, short control handle, "
        "thin narrow blade body, serrated or straight cutting edge",

    10: "tall stem for grip, round concave bowl shape, transparent glass material, "
        "thin flared rim at top, wine glass silhouette",

    11: "narrow pour spout opening, granular flow nozzle, "
        "cylindrical dispenser body, flip-top or screw lid, small hole aperture",

    12: "wide flat blade head, flexible thin spreading edge, "
        "short handle, smooth spatula surface, broad rectangular tip",

    13: "large water-holding capacity, open bucket or pitcher shape, "
        "handle for carrying, wide mouth opening, non-flammable exterior",

    14: "heavy broad flat striking head, long swing handle, "
        "mallet or beater silhouette, padded wide surface, rigid body",
}


# ─────────────────────────────────────────────────────────────────────
#  Text Encoder (frozen DistilBERT → mean pool → project to embed_dim)
# ─────────────────────────────────────────────────────────────────────
class TextKnowledgeEncoder(nn.Module):
    """
    Encodes a list of attribute strings into fixed-size vectors.
    Uses a frozen DistilBERT for token embeddings + mean pooling,
    then a learned linear projection down to embed_dim (256).
    The projection is fit by closed-form on the 14 task strings.
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        model_name = "distilbert-base-uncased"
        print(f"  Loading frozen text encoder: {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False  # always frozen

        # Linear projection from DistilBERT hidden size (768) → embed_dim
        self.proj = nn.Linear(768, embed_dim, bias=False)

    @torch.no_grad()
    def encode(self, texts: list[str], device: str = "cpu") -> torch.Tensor:
        """
        texts  : list of N attribute strings
        returns: (N, embed_dim) float32 tensor (L2-normalised)
        """
        self.encoder.to(device)
        self.proj.to(device)
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        out         = self.encoder(**tokens)           # (N, seq_len, 768)
        attn_mask   = tokens["attention_mask"]         # (N, seq_len)
        mask_exp    = attn_mask.unsqueeze(-1).float()  # (N, seq_len, 1)
        pooled      = (out.last_hidden_state * mask_exp).sum(1) / mask_exp.sum(1)  # (N, 768)
        projected   = self.proj(pooled)               # (N, embed_dim)

        # L2-normalise so cosine similarity is well-behaved in cross-attention
        projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)
        return projected.cpu()


# ─────────────────────────────────────────────────────────────────────
#  Main: generate and save
# ─────────────────────────────────────────────────────────────────────
def generate(embed_dim: int = 256, device: str = "cpu"):
    out_dir  = os.path.join(os.path.dirname(__file__), "weights_e2e")
    out_path = os.path.join(out_dir, "task_knowledge_vectors.pt")
    os.makedirs(out_dir, exist_ok=True)

    # Collect strings in task-id order (1→14)
    texts = [TASK_KNOWLEDGE[t] for t in sorted(TASK_KNOWLEDGE)]

    print("="*60)
    print("  SATAY-ViT V4 — MLCoT Knowledge Embedding Generation")
    print("="*60)
    for t, txt in zip(sorted(TASK_KNOWLEDGE), texts):
        print(f"  Task {t:>2}: {txt[:60]}…")
    print()

    encoder  = TextKnowledgeEncoder(embed_dim=embed_dim)
    vectors  = encoder.encode(texts, device=device)   # (14, 256)

    print(f"  Generated tensor shape : {tuple(vectors.shape)}")
    print(f"  Vector norms           : min={vectors.norm(dim=-1).min():.4f}  "
          f"max={vectors.norm(dim=-1).max():.4f}")

    torch.save(vectors, out_path)
    print(f"\n  Saved -> {out_path}")
    print("="*60)
    return vectors


if __name__ == "__main__":
    generate(embed_dim=256, device="cuda" if torch.cuda.is_available() else "cpu")
