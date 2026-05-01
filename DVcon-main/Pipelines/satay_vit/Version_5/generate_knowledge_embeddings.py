"""
generate_knowledge_embeddings.py  —  SATAY-ViT V5
==================================================
Generates the 14 RAW CLIP text embeddings (one per COCO-Task) and saves
them as a (14, 512) float32 tensor to:
    Version_5/weights_e2e/raw_knowledge_vectors.pt

KEY DIFFERENCE FROM V4:
  - Uses CLIP ViT-B/16 text encoder (visually grounded, not language-only)
  - Output is 512D, NOT projected to 256D here
  - The projection (512→256) now lives INSIDE MEViTHead and is TRAINABLE
  - This lets backprop push the 14 task vectors apart during training

Usage:
    python generate_knowledge_embeddings.py
"""

import os
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────
#  MLCoT Level-3 Visual Attribute Strings (one per COCO-Task, 1-indexed)
# ─────────────────────────────────────────────────────────────────────
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
#  CLIP Text Encoder (frozen — outputs 512D visually-grounded embeddings)
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_with_clip(texts: list, device: str = "cpu") -> torch.Tensor:
    """
    Encodes each text string using CLIP ViT-B/16 text encoder.
    Returns (N, 512) float32 tensor, L2-normalised.

    CLIP text embeddings are visually grounded — trained to be aligned
    with image encoder features. This is a much better starting point
    than DistilBERT for visual cross-attention tasks.
    """
    from transformers import CLIPTextModel, CLIPTokenizer

    model_name = "openai/clip-vit-base-patch16"
    print(f"  Loading CLIP text encoder: {model_name} ...")
    tokenizer  = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModel.from_pretrained(model_name).to(device)
    text_model.eval()

    # CLIP tokenizer has max_length=77
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    out = text_model(**tokens)
    # Use the pooler_output (EOS token representation) — the standard CLIP text embedding
    embeddings = out.pooler_output        # (N, 512)

    # L2-normalise
    embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)
    return embeddings.cpu().float()


# ─────────────────────────────────────────────────────────────────────
#  Main: generate and save
# ─────────────────────────────────────────────────────────────────────
def generate(device: str = "cpu"):
    out_dir  = os.path.join(os.path.dirname(__file__), "weights_e2e")
    out_path = os.path.join(out_dir, "raw_knowledge_vectors.pt")
    os.makedirs(out_dir, exist_ok=True)

    texts = [TASK_KNOWLEDGE[t] for t in sorted(TASK_KNOWLEDGE)]

    print("=" * 60)
    print("  SATAY-ViT V5 -- CLIP Knowledge Embedding Generation")
    print("=" * 60)
    print("  NOTE: Projection (512D->256D) is now INSIDE the model")
    print("        and is TRAINABLE. Only raw CLIP embeddings saved here.")
    print()
    for t, txt in zip(sorted(TASK_KNOWLEDGE), texts):
        print(f"  Task {t:>2}: {txt[:60]}...")

    vectors = encode_with_clip(texts, device=device)   # (14, 512)

    # Verify inter-task separability — should be << V4's 0.841
    sim_mat  = vectors @ vectors.T
    mask     = ~torch.eye(14, dtype=torch.bool)
    off_diag = sim_mat[mask]
    print(f"\n  Generated tensor shape  : {tuple(vectors.shape)}")
    print(f"  Vector norms            : min={vectors.norm(dim=-1).min():.4f}  "
          f"max={vectors.norm(dim=-1).max():.4f}")
    print(f"  Inter-task cosine sim   : mean={off_diag.mean():.4f}  "
          f"min={off_diag.min():.4f}  max={off_diag.max():.4f}")
    print(f"  (lower mean = more separated = better task discrimination)")

    torch.save(vectors, out_path)
    print(f"\n  Saved -> {out_path}")
    print("=" * 60)
    return vectors


if __name__ == "__main__":
    generate(device="cuda" if torch.cuda.is_available() else "cpu")
