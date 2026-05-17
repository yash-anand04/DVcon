"""
clip_teacher.py  –  Frozen CLIP teacher for V8 student-teacher training.

Produces a soft per-box relevance target by:
  1. Cropping each detected box from the 640×640 image,
  2. Encoding all crops in a single CLIP image-encoder forward,
  3. Encoding each task description with the CLIP text encoder (cached),
  4. Returning cosine similarity per (box, task).

The student (SATAYViT_V8) learns to regress these similarities, so it
inherits CLIP's semantic understanding of "what objects fit which tasks"
without needing CLIP at inference time.
"""

import torch
import torch.nn.functional as F
from PIL import Image

try:
    import clip as openai_clip
    CLIP_AVAILABLE = True
except Exception:
    CLIP_AVAILABLE = False


# Same descriptions as model.py — kept in sync deliberately.
TASK_DESCRIPTIONS = [
    "",                              # index 0 unused (1-indexed task ids)
    "an object you can step on",
    "an object for sitting comfortably",
    "a vase for placing flowers",
    "a tool for getting potatoes out of a fire",
    "a container for watering a plant",
    "a tool for getting a lemon out of tea",
    "a tool for digging a hole",
    "a tool for opening a bottle of beer",
    "a tool for opening a parcel",
    "a glass for serving wine",
    "a container for pouring sugar",
    "a knife for smearing butter on bread",
    "a tool for extinguishing fire",
    "a tool for pounding a carpet",
]


class CLIPTeacher:
    """
    Frozen CLIP ViT-B/32 wrapper that produces (box, task) similarity scores.

    Usage:
        teacher = CLIPTeacher(device='cuda')
        sims = teacher.score_image(pil_640, boxes_xyxy_640, task_id)   # → [N] in roughly [-0.1, 0.4]
    """

    def __init__(self, device="cpu", model_name: str = "ViT-B/32"):
        if not CLIP_AVAILABLE:
            raise RuntimeError(
                "openai-clip is not installed. `pip install openai-clip` and retry."
            )
        self.device = device
        self.model, self.preprocess = openai_clip.load(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Pre-compute task text embeddings (only 14 of them — done once).
        with torch.no_grad():
            text_embs = torch.zeros(len(TASK_DESCRIPTIONS), 512, device=device)
            for i, desc in enumerate(TASK_DESCRIPTIONS):
                if not desc:
                    continue
                tok = openai_clip.tokenize([desc]).to(device)
                emb = self.model.encode_text(tok).float()
                text_embs[i] = F.normalize(emb, dim=-1).squeeze(0)
            self.task_text_embs = text_embs   # [15, 512]

    @torch.no_grad()
    def score_image(self, pil_image_640, boxes_xyxy, task_id: int):
        """
        pil_image_640 : PIL.Image already resized to 640×640
        boxes_xyxy    : torch.Tensor [N, 4] in 640-px space
        task_id       : int, 1-indexed (matches model.py convention)
        Returns       : torch.Tensor [N] cosine sims, on self.device
        """
        if boxes_xyxy.shape[0] == 0:
            return torch.zeros(0, device=self.device)

        crops = []
        for box in boxes_xyxy.tolist():
            x1, y1, x2, y2 = box
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = max(x1 + 1, int(round(x2)))
            y2 = max(y1 + 1, int(round(y2)))
            crop = pil_image_640.crop((x1, y1, x2, y2))
            crops.append(self.preprocess(crop))
        crop_tensor = torch.stack(crops).to(self.device)        # [N, 3, 224, 224]

        img_emb = self.model.encode_image(crop_tensor).float()   # [N, 512]
        img_emb = F.normalize(img_emb, dim=-1)

        task_emb = self.task_text_embs[task_id]                  # [512]
        sims     = img_emb @ task_emb                            # [N]
        return sims

    @torch.no_grad()
    def score_batch(self, pil_images_640, boxes_per_image, task_ids):
        """
        Batched variant — runs one CLIP image-encoder forward over ALL crops
        across the mini-batch, then splits the results back per-image.

        pil_images_640   : list of B PIL images (each 640×640)
        boxes_per_image  : list of B tensors [N_i, 4]
        task_ids         : list/tensor of B ints

        Returns:
          sims_per_image : list of B tensors [N_i] of cosine similarities
        """
        if len(pil_images_640) == 0:
            return []

        all_crops, counts = [], []
        for pil, boxes in zip(pil_images_640, boxes_per_image):
            n = boxes.shape[0]
            counts.append(n)
            for box in boxes.tolist():
                x1, y1, x2, y2 = box
                x1 = max(0, int(round(x1)))
                y1 = max(0, int(round(y1)))
                x2 = max(x1 + 1, int(round(x2)))
                y2 = max(y1 + 1, int(round(y2)))
                all_crops.append(self.preprocess(pil.crop((x1, y1, x2, y2))))

        if not all_crops:
            return [torch.zeros(0, device=self.device) for _ in counts]

        crop_tensor = torch.stack(all_crops).to(self.device)
        img_emb     = self.model.encode_image(crop_tensor).float()
        img_emb     = F.normalize(img_emb, dim=-1)                # [total, 512]

        out, ptr = [], 0
        for n, t in zip(counts, task_ids):
            if n == 0:
                out.append(torch.zeros(0, device=self.device))
                continue
            task_emb = self.task_text_embs[int(t)]
            sims     = img_emb[ptr:ptr + n] @ task_emb            # [n]
            out.append(sims)
            ptr += n
        return out
