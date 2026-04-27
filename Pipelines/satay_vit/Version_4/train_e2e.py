"""
train_e2e.py  —  SATAY-ViT V4 (MLCoT) Training Script
=======================================================
Trains SATAYViT_E2E V4 (YOLO backbone + FPN @ 16×16 + ME-ViT head
with FROZEN MLCoT knowledge embeddings) on the COCO-Tasks dataset
using BCE heatmap loss.

Prerequisites:
    python generate_knowledge_embeddings.py   # creates task_knowledge_vectors.pt

Usage:
    python train_e2e.py [--epochs N] [--batch BATCH] [--lr LR] [--freeze-epochs N]
"""

import os, sys, argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make sure imports resolve from the Version_4 directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # project root
sys.path.insert(0, os.path.dirname(__file__))                   # Version_4/

from utils.data_loader import COCOTasksDataset, custom_collate
from model_e2e import SATAYViT_E2E
from utils.plot_metrics import plot_training_losses

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────
#  Defaults
# ─────────────────────────────────────────────────────────────
DEFAULTS = dict(
    data_root      = "e:/DVcon/DVcon/Data_Preprocessed",        # 16×16 dataset
    weights_dir    = os.path.join(CURRENT_DIR, "weights_e2e"),  # Version_4/weights_e2e/
    knowledge_path = os.path.join(CURRENT_DIR, "weights_e2e", "task_knowledge_vectors.pt"),
    epochs         = 15,
    batch          = 16,
    lr             = 5e-5,           # conservative: fine-tuning a pretrained backbone
    backbone_lr    = 5e-6,           # separate, lower lr for YOLO backbone layers
    freeze_epochs  = 3,              # freeze backbone for first N epochs, then unfreeze
    num_workers    = 2,
    embed_dim      = 256,
)


def build_optimizer(model, lr, backbone_lr):
    """
    Separate parameter groups: backbone gets a smaller lr.
    The task_embedding (frozen) has requires_grad=False so it is
    automatically excluded from all parameter groups.
    """
    backbone_params = list(model.backbone.parameters())
    head_params     = (
        list(model.aggregator.parameters()) +
        list(model.vit_head.parameters())
    )
    # Filter out any frozen params (task_embedding) from head_params
    head_params = [p for p in head_params if p.requires_grad]

    return optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": lr},
    ], weight_decay=1e-4)


def set_backbone_grad(model, requires_grad: bool):
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")
    print(f"Knowledge path : {cfg['knowledge_path']}")
    os.makedirs(cfg["weights_dir"], exist_ok=True)

    # ── data ─────────────────────────────────────────────────────
    print("Loading datasets …")
    train_ds = COCOTasksDataset(cfg["data_root"], split="train", grid_size=16)
    val_ds   = COCOTasksDataset(cfg["data_root"], split="test",  grid_size=16)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=custom_collate
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── model ────────────────────────────────────────────────────
    model = SATAYViT_E2E(
        embed_dim      = cfg["embed_dim"],
        knowledge_path = cfg["knowledge_path"],
    ).to(device)

    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_pars = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters : {trainable:,}  (frozen task_embedding not counted)")
    print(f"Total parameters     : {total_pars:,}")

    criterion = nn.BCELoss()
    optimizer = build_optimizer(model, cfg["lr"], cfg["backbone_lr"])

    # Cosine annealing over the full training run
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.05
    )

    # ── training loop ─────────────────────────────────────────────
    best_val_loss    = float("inf")
    train_history, val_history = [], []
    ckpt_path = os.path.join(cfg["weights_dir"], "satay_vit_e2e_best.pt")

    # Phase 1: freeze backbone for warm-up
    set_backbone_grad(model, False)
    print(f"\nPhase 1: Warming up head for {cfg['freeze_epochs']} epochs "
          f"(backbone frozen) …")

    for epoch in range(cfg["epochs"]):
        # Unfreeze backbone after warm-up period
        if epoch == cfg["freeze_epochs"]:
            set_backbone_grad(model, True)
            print("\nPhase 2: Backbone unfrozen — fine-tuning end-to-end …")

        # ── train ──────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:>2}/{cfg['epochs']} [Train]")
        for batch in pbar:
            imgs     = batch["image"].to(device)
            tasks    = batch["task_id"].to(device)
            gt_hmaps = batch["heatmap"].to(device)

            optimizer.zero_grad()
            pred = model(imgs, tasks)
            loss = criterion(pred, gt_hmaps)
            loss.backward()

            # Gradient clipping — important when fine-tuning a pretrained backbone
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train = train_loss / len(train_loader)
        scheduler.step()

        # ── validate ───────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1:>2}/{cfg['epochs']} [Valid]"):
                imgs     = batch["image"].to(device)
                tasks    = batch["task_id"].to(device)
                gt_hmaps = batch["heatmap"].to(device)
                val_loss += criterion(model(imgs, tasks), gt_hmaps).item()

        avg_val = val_loss / len(val_loader)
        train_history.append(avg_train)
        val_history.append(avg_val)

        backbone_frozen = "frozen" if epoch < cfg["freeze_epochs"] else "active"
        print(f"  Epoch {epoch+1:>2} | Train: {avg_train:.4f} | "
              f"Val: {avg_val:.4f} | backbone={backbone_frozen}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch":          epoch + 1,
                "state_dict":     model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "val_loss":       best_val_loss,
                "cfg":            cfg,
                "knowledge_path": cfg["knowledge_path"],   # store for evaluation
            }, ckpt_path)
            print(f"  ==> Saved best checkpoint → {ckpt_path}")

    # ── post-training ────────────────────────────────────────────
    log = {
        "train_history": train_history,
        "val_history":   val_history,
        "best_val_loss": best_val_loss,
        "cfg":           cfg,
    }
    log_path = os.path.join(cfg["weights_dir"], "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    plot_path = plot_training_losses(
        train_history, val_history,
        save_dir=cfg["weights_dir"]
    )
    print(f"\nTraining finished!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"  Loss curve    : {plot_path}")
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--batch",          type=int,   default=DEFAULTS["batch"])
    parser.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    parser.add_argument("--backbone-lr",    type=float, default=DEFAULTS["backbone_lr"])
    parser.add_argument("--freeze-epochs",  type=int,   default=DEFAULTS["freeze_epochs"])
    parser.add_argument("--embed-dim",      type=int,   default=DEFAULTS["embed_dim"])
    parser.add_argument("--data-root",      type=str,   default=DEFAULTS["data_root"])
    parser.add_argument("--weights-dir",    type=str,   default=DEFAULTS["weights_dir"])
    parser.add_argument("--knowledge-path", type=str,   default=DEFAULTS["knowledge_path"])
    args = parser.parse_args()

    cfg = {
        **DEFAULTS,
        "epochs":         args.epochs,
        "batch":          args.batch,
        "lr":             args.lr,
        "backbone_lr":    args.backbone_lr,
        "freeze_epochs":  args.freeze_epochs,
        "embed_dim":      args.embed_dim,
        "data_root":      args.data_root,
        "weights_dir":    args.weights_dir,
        "knowledge_path": args.knowledge_path,
    }

    train(cfg)
