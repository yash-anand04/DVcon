import os
import matplotlib.pyplot as plt


def plot_training_losses(train_losses, val_losses, labels=None, save_dir="weights"):
    """Save a training vs validation BCE loss curve to save_dir/loss_curve.png."""
    os.makedirs(save_dir, exist_ok=True)
    steps = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(max(8, len(train_losses) * 0.5), 5))
    ax.plot(steps, train_losses, "o-", color="#C84B0E", lw=1.5, markersize=4, label="Train BCE")
    ax.plot(steps, val_losses,   "s--", color="#4C9BE8", lw=1.5, markersize=4, label="Val BCE")

    if labels:
        ax.set_xticks(list(steps))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Sub-Epoch (shard)")
    else:
        ax.set_xlabel("Epoch")

    ax.set_ylabel("BCE Loss")
    ax.set_title("TORCA — Training / Validation Loss")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path
