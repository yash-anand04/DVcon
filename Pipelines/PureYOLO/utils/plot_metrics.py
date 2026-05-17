import matplotlib.pyplot as plt
import os


def plot_training_losses(train_losses, val_losses, labels=None, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    x = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, train_losses, "b-o", label="Training Loss")
    ax.plot(x, val_losses,   "r-s", label="Validation Loss")

    if labels:
        tick_step = max(1, len(labels) // 20)
        ax.set_xticks(list(x)[::tick_step])
        ax.set_xticklabels(labels[::tick_step], rotation=45, ha="right", fontsize=7)
    else:
        ax.set_xlabel("Epoch")

    ax.set_title("V1C (PureYOLO) — Training / Validation Loss")
    ax.set_ylabel("BCE Loss")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    save_path = os.path.join(save_dir, "loss_curve.png")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path

if __name__ == "__main__":
    # Dummy test
    plot_training_losses([0.13737044824228165, 0.11335331098191322, 0.10497277901877486, 0.09782154457968852, 0.08910428833216429, 0.08189815718266699],
                         [0.11554649376799707, 0.1102105066764374, 0.10697243040070638, 0.10927406076145675, 0.10736469457377762, 0.10906132369700802])
    print("Test plot generated.")
