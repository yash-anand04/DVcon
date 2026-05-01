import matplotlib.pyplot as plt
import os

def plot_training_losses(train_losses, val_losses, save_dir="plots"):
    """
    Plots the training vs validation loss curves to debug underfitting/overfitting.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss')
    
    plt.title('ME-ViT Reasoner - Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

if __name__ == "__main__":
    # Dummy test
    plot_training_losses([0.13737044824228165, 0.11335331098191322, 0.10497277901877486, 0.09782154457968852, 0.08910428833216429, 0.08189815718266699],
                         [0.11554649376799707, 0.1102105066764374, 0.10697243040070638, 0.10927406076145675, 0.10736469457377762, 0.10906132369700802])
    print("Test plot generated.")
