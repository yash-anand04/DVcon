import os
import sys
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Add pipeline to python path so we can import dataloader
sys.path.append(os.path.dirname(__file__))
from utils.data_loader import COCOTasksDataset

TASK_NAMES = {
    1:  "Step on something",         2:  "Sit comfortably",      3:  "Place flowers",    
    4:  "Get potatoes out of fire",  5:  "Water plant",          6:  "Get lemon out of tea",
    7:  "Dig hole",                  8:  "Open bottle of beer",  9:  "Open parcel", 
    10: "Serve wine",                11: "Pour sugar",           12: "Smear butter",
    13: "Extinguish fire",           14: "Pound carpet",
}

def visualize_sample(dataset, index=0):
    """
    Pulls a sample from the dataset and plots:
    1. The raw image with all ground truth boxes (Preferred = Green, Ignored = Red)
    2. The 16x16 Ground Truth Heatmap that the ME-ViT is trained to replicate
    3. The Image with the Heatmap overlaid
    """
    sample = dataset[index]
    
    # 1. Extract standard variables
    img_tensor = sample["image"]           # Shape: [3, 640, 640]
    heatmap_tensor = sample["heatmap"]     # Shape: [16, 16]
    task_id = sample["task_id"].item()
    
    # Extract boxes and preferences out of the raw dictionary because 
    # __getitem__ deletes them to keep training tensors clean.
    # We will read them directly from the dataset's sample metadata.
    metadata = dataset.samples[index]
    boxes = metadata["boxes"]
    prefs = metadata["prefs"]
    
    # 2. Convert Image tensor back to numpy RGB for matplotlib
    # Dataset outputs normalized tensors: (img / 255.0)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    # the image is already contiguous RGB format from the dataloader
    
    # Draw Bounding Boxes
    img_with_boxes = img_np.copy()
    for box, pref in zip(boxes, prefs):
        x, y, w, h = box
        start_point = (int(x), int(y))
        end_point = (int(x + w), int(y + h))
        
        # Green for Preferred objects (pref == 1), Red for others (pref == 0)
        color = (0, 255, 0) if pref == 1 else (255, 0, 0) 
        thickness = 3 if pref == 1 else 1
        
        cv2.rectangle(img_with_boxes, start_point, end_point, color, thickness)
        
    # 3. Process Heatmap
    heatmap_np = heatmap_tensor.numpy()
    
    # Resize the 16x16 heatmap smoothly to 640x640 to match image
    heatmap_resized = cv2.resize(heatmap_np, (640, 640), interpolation=cv2.INTER_CUBIC)
    # Apply a Jet colormap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    
    # Overlay Image and Heatmap
    overlay = cv2.addWeighted(img_np, 0.5, heatmap_color, 0.5, 0)
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Task {task_id}: {TASK_NAMES[task_id]}", fontsize=18, fontweight='bold')
    
    axes[0].imshow(img_with_boxes)
    axes[0].set_title("Image + Ground Truth Boxes\n(Green = Preferred target, Red = Ignore)")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap_np, cmap='jet')
    axes[1].set_title("16x16 Heatmap Target\n(What ME-ViT tries to predict)")
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay\n(Interpolated over Image)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Loading Dataset...")
    DATA_ROOT = "e:/DVcon/DVcon/Data_Preprocessed"
    
    # Load test dataset so we don't accidentally visualize jittered/augmented images
    dataset = COCOTasksDataset(DATA_ROOT, split="test") 
    print(f"Loaded {len(dataset)} samples.")
    
    # Try finding an image where at least 1 object is preferred to make visualization useful
    for i in range(len(dataset)):
        if 1 in dataset.samples[i]["prefs"]:
            print(f"Visualizing Sample Index: {i}...")
            visualize_sample(dataset, index=i)
            break
