import os
import json
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class COCOTasksDataset(Dataset):
    def __init__(self, data_root, split='train', grid_size=16):
        self.grid_size = grid_size
        """
        data_root: e.g. "e:/DVcon/DVcon/Data_Preprocessed"
        split: "train" or "test"
        """
        self.split_dir = os.path.join(data_root, split)
        index_path = os.path.join(self.split_dir, "samples.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} not found. Did you run preprocess_dataset.py?")
            
        with open(index_path, "r") as f:
            self.samples = json.load(f)
            
        print(f"Loaded {len(self.samples)} PREPROCESSED samples for {split} split. Target grid: {self.grid_size}x{self.grid_size}")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load pre-resized image directly
        img = cv2.imread(sample["image_path"])
        if img is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Instantly transform to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # CxHxW, 0-1

        # Standard ViT Normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Load pre-computed heatmap effortlessly
        heatmap_tensor = torch.load(sample["heatmap_path"], weights_only=True)
        
        # --- NEW LOGIC: Dynamic Resizing ---
        # If the loaded heatmap is 16x16 but we requested 32x32, interpolate it seamlessly.
        if heatmap_tensor.shape != (self.grid_size, self.grid_size):
            # F.interpolate expects (Batch, Channel, Height, Width), so we add and remove dummy dimensions
            heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0) 
            heatmap_tensor = F.interpolate(
                heatmap_tensor, 
                size=(self.grid_size, self.grid_size), 
                mode='bilinear', 
                align_corners=False
            )
            heatmap_tensor = heatmap_tensor.squeeze(0).squeeze(0)
        # -----------------------------------

        task_tensor = torch.tensor(sample["task_id"], dtype=torch.long)
        
        boxes = sample["boxes"]
        prefs = sample["prefs"]
        
        return {
            "image": img_tensor,
            "heatmap": heatmap_tensor,
            "task_id": task_tensor,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if len(boxes)>0 else torch.zeros((0,4)),
            "prefs": torch.tensor(prefs, dtype=torch.long) if len(prefs)>0 else torch.zeros((0,), dtype=torch.long),
            "image_path": sample["image_path"]
        }

def custom_collate(batch):
    """
    Custom collate_fn to successfully batch images and heatmaps
    while gracefully handling variable-length bounding boxes.
    """
    images = torch.stack([b["image"] for b in batch])
    heatmaps = torch.stack([b["heatmap"] for b in batch])
    task_ids = torch.stack([b["task_id"] for b in batch])
    
    # Keep variable length annotations as lists
    boxes = [b["boxes"] for b in batch]
    prefs = [b["prefs"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    
    return {
        "image": images,
        "heatmap": heatmaps,
        "task_id": task_ids,
        "boxes": boxes,
        "prefs": prefs,
        "image_paths": image_paths
    }