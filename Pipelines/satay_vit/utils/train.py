import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import COCOTasksDataset, custom_collate
from model import MEViTReasoner

def train_satay_vit(data_root="e:/DVcon/DVcon/Data_Preprocessed", epochs=10, batch_size=16, lr=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Training on {device}")
    
    # 1. Load Data
    train_dataset = COCOTasksDataset(data_root, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    
    val_dataset = COCOTasksDataset(data_root, split="test") # COCO-Tasks test set acts as validation
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate)
    
    # 2. Model, Loss, Optimizer
    model = MEViTReasoner().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy for heatmaps
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) # Standard for ViT
    
    # 3. Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            imgs = batch["image"].to(device)
            tasks = batch["task_id"].to(device)
            gt_heatmaps = batch["heatmap"].to(device)
            
            optimizer.zero_grad()
            
            pred_heatmaps = model(imgs, tasks)
            
            # Bx16x16 against Bx16x16
            loss = criterion(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # 4. Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                imgs = batch["image"].to(device)
                tasks = batch["task_id"].to(device)
                gt_heatmaps = batch["heatmap"].to(device)
                
                pred_heatmaps = model(imgs, tasks)
                loss = criterion(pred_heatmaps, gt_heatmaps)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/mevit_best.pt")
            print("Saved new best model!")

if __name__ == "__main__":
    # If standard torch/CUDA works perfectly, this starts right away.
    train_satay_vit()
