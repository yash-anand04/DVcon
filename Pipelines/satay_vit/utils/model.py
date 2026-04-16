import torch
import torch.nn as nn
import torch.nn.functional as F

class MEViTReasoner(nn.Module):
    """
    Memory-Efficient Vision Transformer (ME-ViT) Reasoner.
    Takes an input image and a Task ID/Embedding, and generates a Task Relevance Heatmap.
    """
    def __init__(self, img_size=640, patch_size=40, embed_dim=512, num_tasks=14):
        super(MEViTReasoner, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 640 // 40 = 16
        self.num_patches = self.grid_size * self.grid_size # 16x16 = 256
        self.embed_dim = embed_dim
        
        # 1. Patch Embedding (Linear projection of patches)
        # 3 channels -> embed_dim
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings to retain spatial info
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Task Embeddings: simple lookup table for the 14 tasks
        # In a real multimodal model, this would be a frozen CLIP text encoder
        self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim) # +1 to 1-index tasks easily
        
        # 2. Vision Transformer Blocks (Simplified for Edge)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 2, 
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Cross Attention: Image Patches (Q) vs Task Embedding (K, V)
        # For simplicity, we can do a simple dot-product attention
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, task_id):
        """
        x: (B, C, H, W) where H=W=640
        task_id: (B,)
        Returns:
            heatmap: (B, 16, 16) values between 0 and 1
        """
        B = x.size(0)
        
        # --- Image Encoding ---
        # (B, 512, 16, 16)
        patches = self.patch_embed(x) 
        # (B, 512, 256) -> (B, 256, 512)
        patches = patches.flatten(2).transpose(1, 2) 
        
        # Add positional embedding
        patches = patches + self.pos_embed
        
        # Self-attention among patches
        patches = self.transformer(patches)
        
        # --- Task Encoding ---
        # (B, 512) -> (B, 1, 512)
        task_emb = self.task_embedding(task_id).unsqueeze(1) 
        
        # --- Cross Attention (Reasoning) ---
        # Query: patches, Key: Task, Value is not strictly needed if we just want scores,
        # but let's compute similarity scores directly.
        Q = self.q_proj(patches) # (B, 256, 512)
        K = self.k_proj(task_emb) # (B, 1, 512)
        
        # Dot product: (B, 256, 512) @ (B, 512, 1) -> (B, 256, 1)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)
        
        # Sigmoid to normalize relevance to [0, 1] per patch
        heatmap_flat = torch.sigmoid(scores.squeeze(2)) # (B, 256)
        
        # Reshape to (B, 16, 16)
        heatmap = heatmap_flat.view(B, self.grid_size, self.grid_size)
        
        return heatmap

def get_yolo_localizer():
    # Load the official ultralytics model
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt') # downloads the pre-trained nano model automatically
    return model

if __name__ == "__main__":
    model = MEViTReasoner()
    dummy_img = torch.randn(2, 3, 640, 640)
    dummy_task = torch.tensor([1, 14])
    out = model(dummy_img, dummy_task)
    print(f"ME-ViT Output Shape: {out.shape}") # Expected: [2, 16, 16]
