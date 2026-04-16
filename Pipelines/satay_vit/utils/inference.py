import torch

def fuse_yolo_and_vit(yolo_boxes, yolo_confs, vit_heatmap, img_size=(640, 640)):
    """
    Fuses the outputs of the YOLO localizer and ViT reasoner.
    yolo_boxes: Tensor (N, 4) in [x_min, y_min, w, h] format relative to 640x640
    yolo_confs: Tensor (N,) of YOLO objectness/class confidence
    vit_heatmap: Tensor (16, 16) with values in [0, 1] representing task relevance
    
    Returns:
        best_box_idx: standard integer index of the most appropriate box
        fused_scores: Tensor (N,)
    """
    if len(yolo_boxes) == 0:
        return None, []
        
    grid_h, grid_w = vit_heatmap.shape
    patch_w = img_size[0] / grid_w
    patch_h = img_size[1] / grid_h
    
    fused_scores = []
    
    for i, box in enumerate(yolo_boxes):
        x, y, w, h = box
        
        # Calculate center of the bounding box
        cx = x + w / 2.0
        cy = y + h / 2.0
        
        # Map to heatmap grid indices
        grid_x = int(cx / patch_w)
        grid_y = int(cy / patch_h)
        
        # Clamp to grid boundaries just in case
        grid_x = max(0, min(grid_w - 1, grid_x))
        grid_y = max(0, min(grid_h - 1, grid_y))
        
        relevance_score = vit_heatmap[grid_y, grid_x]
        
        # Semantic Fusion
        fused = yolo_confs[i] * relevance_score
        fused_scores.append(fused)
        
    fused_scores = torch.tensor(fused_scores)
    best_box_idx = torch.argmax(fused_scores).item()
    
    return best_box_idx, fused_scores

def draw_inference(img, yolo_boxes, yolo_confs, fused_scores, best_idx, task_name="Task"):
    """ Utility to visualize the result. Not strictly needed for FPGA, but good for PyTorch eval. """
    import cv2
    import numpy as np
    
    img_draw = img.copy()
    
    for i, box in enumerate(yolo_boxes):
        x, y, w, h = [int(v) for v in box]
        conf = float(yolo_confs[i])
        fused = float(fused_scores[i])
        
        if i == best_idx:
            color = (0, 255, 0) # Green for BEST
            thick = 3
        else:
            color = (0, 0, 255) # Red for ignored
            thick = 1
            
        cv2.rectangle(img_draw, (x, y), (x+w, y+h), color, thick)
        
        label = f"{fused:.2f} (Y:{conf:.2f})"
        cv2.putText(img_draw, label, (x, max(0, y-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thick)
                    
    cv2.putText(img_draw, f"Goal: {task_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
    return img_draw
