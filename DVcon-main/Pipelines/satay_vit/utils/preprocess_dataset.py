import os
import json
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random

def get_base_samples(data_root, split):
    """Parses JSONs exactly like the old DataLoader."""
    img_dir = os.path.join(data_root, split, "images")
    ann_dir = os.path.join(data_root, split, "annotations")
    
    samples = []
    
    for task_id in range(1, 15):
        ann_file = os.path.join(ann_dir, f"task_{task_id}_{split}.json")
        if not os.path.exists(ann_file):
            continue
            
        with open(ann_file, "r") as f:
            data = json.load(f)
            
        ann_dict = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in ann_dict:
                ann_dict[img_id] = []
            ann_dict[img_id].append(ann)
            
        for img_info in data["images"]:
            img_id = img_info["id"]
            img_path = os.path.join(img_dir, img_info["file_name"])
            
            img_anns = ann_dict.get(img_id, [])
            
            if not os.path.exists(img_path):
                continue
                
            samples.append({
                "task_id": task_id,
                "image_path": img_path,
                "orig_w": img_info["width"],
                "orig_h": img_info["height"],
                "annotations": img_anns,
                "img_id": img_id
            })
    return samples

def letterbox_image_and_boxes(img, boxes, new_shape=(640, 640)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    new_boxes = []
    for (x, y, w, h) in boxes:
        new_x = x * r + left
        new_y = y * r + top
        new_w = w * r
        new_h = h * r
        new_boxes.append([new_x, new_y, new_w, new_h])
        
    return img, new_boxes

def process_and_dump_dataset(data_root="e:/DVcon/DVcon/Data", output_root="e:/DVcon/DVcon/Data_Preprocessed_32", img_size=(640, 640), grid_size=(32, 32)):
    print(f"Beginning massive offline preprocessing to {output_root}")
    os.makedirs(output_root, exist_ok=True)
    
    for split in ["train", "test"]:
        print(f"Processing {split} split...")
        samples = get_base_samples(data_root, split)
        
        split_out_dir = os.path.join(output_root, split)
        img_out_dir = os.path.join(split_out_dir, "images")
        hm_out_dir = os.path.join(split_out_dir, "heatmaps")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(hm_out_dir, exist_ok=True)
        
        new_samples_index = []
        
        patch_w = img_size[0] / grid_size[0]
        patch_h = img_size[1] / grid_size[1]
        
        for sample in tqdm(samples):
            task_id = sample["task_id"]
            img_id = sample["img_id"]
            
            img = cv2.imread(sample["image_path"])
            if img is None:
                continue
                
            raw_boxes = []
            prefs = []
            for ann in sample["annotations"]:
                raw_boxes.append(ann["bbox"])
                prefs.append(ann["category_id"])
                
            # Resize
            img_lb, new_boxes = letterbox_image_and_boxes(img, raw_boxes, img_size)
            
            # Map visual heatmap
            heatmap = np.zeros(grid_size, dtype=np.float32)
            for box, pref in zip(new_boxes, prefs):
                if pref == 1:
                    x, y, w, h = box
                    start_x = max(0, int(x / patch_w))
                    end_x = min(grid_size[0] - 1, int((x + w) / patch_w))
                    start_y = max(0, int(y / patch_h))
                    end_y = min(grid_size[1] - 1, int((y + h) / patch_h))
                    heatmap[start_y:end_y+1, start_x:end_x+1] = 1.0
                    
            # Set Paths for Standard Output
            img_filename = f"{task_id}_{img_id}_norm.jpg"
            hm_filename = f"{task_id}_{img_id}.pt"
            
            img_save_path = os.path.join(img_out_dir, img_filename)
            hm_save_path = os.path.join(hm_out_dir, hm_filename)
            
            # Save Base
            cv2.imwrite(img_save_path, img_lb)
            if not os.path.exists(hm_save_path):
                torch.save(torch.from_numpy(heatmap), hm_save_path)
            
            new_samples_index.append({
                "task_id": task_id,
                "image_path": img_save_path,
                "heatmap_path": hm_save_path,
                "boxes": new_boxes,
                "prefs": prefs
            })
            
            # Augmentation Generation strictly for Train
            if split == "train":
                # Convert BGR CV2 to RGB Tensor format for torchvision augmentations
                img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                
                # Jitter
                img_tensor = TF.adjust_brightness(img_tensor, random.uniform(0.7, 1.3))
                img_tensor = TF.adjust_contrast(img_tensor, random.uniform(0.7, 1.3))
                img_tensor = TF.adjust_saturation(img_tensor, random.uniform(0.7, 1.3))
                
                # Convert back to uint8 properly bound
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img_aug_rgb = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_aug_bgr = cv2.cvtColor(img_aug_rgb, cv2.COLOR_RGB2BGR)
                
                img_aug_path = os.path.join(img_out_dir, f"{task_id}_{img_id}_aug.jpg")
                cv2.imwrite(img_aug_path, img_aug_bgr)
                
                new_samples_index.append({
                    "task_id": task_id,
                    "image_path": img_aug_path,
                    "heatmap_path": hm_save_path, # Keep identical heatmap! Super efficient
                    "boxes": new_boxes,
                    "prefs": prefs
                })
                
        # Save JSON index for rapid DataLoader reading
        index_path = os.path.join(split_out_dir, "samples.json")
        with open(index_path, "w") as f:
            json.dump(new_samples_index, f)
            
    print(f"Pre-Processing Successfully stored natively inside {output_root}")

if __name__ == "__main__":
    process_and_dump_dataset()
