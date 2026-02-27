import os
import cv2
import json
import base64
import zlib
import io
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Target unified mapping:
# 0: Background
# 1: Flat Roof
# 2: Sloped Roof
# 3: Solar Panel
# 4: Roof Obstacle (chimney, antenna, window, dormer)

OUT_DIR = Path('/Users/voskan/Desktop/DeepRoof-2026/data/MasterRoofDataset')
NINJA_DIR = Path('/Users/voskan/roofscope_data/semantic-segmentation-satellite-imagery-DatasetNinja')
RID_DIR = Path('/Users/voskan/roofscope_data/roof_information_dataset_2')

def setup_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / 'images').mkdir(parents=True)
    (OUT_DIR / 'masks').mkdir(parents=True)

def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        return mask[:, :, 3] > 0
    elif len(mask.shape) == 3:
        return mask[:, :, 0] > 0
    return mask > 0

def process_dataset_ninja():
    print("Processing Dataset Ninja (Supervisely format)...")
    splits = ['train', 'val']
    
    ninja_mapping = {
        'solar panels': 3,
        'chimney': 4,
        'satellite antenna': 4,
        'window': 4,
        'secondary structure': 4,
        'property roof': 2 # Default to sloped/generic roof
    }
    
    count = 0
    for split in splits:
        ann_dir = NINJA_DIR / split / 'ann'
        img_dir = NINJA_DIR / split / 'img'
        
        if not ann_dir.exists(): continue
        
        for ann_name in tqdm(os.listdir(ann_dir), desc=f"Ninja {split}"):
            if not ann_name.endswith('.json'): continue
            
            img_name = ann_name.replace('.json', '')
            img_path = img_dir / img_name
            if not img_path.exists(): continue
            
            with open(ann_dir / ann_name) as f:
                data = json.load(f)
                
            h, w = data['size']['height'], data['size']['width']
            unified_mask = np.zeros((h, w), dtype=np.uint8)
            
            objects = data.get('objects', [])
            # Sort objects so panels and obstacles are drawn LAST (on top)
            def sort_key(obj):
                title = obj.get('classTitle', '')
                if title == 'solar panels': return 10
                if title in ['chimney', 'satellite antenna', 'window', 'secondary structure']: return 5
                return 0
            
            objects = sorted(objects, key=sort_key)
            
            mask_has_labels = False
            for obj in objects:
                title = obj.get('classTitle', '')
                if title in ninja_mapping:
                    class_id = ninja_mapping[title]
                    if 'bitmap' in obj and 'data' in obj['bitmap']:
                        origin = obj['bitmap']['origin']
                        x, y = origin[0], origin[1]
                        bitmask = base64_2_mask(obj['bitmap']['data'])
                        
                        bh, bw = bitmask.shape
                        y_end = min(y + bh, h)
                        x_end = min(x + bw, w)
                        bitmask = bitmask[0:(y_end-y), 0:(x_end-x)]
                        
                        # Apply to unified mask
                        valid_pixels = bitmask > 0
                        unified_mask[y:y_end, x:x_end][valid_pixels] = class_id
                        mask_has_labels = True
                        
            if mask_has_labels:
                out_img_name = f"ninja_{img_name}"
                out_mask_name = f"ninja_{img_name.split('.')[0]}.png" # Change to png
                
                shutil.copy(img_path, OUT_DIR / 'images' / out_img_name)
                cv2.imwrite(str(OUT_DIR / 'masks' / out_mask_name), unified_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
                count += 1
    print(f"Ninja Processing Complete! Migrated {count} valid images.")

def process_rid():
    print("Processing Roof Information Dataset 2...")
    img_dir = RID_DIR / 'images'
    seg_dir = RID_DIR / 'masks' / 'masks_segments'
    
    if not img_dir.exists() or not seg_dir.exists():
        print("RID images or masks not found.")
        return
        
    count = 0
    for img_name in tqdm(os.listdir(img_dir), desc="RID Files"):
        if not img_name.endswith('.png') and not img_name.endswith('.tif') and not img_name.endswith('.jpg'): continue
        
        img_path = img_dir / img_name
        seg_path = seg_dir / img_name
        
        if not seg_path.exists():
            # sometimes extension differs, try png
            seg_path = seg_dir / img_name.replace('.tif', '.png').replace('.jpg','.png')
            if not seg_path.exists():
                continue
                
        # Load mask
        seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
        if seg_mask is None: continue
        
        unified_mask = np.zeros_like(seg_mask, dtype=np.uint8)
        
        # RID logic: 
        # Typically values > 0 are roof segments. We will map to 2 (Sloped) for generic RID segments.
        # If there are flat roofs defined as specific IDs, they would be 1. We will map all roof pixels > 0 to 2 for now,
        # unless we find a clear distinction.
        unified_mask[seg_mask > 0] = 2 
        
        out_img_name = f"rid_{img_name}"
        out_mask_name = f"rid_{img_name.split('.')[0]}.png"
        
        shutil.copy(img_path, OUT_DIR / 'images' / out_img_name)
        cv2.imwrite(str(OUT_DIR / 'masks' / out_mask_name), unified_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        count += 1
        
    print(f"RID Processing Complete! Migrated {count} images.")

def create_train_split():
    files = [f.split('.')[0] for f in os.listdir(OUT_DIR / 'masks') if f.endswith('.png')]
    with open(OUT_DIR / 'train.txt', 'w') as f:
        for fname in files:
            f.write(f"{fname}\n")
    print(f"Generated train.txt with {len(files)} lines.")

if __name__ == '__main__':
    setup_dirs()
    process_dataset_ninja()
    process_rid()
    create_train_split()
    print("FINISHED MASTER MERGE.")
