import os
import cv2
import numpy as np
from collections import Counter
from tqdm import tqdm

def analyze_dataset(data_dir, sample_ratio=0.1):
    mask_dir = os.path.join(data_dir, 'masks')
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    num_samples = max(1, int(len(mask_files) * sample_ratio))
    sampled_files = np.random.choice(mask_files, num_samples, replace=False)
    
    print(f"Total masks found: {len(mask_files)}. Analyzing a random {sample_ratio*100}% sample ({num_samples} images)...")
    
    total_pixels = 0
    class_counts = Counter()
    invalid_classes = set()
    shape_mismatches = 0
    
    for f in tqdm(sampled_files, desc="Analyzing masks"):
        img_path = os.path.join(data_dir, 'images', f.replace('.png', '.jpg'))
        mask_path = os.path.join(mask_dir, f)
        
        if not os.path.exists(img_path):
            img_path = img_path.replace('.jpg', '.png')
            
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if img is not None and mask is not None:
                if img.shape[:2] != mask.shape[:2]:
                    shape_mismatches += 1
                
                total_pixels += mask.size
                unique, counts = np.unique(mask, return_counts=True)
                for u, c in zip(unique, counts):
                    class_counts[u] += c
                    if u not in [0, 1, 2, 3, 4]:
                        invalid_classes.add(u)
                        
    print("\n--- MassiveMasterDataset Analysis Report ---")
    print(f"Shape Mismatches (Img vs Mask): {shape_mismatches}")
    print(f"Invalid Classes Found (should be empty): {invalid_classes}")
    print("Class Pixel Distribution:")
    classes = {0: 'Background', 1: 'Flat Roof', 2: 'Sloped Roof', 3: 'Solar Panel', 4: 'Obstacle'}
    for k, v in sorted(class_counts.items()):
        name = classes.get(k, f"Invalid_{k}")
        pct = (v / total_pixels) * 100
        print(f"  - Class {k} ({name}): {v} pixels ({pct:.4f}%)")

if __name__ == '__main__':
    analyze_dataset('/Users/voskan/Desktop/DeepRoof-2026/data/MassiveMasterDataset', sample_ratio=0.05)
