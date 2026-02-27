"""
Exhaustive analysis of MassiveMasterDataset.
Checks: file integrity, shapes, class distributions, orphaned files,
per-source breakdown, class co-occurrence, empty masks, etc.
"""
import os
import cv2
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path('/Users/voskan/Desktop/DeepRoof-2026/data/MassiveMasterDataset')

def run():
    img_dir = DATA_DIR / 'images'
    mask_dir = DATA_DIR / 'masks'
    
    img_files = set(f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.tif')))
    mask_files = set(f for f in os.listdir(mask_dir) if f.endswith('.png'))
    
    # Strip extensions to match
    img_stems = {Path(f).stem for f in img_files}
    mask_stems = {Path(f).stem for f in mask_files}
    
    orphan_imgs = img_stems - mask_stems
    orphan_masks = mask_stems - img_stems
    matched = img_stems & mask_stems
    
    print(f"=== FILE INTEGRITY ===")
    print(f"Total image files: {len(img_files)}")
    print(f"Total mask files:  {len(mask_files)}")
    print(f"Matched pairs:     {len(matched)}")
    print(f"Orphan images (no mask): {len(orphan_imgs)}")
    print(f"Orphan masks (no image): {len(orphan_masks)}")
    
    # train.txt check
    train_txt = DATA_DIR / 'train.txt'
    if train_txt.exists():
        with open(train_txt) as f:
            train_lines = [l.strip() for l in f if l.strip()]
        print(f"train.txt entries: {len(train_lines)}")
        missing_from_disk = [t for t in train_lines if t not in mask_stems]
        print(f"train.txt entries missing from disk: {len(missing_from_disk)}")
    
    # Per-source breakdown
    sources = defaultdict(int)
    for stem in matched:
        prefix = stem.split('_')[0]
        sources[prefix] += 1
    print(f"\n=== PER-SOURCE BREAKDOWN ===")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} pairs")
    
    # Pixel analysis (full scan)
    print(f"\n=== FULL PIXEL ANALYSIS (scanning ALL {len(matched)} masks) ===")
    class_pixel_counts = Counter()
    class_image_counts = Counter()  # How many images contain each class
    total_pixels = 0
    empty_masks = 0
    shape_counter = Counter()
    shape_mismatches = 0
    corrupt_files = 0
    invalid_class_files = []
    
    sorted_matched = sorted(matched)
    batch_size = 500
    for batch_start in range(0, len(sorted_matched), batch_size):
        batch = sorted_matched[batch_start:batch_start+batch_size]
        if batch_start % 5000 == 0:
            print(f"  ...processing {batch_start}/{len(sorted_matched)}")
        for stem in batch:
            # Find actual image file
            img_path = None
            for ext in ['.jpg', '.png', '.tif']:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            mask_path = mask_dir / f"{stem}.png"
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                corrupt_files += 1
                continue
                
            if img_path:
                img = cv2.imread(str(img_path))
                if img is not None:
                    shape_counter[img.shape[:2]] += 1
                    if img.shape[:2] != mask.shape[:2]:
                        shape_mismatches += 1
            
            unique_vals = np.unique(mask)
            total_pixels += mask.size
            
            invalid = [v for v in unique_vals if v not in [0, 1, 2, 3, 4]]
            if invalid:
                invalid_class_files.append((stem, invalid))
            
            for v in unique_vals:
                class_image_counts[v] += 1
            
            u, c = np.unique(mask, return_counts=True)
            for val, cnt in zip(u, c):
                class_pixel_counts[val] += cnt
            
            if mask.max() == 0:
                empty_masks += 1
    
    print(f"\n=== RESULTS ===")
    print(f"Corrupt/unreadable masks: {corrupt_files}")
    print(f"Shape mismatches (img vs mask): {shape_mismatches}")
    print(f"Empty masks (all zeros): {empty_masks}")
    print(f"Files with invalid class IDs: {len(invalid_class_files)}")
    if invalid_class_files[:5]:
        for stem, vals in invalid_class_files[:5]:
            print(f"  {stem}: invalid IDs = {vals}")
    
    print(f"\nImage shape distribution:")
    for shape, cnt in shape_counter.most_common(10):
        print(f"  {shape}: {cnt} images")
    
    CLASS_NAMES = {0: 'Background', 1: 'Flat Roof', 2: 'Sloped Roof', 3: 'Solar Panel', 4: 'Roof Obstacle'}
    print(f"\n=== CLASS PIXEL DISTRIBUTION ===")
    for cls_id in sorted(class_pixel_counts.keys()):
        name = CLASS_NAMES.get(cls_id, f'UNKNOWN_{cls_id}')
        pixels = class_pixel_counts[cls_id]
        pct = (pixels / total_pixels) * 100
        img_count = class_image_counts[cls_id]
        img_pct = (img_count / len(matched)) * 100
        print(f"  Class {cls_id} ({name}): {pixels:>15,} px ({pct:>7.4f}%)  |  present in {img_count:>6} images ({img_pct:.1f}%)")
    
    print(f"\nTotal pixels analyzed: {total_pixels:,}")
    
    # Compute ideal class weights based on inverse frequency
    print(f"\n=== RECOMMENDED CLASS WEIGHTS ===")
    fg_classes = {k: v for k, v in class_pixel_counts.items() if k > 0}
    if fg_classes:
        max_fg = max(fg_classes.values())
        print(f"  Based on inverse pixel frequency (normalized to max foreground class):")
        weights = [1.0]  # BG weight
        for cls_id in range(1, 5):
            if cls_id in class_pixel_counts and class_pixel_counts[cls_id] > 0:
                w = max_fg / class_pixel_counts[cls_id]
                w = min(w, 100.0)  # Cap at 100
                weights.append(round(w, 1))
                print(f"    Class {cls_id} ({CLASS_NAMES[cls_id]}): weight = {w:.1f}")
            else:
                weights.append(1.0)
                print(f"    Class {cls_id} ({CLASS_NAMES[cls_id]}): NOT FOUND, weight = 1.0")
        weights.append(0.1)  # no-object
        print(f"  Recommended class_weight list: {weights}")

if __name__ == '__main__':
    run()
