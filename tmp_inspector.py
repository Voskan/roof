import os
import cv2
import json
import numpy as np

# Analyze RID (roof_information_dataset_2)
rid_segments_dir = '/Users/voskan/roofscope_data/roof_information_dataset_2/masks/masks_segments'
rid_ss_dir = '/Users/voskan/roofscope_data/roof_information_dataset_2/masks/masks_superstructures'

print("--- RID ---")
if os.path.exists(rid_segments_dir):
    files = [f for f in os.listdir(rid_segments_dir) if f.endswith('.png') or f.endswith('.tif')]
    if files:
        f = os.path.join(rid_segments_dir, files[0])
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        print(f"Segment mask {files[0]} shape {img.shape} unique {np.unique(img)}")

if os.path.exists(rid_ss_dir):
    files = [f for f in os.listdir(rid_ss_dir) if f.endswith('.png') or f.endswith('.tif')]
    if files:
        f = os.path.join(rid_ss_dir, files[0])
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        print(f"Superstructure mask {files[0]} shape {img.shape} unique {np.unique(img)}")

# Analyze DatasetNinja
ninja_ann_dir = '/Users/voskan/roofscope_data/semantic-segmentation-satellite-imagery-DatasetNinja/train/ann'
print("\n--- DatasetNinja ---")
if os.path.exists(ninja_ann_dir):
    files = [f for f in os.listdir(ninja_ann_dir) if f.endswith('.json')]
    if files:
        with open(os.path.join(ninja_ann_dir, files[0])) as f:
            data = json.load(f)
            print(f"Ninja Ann file keys: {data.keys()}")
            print(f"Example objects: {[o.get('classTitle') for o in data.get('objects', [])[:5]]}")
