import json
import os
import glob
import cv2
import numpy as np

print("--- Checking iSAID (COCO format expected) ---")
isaid_json = '/Users/voskan/roofscope_data/iSAID/train/instancesonly_filtered_train.json'
if os.path.exists(isaid_json):
    with open(isaid_json, 'r') as f:
        data = json.load(f)
        if 'categories' in data:
            for cat in data['categories']:
                print(f"iSAID category: {cat}")
else:
    print(f"{isaid_json} not found. Trying to find any json in iSAID...")
    jsons = glob.glob('/Users/voskan/roofscope_data/iSAID/**/*.json', recursive=True)
    for j in jsons[:2]:
        print(f"Found: {j}")
        with open(j, 'r') as f:
            d = json.load(f)
            print(f"Keys: {d.keys()}")
            if 'categories' in d:
                print(f"Categories: {d['categories']}")

print("\n--- Checking SODwS-V1 (Images/Labels/Masks) ---")
sodws_masks_dir = '/Users/voskan/roofscope_data/SODwS-V1/Location_A/masks'
if os.path.exists(sodws_masks_dir):
    masks = os.listdir(sodws_masks_dir)
    print(f"Found {len(masks)} masks in Location_A")
    if masks:
        mask_path = os.path.join(sodws_masks_dir, masks[0])
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        print(f"Sample mask: {masks[0]}, Shape: {mask.shape}, Unique values: {np.unique(mask)}")

print("\n--- Checking yolo_satellite ---")
yolo_yaml = '/Users/voskan/roofscope_data/yolo_satellite/data.yaml'
if os.path.exists(yolo_yaml):
    with open(yolo_yaml, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'names:' in line or '-' in line:
                print(line.strip())
