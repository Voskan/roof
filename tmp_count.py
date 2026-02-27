import os
import json
import glob

def count_roof3d():
    ann_file = '/Users/voskan/roofscope_data/ROOF3D/train/annotation_plane.json'
    if os.path.exists(ann_file):
        with open(ann_file, 'r') as f:
            data = json.load(f)
            imgs = len(data.get('images', []))
            anns = len(data.get('annotations', []))
            print(f"ROOF3D Train images: {imgs}, annotations: {anns}")

def count_yolo():
    yolo_dir = '/Users/voskan/roofscope_data/yolo_satellite/images/train'
    if os.path.exists(yolo_dir):
        imgs = len(glob.glob(os.path.join(yolo_dir, '*.*')))
        print(f"yolo_satellite Train images: {imgs}")
        
def count_sodws():
    sodws_dir = '/Users/voskan/roofscope_data/SODwS-V1'
    if os.path.exists(sodws_dir):
        imgs = len(glob.glob(os.path.join(sodws_dir, '**', 'images', '*.*'), recursive=True))
        print(f"SODwS-V1 total images: {imgs}")

print("--- Data Volume Analysis ---")
count_roof3d()
count_yolo()
count_sodws()
