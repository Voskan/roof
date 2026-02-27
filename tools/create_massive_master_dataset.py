import os
import cv2
import json
import base64
import zlib
import shutil
import numpy as np
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

"""
Generates the Massive Master Roof Dataset (>30k images) by merging:
1. DatasetNinja (Base64) -> Panels, Obstacles
2. YOLO Satellite (TXT) -> Panels, Obstacles
3. ROOF3D (COCO) -> Generic Roofs
4. RID (PNG Masks) -> Flat/Sloped Roofs

Expands data volume 4x via standard rigid rotations (No deformation):
- 0 deg, 90 deg, 180 deg, 270 deg.
"""

OUT_DIR = Path('/Users/voskan/Desktop/DeepRoof-2026/data/MassiveMasterDataset')
NINJA_DIR = Path('/Users/voskan/roofscope_data/semantic-segmentation-satellite-imagery-DatasetNinja')
RID_DIR = Path('/Users/voskan/roofscope_data/roof_information_dataset_2')
ROOF3D_DIR = Path('/Users/voskan/roofscope_data/ROOF3D')
YOLO_DIR = Path('/Users/voskan/roofscope_data/yolo_satellite')

def setup_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / 'images').mkdir(parents=True)
    (OUT_DIR / 'masks').mkdir(parents=True)

def apply_rotations(img, mask, base_name):
    # Original
    yield img, mask, f"{base_name}_rot0"
    
    # Rot 90
    yield cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE), f"{base_name}_rot90"
    
    # Rot 180
    yield cv2.rotate(img, cv2.ROTATE_180), cv2.rotate(mask, cv2.ROTATE_180), f"{base_name}_rot180"
    
    # Rot 270
    yield cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE), f"{base_name}_rot270"

def write_samples(samples):
    count = 0
    for img, mask, name in samples:
        if img is not None and mask is not None:
            cv2.imwrite(str(OUT_DIR / 'images' / f'{name}.jpg'), img)
            cv2.imwrite(str(OUT_DIR / 'masks' / f'{name}.png'), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            count += 1
    return count

# --- DATASET NINJA ---
def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        return mask[:, :, 3] > 0
    elif len(mask.shape) == 3:
        return mask[:, :, 0] > 0
    return mask > 0

def worker_ninja(ann_name, ann_dir, img_dir):
    try:
        img_name = ann_name.replace('.json', '')
        img_path = img_dir / img_name
        if not img_path.exists(): return 0
        
        with open(ann_dir / ann_name) as f:
            data = json.load(f)
            
        h, w = data['size']['height'], data['size']['width']
        unified_mask = np.zeros((h, w), dtype=np.uint8)
        
        objects = sorted(data.get('objects', []), key=lambda o: 10 if o.get('classTitle') == 'solar panels' else (5 if o.get('classTitle') in ['chimney', 'satellite antenna', 'window'] else 0))
        ninja_mapping = {'solar panels': 3, 'chimney': 4, 'satellite antenna': 4, 'window': 4, 'secondary structure': 4, 'property roof': 2}
        
        has_labels = False
        for obj in objects:
            title = obj.get('classTitle', '')
            if title in ninja_mapping and 'bitmap' in obj and 'data' in obj['bitmap']:
                orig_x, orig_y = obj['bitmap']['origin']
                bitmask = base64_2_mask(obj['bitmap']['data'])
                bh, bw = bitmask.shape
                y_end, x_end = min(orig_y + bh, h), min(orig_x + bw, w)
                bitmask_cropped = bitmask[0:(y_end-orig_y), 0:(x_end-orig_x)]
                unified_mask[orig_y:y_end, orig_x:x_end][bitmask_cropped > 0] = ninja_mapping[title]
                has_labels = True
                
        if has_labels:
            img = cv2.imread(str(img_path))
            return write_samples(apply_rotations(img, unified_mask, f"ninja_{img_name.split('.')[0]}"))
    except: ...
    return 0

def extract_ninja():
    print("Extracting DatasetNinja...")
    total = 0
    for split in ['train', 'val']:
        ann_dir, img_dir = NINJA_DIR / split / 'ann', NINJA_DIR / split / 'img'
        if not ann_dir.exists(): continue
        files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(tqdm(ex.map(lambda f: worker_ninja(f, ann_dir, img_dir), files), total=len(files), desc=f"Ninja {split}"))
        total += sum(results)
    return total

# --- ROOF3D ---
def extract_roof3d():
    print("Extracting ROOF3D (COCO format)...")
    ann_path = ROOF3D_DIR / 'train' / 'annotation_plane.json'
    img_dir = ROOF3D_DIR / 'train' / 'rgb'
    if not ann_path.exists(): return 0
    
    with open(ann_path) as f:
        coco = json.load(f)
        
    img_dict = {img['id']: {"file_name": img['file_name'], "h": img['height'], "w": img['width']} for img in coco['images']}
    ann_dict = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_dict: ann_dict[img_id] = []
        ann_dict[img_id].append(ann)
        
    def worker_roof3d(img_id):
        info = img_dict[img_id]
        img_path = img_dir / info['file_name']
        if not img_path.exists(): return 0
        img = cv2.imread(str(img_path))
        if img is None: return 0
        
        mask = np.zeros((info['h'], info['w']), dtype=np.uint8)
        for ann in ann_dict.get(img_id, []):
            seg_data = ann.get('segmentation', [])
            if isinstance(seg_data, dict): continue
            for seg in seg_data:
                if len(seg) < 6: continue
                poly = np.array(seg).reshape((int(len(seg)/2), 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 2) # Assume all ROOF3D buildings are general Roofs (2)
                
        return write_samples(apply_rotations(img, mask, f"roof3d_{Path(info['file_name']).stem}"))

    total = 0
    img_ids = list(img_dict.keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(tqdm(ex.map(worker_roof3d, img_ids), total=len(img_ids), desc="ROOF3D"))
    total += sum(results)
    return total

# --- RID ---
def extract_rid():
    print("Extracting RID...")
    img_dir, seg_dir = RID_DIR / 'images', RID_DIR / 'masks' / 'masks_segments'
    if not img_dir.exists() or not seg_dir.exists(): return 0
    
    def worker_rid(img_name):
        img_path = img_dir / img_name
        seg_path = seg_dir / img_name
        if not seg_path.exists():
            seg_path = seg_dir / img_name.replace('.tif', '.png').replace('.jpg','.png')
            if not seg_path.exists(): return 0
            
        img = cv2.imread(str(img_path))
        seg = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
        if img is None or seg is None: return 0
        
        mask = np.zeros(seg.shape[:2], dtype=np.uint8)
        mask[seg > 0] = 2 # General Sloped/Flat assumption for RID segments
        return write_samples(apply_rotations(img, mask, f"rid_{Path(img_name).stem}"))

    files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.tif', '.jpg'))]
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(tqdm(ex.map(worker_rid, files), total=len(files), desc="RID"))
    total += sum(results)
    return total

# --- YOLO SATELLITE ---
def extract_yolo():
    print("Extracting YOLO Satellite...")
    img_dir, lbl_dir = YOLO_DIR / 'images' / 'train', YOLO_DIR / 'labels' / 'train'
    if not img_dir.exists(): return 0
    yolo_mapping = {14: 3, 2: 4, 12: 4, 24: 4, 10: 2} # 14: panel, 2: chimney, 12: antenna, 24: window, 10: roof
    
    def worker_yolo(img_name):
        img_path = img_dir / img_name
        lbl_path = lbl_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        if not lbl_path.exists(): return 0
        
        img = cv2.imread(str(img_path))
        if img is None: return 0
        h, w = img.shape[:2]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(parts[0])
                if cls_id in yolo_mapping:
                    if len(parts) > 5: # Polygon
                        pts = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                        pts[:, 0] *= w; pts[:, 1] *= h
                        cv2.fillPoly(mask, [pts.astype(np.int32)], yolo_mapping[cls_id])
                    else: # Bbox
                        cx, cy, bw, bh = float(parts[1])*w, float(parts[2])*h, float(parts[3])*w, float(parts[4])*h
                        x1, y1 = int(cx - bw/2), int(cy - bh/2)
                        x2, y2 = int(cx + bw/2), int(cy + bh/2)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), yolo_mapping[cls_id], -1)
                        
        return write_samples(apply_rotations(img, mask, f"yolo_{Path(img_name).stem}"))

    files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(tqdm(ex.map(worker_yolo, files), total=len(files), desc="YOLO"))
    total += sum(results)
    return total

if __name__ == '__main__':
    setup_dirs()
    t1 = extract_ninja()
    t2 = extract_yolo()
    t3 = extract_roof3d()
    t4 = extract_rid()
    
    total_imgs = t1 + t2 + t3 + t4
    
    print("Creating Training Split...")
    files = [f.split('.')[0] for f in os.listdir(OUT_DIR / 'masks') if f.endswith('.png')]
    with open(OUT_DIR / 'train.txt', 'w') as f:
        for fname in files: f.write(f"{fname}\n")
            
    print(f"=====================================")
    print(f"MASSIVE DATASET GENERATION COMPLETE.")
    print(f"Total Files Generated: {total_imgs} images")
    print(f"=====================================")
