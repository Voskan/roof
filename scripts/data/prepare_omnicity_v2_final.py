import os
import json
import zipfile
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import rasterio
import shutil
from collections import defaultdict

def extract_zip(zip_path: Path, extract_to: Path):
    if not zip_path.exists(): return False
    print(f"Extracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def calculate_surface_normal(height_map: np.ndarray) -> np.ndarray:
    dz_dx = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(height_map)], axis=-1)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    return normals / np.maximum(norm, 1e-8)

def main():
    parser = argparse.ArgumentParser(description='Final Comprehensive OmniCity Prep')
    parser.add_argument('--data-root', type=str, required=True, help='Path containing datasets/')
    parser.add_argument('--output-dir', type=str, default='data/OmniCity', help='Output root')
    parser.add_argument('--skip-extract', action='store_true', help='Skip unzipping archives')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)
    
    # Paths Discovery (Based on Granular Report)
    datasets_dir = data_root / 'datasets' if (data_root / 'datasets').exists() else data_root
    
    raw_root = datasets_dir / 'OmniCity' / 'OpenDataLab___OmniCity' / 'raw'
    if not raw_root.exists():
        # Fallback recursive search for OmniCity-dataset
        for found in datasets_dir.rglob('OmniCity-dataset'):
            if found.is_dir():
                raw_root = found.parent
                break
                
    if not raw_root.exists():
        print(f"Error: Could not find raw OmniCity data in {datasets_dir}")
        return

    print(f"Generating training data into: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'images').mkdir(exist_ok=True)
    (output_root / 'masks').mkdir(exist_ok=True)
    (output_root / 'normals').mkdir(exist_ok=True)

    # 1. Unzip Everything Needed
    # Satellite Level
    sat_level = raw_root / 'OmniCity-dataset' / 'satellite-level'
    
    if not args.skip_extract:
        view1_zip = sat_level / 'image-satellite' / 'satellite-image-view1.zip'
        view1_extract = sat_level / 'image-satellite' / 'view1'
        if view1_zip.exists() and not view1_extract.exists():
            extract_zip(view1_zip, view1_extract)

        height_zip_train = sat_level / 'annotation-height' / 'annotation-height-train.zip'
        height_train_dir = sat_level / 'annotation-height' / 'train'
        if height_zip_train.exists() and not height_train_dir.exists():
            extract_zip(height_zip_train, height_train_dir)

        height_zip_test = sat_level / 'annotation-height' / 'annotation-height-test.zip'
        height_test_dir = sat_level / 'annotation-height' / 'test'
        if height_zip_test.exists() and not height_test_dir.exists():
            extract_zip(height_zip_test, height_test_dir)
    else:
        print("Skipping extraction as requested.")

    # 2. Map Files
    # We use View 1 as primary images
    view1_extract = sat_level / 'image-satellite' / 'view1'
    height_train_dir = sat_level / 'annotation-height' / 'train'
    
    view1_img_dir = view1_extract / 'satellite-image-view1' / 'view1-train'
    if not view1_img_dir.exists():
        # Fallback for different zip structures
        for found in view1_extract.rglob('view1-train'):
             if found.is_dir(): view1_img_dir = found; break

    height_maps_dir = height_train_dir / 'annotation-height-train'
    if not height_maps_dir.exists():
        for found in height_train_dir.rglob('annotation-height-train'):
             if found.is_dir(): height_maps_dir = found; break

    # 3. Load Annotations
    ann_json_path = sat_level / 'annotation-seg' / 'annotation-seg-view1-train.json'
    if not ann_json_path.exists():
        print(f"Warning: Annotation JSON not found at {ann_json_path}")
        return

    with open(ann_json_path, 'r') as f:
        coco_data = json.load(f)

    id_to_ann = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        id_to_ann[ann['image_id']].append(ann)

    # 4. Process Union
    processed_ids = []
    print("Processing images into flat structure...")
    for img_info in tqdm(coco_data.get('images', [])):
        img_id = img_info['id']
        file_name = img_info['file_name']
        stem = Path(file_name).stem
        
        src_img = view1_img_dir / file_name
        src_height = height_maps_dir / (stem + '.tif')
        
        if not src_img.exists() or not src_height.exists():
            continue
            
        # Copy Image
        shutil.copy2(src_img, output_root / 'images' / (stem + '.jpg'))
        
        # Process Normals
        with rasterio.open(src_height) as src:
            hmap = src.read(1)
        normals = calculate_surface_normal(hmap)
        np.save(output_root / 'normals' / (stem + '.npy'), normals)
        
        # Process Mask
        mask = np.zeros(hmap.shape, dtype=np.uint16)
        for idx, ann in enumerate(id_to_ann[img_id], 1):
            if 'segmentation' in ann and ann['segmentation']:
                for poly in ann['segmentation']:
                    if isinstance(poly, list):
                        p = np.array(poly).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [p], idx)
        cv2.imwrite(str(output_root / 'masks' / (stem + '.png')), mask)
        
        processed_ids.append(stem)

    # 5. Save Split
    with open(output_root / 'train.txt', 'w') as f:
        for p_id in processed_ids:
            f.write(f"{p_id}\n")
            
    # Simple val split (last 10%)
    val_count = len(processed_ids) // 10
    with open(output_root / 'val.txt', 'w') as f:
        for p_id in processed_ids[-val_count:]:
            f.write(f"{p_id}\n")

    print(f"\nSuccessfully prepared {len(processed_ids)} training samples.")
    print(f"Training lists generated at {output_root}/train.txt")

if __name__ == "__main__":
    main()
