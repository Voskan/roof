import os
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import rasterio
from typing import List, Dict, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Process OmniCity Dataset for DeepRoof-2026')
    parser.add_argument('--data-root', type=str, required=True, help='Path to OmniCity dataset root')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for processed data')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset split to process')
    return parser.parse_args()

def calculate_surface_normal(height_map: np.ndarray) -> np.ndarray:
    """
    Calculate surface normal vectors from a height map using gradients.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        
    Returns:
        np.ndarray: 3D array of normal vectors (H, W, 3) with components (nx, ny, nz).
                    Normals are normalized to unit length.
    """
    # Compute gradients using Sobel operators
    dz_dx = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)
    
    # The normal vector is (-dz/dx, -dz/dy, 1)
    # We negate the gradients because the normal points "up" from the surface
    normal_x = -dz_dx
    normal_y = -dz_dy
    normal_z = np.ones_like(height_map)
    
    # Stack components
    normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
    
    # Normalize vectors
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    # Avoid division by zero
    norm = np.maximum(norm, 1e-8)
    normals_normalized = normals / norm
    
    return normals_normalized

def process_single_image(
    image_id: str,
    annotation: Dict,
    data_root: Path,
    output_dir: Path
):
    """
    Process a single image: load height map, generate instance masks, calculate normals.
    """
    # Paths (Adjust based on actual OmniCity structure)
    # Assuming standard structure: images/{id}.tif, height/{id}.tif
    height_path = data_root / 'height' / f"{image_id}.tif"
    
    if not height_path.exists():
        # Fallback or skip if not found
        # print(f"Warning: Height map not found: {height_path}")
        return

    # Load Height Map
    with rasterio.open(height_path) as src:
        height_map = src.read(1)
        meta = src.meta.copy()

    H, W = height_map.shape
    
    # Initialize buffers
    instance_mask = np.zeros((H, W), dtype=np.uint16)
    
    # Process annotations (assuming COCO-like format or OmniCity specific)
    # OmniCity usually provides building instance IDs. 
    # For this script, we assume 'annotation' contains a list of polygons/masks for roof facets
    
    # Note: Logic below depends on exact JSON structure of OmniCity
    # Here we assume a list of objects with segmentation polygons
    
    for idx, obj in enumerate(annotation.get('objects', []), start=1):
        # Draw polygon
        poly = np.array(obj['polygon']).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(instance_mask, [poly], idx)
        
    # Calculate Normals
    normals = calculate_surface_normal(height_map)
    
    # Save outputs
    # 1. Save Instance Mask
    mask_out_path = output_dir / 'masks' / f"{image_id}.png"
    cv2.imwrite(str(mask_out_path), instance_mask)
    
    # 2. Save Normals (Optional visualization, mapped to 0-255)
    # n coords are [-1, 1], map to [0, 255]
    normals_vis = ((normals + 1) * 127.5).astype(np.uint8)
    # OpenCV uses BGR, interpret normals as RGB (x, y, z) -> (b, g, r) if needed
    # Usually normals are saved as float .npy or .tif for training
    normal_out_path = output_dir / 'normals' / f"{image_id}.npy"
    np.save(normal_out_path, normals)
    
    # Save visual
    normal_vis_path = output_dir / 'normals_vis' / f"{image_id}.jpg"
    cv2.imwrite(str(normal_vis_path), cv2.cvtColor(normals_vis, cv2.COLOR_RGB2BGR))

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    # Create directories
    (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
    (output_dir / 'normals').mkdir(parents=True, exist_ok=True)
    (output_dir / 'normals_vis').mkdir(parents=True, exist_ok=True)
    
    # Load Annotations
    ann_file = data_root / 'annotations' / f"{args.split}.json"
    if not ann_file.exists():
        print(f"Annotation file not found: {ann_file}")
        return
        
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Iterate over images
    # Assuming COCO format where 'images' list has file_names and ids
    images = coco_data.get('images', [])
    annotations_map = defaultdict(list)
    if 'annotations' in coco_data:
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            # Convert simple annotation to object structure expected by process function
            # This is a simplification; adapt to real OmniCity schema
            obj = {'polygon': ann['segmentation'][0]} 
            annotations_map[image_id].append(obj)
            
    from collections import defaultdict
    
    print(f"Processing {len(images)} images...")
    for img_info in tqdm(images):
        width = img_info['width']
        height = img_info['height']
        file_name = img_info['file_name']
        image_id = Path(file_name).stem # ID usually filename without ext
        
        # Mock annotation extraction
        # specific logic depends on whether `annotations_map` is keyed by int ID or string filename
        # Here we use image_id from filename stem
        
        # Prepare annotation dict
        # In a real scenario, you match image_id to annotations
        # For validation, we might simulate or skip if no matching logic is perfect yet
        
        # Placeholder for collected objects
        img_ann = {'objects': annotations_map.get(img_info['id'], [])} # Assuming 'id' matches
        
        process_single_image(image_id, img_ann, data_root, output_dir)

if __name__ == '__main__':
    main()
