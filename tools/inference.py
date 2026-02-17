
import argparse
import os
import sys
import numpy as np
import torch
import cv2
import rasterio
from tqdm import tqdm
from pathlib import Path
import logging
from rasterio.transform import Affine

# Add project root to path for local module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmseg.apis import init_model
from deeproof.utils.tta import apply_tta
from deeproof.utils.post_processing import merge_tiles
from deeproof.utils.geometry import get_slope, get_azimuth
from deeproof.utils.vectorization import regularize_building_polygons
from deeproof.utils.export import export_to_geojson
from deeproof.models.deeproof_model import DeepRoofMask2Former # Ensure model registration

# Setup Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepRoof-Inference")

def parse_args():
    parser = argparse.ArgumentParser(description='DeepRoof-2026: Production Inference Pipeline')
    parser.add_argument('--config', help='Path to model configuration file', required=True)
    parser.add_argument('--checkpoint', help='Path to model weights (.pth)', required=True)
    parser.add_argument(
        '--input',
        help='Input image path (.tif/.tiff/.png/.jpg/.jpeg)',
        required=True)
    parser.add_argument('--output', help='Output GeoJSON file', required=True)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference (e.g., cuda:0 or cpu)')
    parser.add_argument('--tile-size', type=int, default=1024, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=800, help='Overlap stride')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='Confidence threshold for detections')
    parser.add_argument('--save_viz', action='store_true', help='Save visualization overlay as PNG')
    return parser.parse_args()


def _ensure_three_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] > 3:
        return image[:, :, :3]
    return image


def _load_input_image(path: str):
    """Load GeoTIFF or regular image and return RGB image + georeference metadata."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix in {'.tif', '.tiff'}:
        logger.info(f"Reading input GeoTIFF: {path}")
        with rasterio.open(path) as src:
            image = src.read().transpose(1, 2, 0)  # H, W, C
            transform = src.transform
            crs = src.crs
        image = _ensure_three_channels(image)
        return image, transform, crs, True

    logger.info(f"Reading input raster image: {path}")
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image from path: {path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = _ensure_three_channels(image)

    # For non-geospatial images, export polygons in pixel coordinates.
    transform = Affine.identity()
    crs = None
    return image, transform, crs, False

def draw_visual_overlay(image, polygons, output_path):
    """Generates a high-quality visualization of detections."""
    viz = image.copy()
    overlay = image.copy()
    for poly in polygons:
        pts = poly.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0)) # Green fill for facets
        cv2.polylines(viz, [pts], True, (255, 255, 255), 1) # White outlines
    
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, viz, 1 - alpha, 0, viz)
    cv2.imwrite(output_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    logger.info(f"Visual overlay exported to {output_path}")

def run_production_inference():
    args = parse_args()
    
    # 1. Initialize Model with production config
    logger.info(f"Loading model state from {args.checkpoint}...")
    try:
        model = init_model(args.config, args.checkpoint, device=args.device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)

    # 2. Load image and metadata (GeoTIFF or regular image)
    image, transform, crs, has_georef = _load_input_image(args.input)
        
    orig_h, orig_w, _ = image.shape
    logger.info(f"Source Image: {orig_w}x{orig_h} | CRS: {crs}")
    if not has_georef:
        logger.info("Non-geospatial input detected. GeoJSON coordinates will be in pixel space.")

    # 3. Reflection Padding (Requirement: divisible tile size)
    # We pad the image so that the sliding window covers the entire area without edge truncation.
    pad_h = (args.tile_size - (orig_h % args.tile_size)) % args.tile_size
    pad_w = (args.tile_size - (orig_w % args.tile_size)) % args.tile_size
    
    if pad_h > 0 or pad_w > 0:
        logger.info(f"Applying reflection padding (H: +{pad_h}, W: +{pad_w}) to handle edge boundaries.")
        padded_img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded_img = image
        
    ph, pw, _ = padded_img.shape

    # 4. Generate Sliding Windows
    logger.info(f"Decomposing image into {args.tile_size}x{args.tile_size} tiles...")
    tiles = []
    offsets = []
    for y in range(0, ph - args.tile_size + 1, args.stride):
        for x in range(0, pw - args.tile_size + 1, args.stride):
            tiles.append(padded_img[y:y+args.tile_size, x:x+args.tile_size, :])
            offsets.append((y, x))
            
    # 5. Tile-by-Tile Inference with Error Recovery
    all_raw_instances = []
    logger.info(f"Starting inference on {len(tiles)} tiles...")
    
    for i in tqdm(range(len(tiles)), desc="DeepRoof-Inference"):
        tile = tiles[i]
        y_off, x_off = offsets[i]
        
        try:
            # Run inference (includes TTA: Test Time Augmentation)
            result = apply_tta(model, tile, device=args.device)
            instances = result.get('instances', [])
            
            for inst in instances:
                # Requirement: Confidence Filtering
                if inst['score'] < args.min_confidence:
                    continue
                
                # Transform local tile coords to global image space
                all_raw_instances.append({
                    'bbox': [
                        inst['bbox'][0] + x_off, 
                        inst['bbox'][1] + y_off, 
                        inst['bbox'][2] + x_off, 
                        inst['bbox'][3] + y_off
                    ],
                    'mask_crop': inst['mask'],
                    'offset': (y_off, x_off),
                    'score': inst['score'],
                    'label': inst['label'],
                    'normal': inst.get('normal', None)
                })
        except Exception as e:
            # Requirement: Per-tile error handling (prevent job crash)
            logger.warning(f"CRITICAL: Tile {i} at offset ({y_off}, {x_off}) failed processing. Reason: {e}")
            continue

    # 6. Global Post-Processing
    if not all_raw_instances:
        logger.warning("No roof facets detected with confidence >= threshold.")
        return

    # Merge overlapping detections from sliding windows
    merged_instances = merge_tiles(all_raw_instances, iou_threshold=0.5)
    
    final_polygons = []
    final_attributes = []
    
    for idx, inst in enumerate(merged_instances):
        # Local mask -> Polygons (Regularized)
        polys = regularize_building_polygons(inst['mask_crop'])
        y_off, x_off = inst['offset']
        
        for p in polys:
            # Adjust to global coordinates
            global_p = p.copy().astype(float)
            global_p[:, 0, 0] += x_off
            global_p[:, 0, 1] += y_off
            
            # Clip to original image bounds (removing padding artifacts)
            global_p[:, 0, 0] = np.clip(global_p[:, 0, 0], 0, orig_w - 1)
            global_p[:, 0, 1] = np.clip(global_p[:, 0, 1], 0, orig_h - 1)
            
            final_polygons.append(global_p)
            
            # Calculate 3D Attributes from Normals
            attr = {
                'instance_id': idx,
                'confidence': round(float(inst['score']), 4),
                'class': int(inst['label'])
            }
            if inst['normal'] is not None:
                n = inst['normal']
                attr['slope_deg'] = round(float(get_slope(n)), 2)
                attr['azimuth_deg'] = round(float(get_azimuth(n)), 2)
            
            final_attributes.append(attr)

    # 7. Final Output
    logger.info(f"Exporting {len(final_polygons)} features to GeoJSON...")
    # CRS is preserved for GeoTIFF inputs; for PNG/JPG we export pixel-space geometries.
    export_to_geojson(final_polygons, final_attributes, args.output, transform, crs)
    
    # Optional Debug Viz
    if args.save_viz:
        viz_out = str(Path(args.output).with_suffix('.png'))
        draw_visual_overlay(image, final_polygons, viz_out)

    logger.info("Inference Job Completed Successfully.")

if __name__ == '__main__':
    run_production_inference()
