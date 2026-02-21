import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from tqdm import tqdm
from mmseg.structures import SegDataSample

# Add project root to path for local module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmseg.apis import init_model

from deeproof.models.deeproof_model import DeepRoofMask2Former  # noqa: F401
from deeproof.utils.calibration import calibrate_probability
from deeproof.utils.export import export_to_geojson
from deeproof.utils.geometry import get_azimuth, get_slope
from deeproof.utils.plane_fitting import (
    depth_mask_to_points,
    fit_plane_ransac,
    plane_to_normal,
    refine_plane_least_squares,
)
from deeproof.utils.post_processing import merge_tiles
from deeproof.utils.qa import geometry_qa_flags
from deeproof.utils.quality_selection import rank_candidates, weighted_fuse_images
from deeproof.utils.roof_graph import extract_and_optimize_graph_from_mask
from deeproof.utils.sam_refine import refine_instances_with_sam
from deeproof.utils.sr import generate_sr_image
from deeproof.utils.tta import apply_tta
from deeproof.utils.vectorization import (
    is_valid_polygon,
    regularize_building_polygons,
    to_cv2_poly,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeepRoof-Inference')


CLASS_NAMES = {
    0: 'background',
    1: 'flat_roof',
    2: 'sloped_roof',
}


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRoof-2026: Production Inference Pipeline')
    parser.add_argument('--config', required=True, help='Path to model configuration file')
    parser.add_argument('--checkpoint', required=True, help='Path to model weights (.pth)')
    parser.add_argument('--input', required=True, help='Input image path (.tif/.tiff/.png/.jpg/.jpeg)')
    parser.add_argument(
        '--input-candidates',
        default='',
        help='Optional txt/json file with candidate images (one path per line). Overrides --input selection.')
    parser.add_argument(
        '--candidate-selection',
        choices=['best', 'weighted'],
        default='best',
        help='Candidate selection strategy for multi-zoom/epoch inputs.')
    parser.add_argument('--output', required=True, help='Output GeoJSON file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference (e.g., cuda:0 or cpu)')
    parser.add_argument('--tile-size', type=int, default=1024, help='Sliding window size')
    parser.add_argument('--stride', type=int, default=800, help='Overlap stride')
    parser.add_argument('--min_confidence', type=float, default=0.35, help='Global score threshold')
    parser.add_argument('--min_confidence_flat', type=float, default=0.35, help='Threshold for flat roof class')
    parser.add_argument('--min_confidence_sloped', type=float, default=0.45, help='Threshold for sloped roof class')
    parser.add_argument('--min_area_px', type=int, default=100, help='Minimum instance area in pixels')
    parser.add_argument('--min_mask_density', type=float, default=0.02, help='Mask pixels / bbox area lower bound')
    parser.add_argument('--max_instances', type=int, default=0, help='Max instances to keep (0 disables cap)')
    parser.add_argument('--tta_min_score', type=float, default=0.10, help='Min score inside TTA before merge')
    parser.add_argument('--tta_merge_iou', type=float, default=0.60, help='Mask IoU threshold for TTA fusion')
    parser.add_argument('--tile_merge_iou', type=float, default=0.50, help='Mask IoU threshold for tile merging')
    parser.add_argument('--sr-enable', action='store_true', help='Enable SR dual-branch inference')
    parser.add_argument('--sr-scale', type=float, default=2.0, help='SR upscale factor')
    parser.add_argument('--sr-backend', default='bicubic', help='SR backend: bicubic|realesrgan')
    parser.add_argument(
        '--sr-fuse-mode',
        choices=['score', 'weighted', 'union'],
        default='weighted',
        help='Fusion method for orig+SR branches')
    parser.add_argument('--graph-enable', action='store_true', help='Enable roof graph extraction/optimization')
    parser.add_argument('--graph-hough-threshold', type=int, default=30, help='Hough threshold for graph line extraction')
    parser.add_argument('--graph-snap-dist', type=float, default=5.0, help='Node snap distance for roof graph')
    parser.add_argument('--graph-save-json', action='store_true', help='Save roof graph sidecar JSON')
    parser.add_argument('--graph-use-model-edge', action='store_true', help='Use model edge head map for graph extraction when available')
    parser.add_argument('--polygon-snap-to-graph', action='store_true', help='Snap polygon vertices to optimized graph lines')
    parser.add_argument('--sam2-enable', action='store_true', help='Enable optional SAM/SAM2 instance refinement')
    parser.add_argument('--sam2-model-type', default='vit_b', help='SAM model type or SAM2 config key')
    parser.add_argument('--sam2-checkpoint', default='', help='Path to SAM/SAM2 checkpoint')
    parser.add_argument('--depth-map', default='', help='Optional depth map (.npy/.png/.tif) for per-facet plane fitting')
    parser.add_argument('--plane-ransac-iters', type=int, default=200, help='RANSAC iterations for plane fitting')
    parser.add_argument('--plane-dist-thr', type=float, default=1.5, help='Distance threshold for plane fit inlier selection')
    parser.add_argument('--plane-min-inliers', type=int, default=50, help='Minimum inliers to accept a fitted plane')
    parser.add_argument('--temperature', type=float, default=1.0, help='Confidence calibration temperature')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Deterministic CUDA mode')
    parser.add_argument('--save_viz', action='store_true', help='Save visualization overlay as PNG')
    parser.add_argument(
        '--viz-fill-alpha',
        type=float,
        default=0.0,
        help='Polygon fill alpha for visualization (0.0 disables fill, contour-only).')
    parser.add_argument(
        '--keep-background',
        action='store_true',
        help='Keep class_id=0 instances (disabled by default for roof-facet output).')
    parser.add_argument('--save_metadata', action='store_true', help='Save metadata sidecar JSON')
    return parser.parse_args()


def _set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _runtime_versions() -> dict:
    versions = {'torch': torch.__version__}
    try:
        import mmcv
        versions['mmcv'] = mmcv.__version__
    except Exception:
        versions['mmcv'] = 'N/A'
    try:
        import mmseg
        versions['mmseg'] = mmseg.__version__
    except Exception:
        versions['mmseg'] = 'N/A'
    try:
        import mmdet
        versions['mmdet'] = mmdet.__version__
    except Exception:
        versions['mmdet'] = 'N/A'
    return versions


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
        logger.info('Reading input GeoTIFF: %s', path)
        with rasterio.open(path) as src:
            image = src.read().transpose(1, 2, 0)
            transform = src.transform
            crs = src.crs
        image = _ensure_three_channels(image)
        return image, transform, crs, True

    logger.info('Reading input raster image: %s', path)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Could not read image from path: {path}')
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = _ensure_three_channels(image)

    transform = Affine.identity()
    crs = None
    return image, transform, crs, False


def _load_candidate_paths(args) -> list:
    if not args.input_candidates:
        return [args.input]
    cand_file = Path(args.input_candidates)
    if not cand_file.exists():
        raise FileNotFoundError(f'Candidate file not found: {cand_file}')
    if cand_file.suffix.lower() == '.json':
        payload = json.loads(cand_file.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            payload = payload.get('inputs', [])
        if not isinstance(payload, list):
            raise ValueError('Candidate JSON must be a list or {"inputs": [...]}')
        paths = [str(p).strip() for p in payload if str(p).strip()]
    else:
        paths = [line.strip() for line in cand_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not paths:
        raise ValueError('No candidate paths found in --input-candidates file.')
    return paths


def _score_threshold_for_label(label: int, args) -> float:
    if int(label) == 2:
        return max(float(args.min_confidence), float(args.min_confidence_sloped))
    if int(label) == 1:
        return max(float(args.min_confidence), float(args.min_confidence_flat))
    return float(args.min_confidence)


def _mask_density(mask: np.ndarray, bbox) -> float:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bbox_area = max((x2 - x1) * (y2 - y1), 1)
    mask_area = int(mask.sum())
    return float(mask_area) / float(bbox_area)


def _passes_quality_gate(inst: dict, args) -> bool:
    score = calibrate_probability(float(inst['score']), float(args.temperature))
    label = int(inst['label'])
    if label <= 0 and not bool(args.keep_background):
        return False
    score_thr = _score_threshold_for_label(label, args)
    if score < score_thr:
        return False

    if inst.get('mask', None) is not None:
        mask = np.asarray(inst['mask'], dtype=bool)
    elif inst.get('mask_crop', None) is not None:
        mask = np.asarray(inst['mask_crop'], dtype=bool)
    else:
        return False
    mask_area = int(mask.sum())
    if mask_area < int(args.min_area_px):
        return False

    density = _mask_density(mask, inst['bbox'])
    if density < float(args.min_mask_density):
        return False

    return True


def _compute_geometry_confidence(score: float, normal, area_px: float, min_area_px: int) -> float:
    score_conf = float(np.clip(score, 0.0, 1.0))
    area_conf = float(np.clip(area_px / max(float(min_area_px) * 5.0, 1.0), 0.0, 1.0))
    normal_conf = 0.5
    if normal is not None:
        normal = np.asarray(normal, dtype=np.float32)
        normal_norm = float(np.linalg.norm(normal))
        normal_conf = float(np.clip(1.0 - abs(normal_norm - 1.0), 0.0, 1.0))
    return float(np.clip(0.60 * score_conf + 0.25 * area_conf + 0.15 * normal_conf, 0.0, 1.0))


def _prepare_instance_for_tile_merge(inst: dict, y_off: int, x_off: int) -> dict:
    x1, y1, x2, y2 = [int(v) for v in inst['bbox']]
    if inst.get('mask_crop', None) is not None and inst.get('offset', None) is not None:
        mask_crop = np.asarray(inst['mask_crop'], dtype=bool)
        inst_off_y, inst_off_x = inst['offset']
        global_off_y = y_off + int(inst_off_y)
        global_off_x = x_off + int(inst_off_x)
    else:
        mask = np.asarray(inst['mask'], dtype=bool)
        mask_crop = mask[y1:y2, x1:x2]
        if mask_crop.size == 0:
            return {}
        global_off_y = y_off + y1
        global_off_x = x_off + x1
    return {
        'bbox': [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off],
        'mask_crop': mask_crop,
        'offset': (global_off_y, global_off_x),
        'score': float(inst['score']),
        'label': int(inst['label']),
        'normal': inst.get('normal', None),
    }


def draw_visual_overlay(
    image: np.ndarray,
    polygons,
    attributes,
    output_path: str,
    fill_alpha: float = 0.0,
):
    """Generate visualization overlay with contour-first rendering."""
    viz = image.copy()
    overlay = image.copy()
    fill_alpha = float(np.clip(fill_alpha, 0.0, 1.0))
    colors = {
        0: (120, 120, 120),   # background / unknown
        1: (40, 180, 99),     # flat roof
        2: (255, 159, 67),    # sloped roof
    }
    for i, poly in enumerate(polygons):
        poly_i32 = to_cv2_poly(poly, round_coords=True)
        if poly_i32 is None:
            continue
        class_id = int(attributes[i].get('class_id', -1)) if i < len(attributes) else -1
        color = colors.get(class_id, (255, 255, 255))
        if fill_alpha > 0.0:
            cv2.fillPoly(overlay, [poly_i32], color)
        cv2.polylines(viz, [poly_i32], True, (255, 255, 255), 2, cv2.LINE_AA)
    if fill_alpha > 0.0:
        cv2.addWeighted(overlay, fill_alpha, viz, 1.0 - fill_alpha, 0, viz)
    cv2.imwrite(output_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    logger.info('Visual overlay exported to %s', output_path)


def _save_metadata(args, output_path: str, image_shape, num_tiles: int, num_instances: int, extras=None):
    metadata = {
        'args': vars(args),
        'runtime_versions': _runtime_versions(),
        'image_shape': list(image_shape),
        'num_tiles': int(num_tiles),
        'num_instances': int(num_instances),
    }
    if extras:
        metadata['extras'] = extras
    meta_path = str(Path(output_path).with_suffix('.meta.json'))
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    logger.info('Metadata saved to %s', meta_path)


def _load_depth_map(path: str, target_hw=None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Depth map not found: {path}')
    if p.suffix.lower() == '.npy':
        depth = np.load(str(p)).astype(np.float32)
    else:
        depth = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f'Could not read depth map: {path}')
        depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if target_hw is not None and depth.shape[:2] != tuple(target_hw):
        h, w = int(target_hw[0]), int(target_hw[1])
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


def _sliding_positions(size: int, tile_size: int, stride: int) -> list:
    """Generate tile start positions with guaranteed right/bottom coverage."""
    size = int(size)
    tile_size = int(tile_size)
    stride = max(int(stride), 1)
    if size <= tile_size:
        return [0]
    last = size - tile_size
    pos = list(range(0, last + 1, stride))
    if not pos or pos[-1] != last:
        pos.append(last)
    # Keep ordering stable and avoid accidental duplicates.
    return list(dict.fromkeys(pos))


def _predict_whole_edge_map(model, image_rgb: np.ndarray, device: str):
    h, w, _ = image_rgb.shape
    img_t = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float()
    sample = SegDataSample(
        metainfo=dict(img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w)))
    try:
        with torch.no_grad():
            results = model.predict(img_t.unsqueeze(0).to(device), [sample])
        if not results:
            return None
        pred_edge = getattr(results[0], 'pred_edge_map', None)
        edge_data = getattr(pred_edge, 'data', None) if pred_edge is not None else None
        if edge_data is None:
            return None
        if torch.is_tensor(edge_data):
            arr = edge_data.detach().cpu().numpy()
        else:
            arr = np.asarray(edge_data)
        if arr.ndim == 3:
            arr = arr[0]
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)
    except Exception:
        return None


def run_production_inference():
    args = parse_args()
    _set_global_seed(seed=int(args.seed), deterministic=bool(args.deterministic))

    logger.info('Loading model state from %s...', args.checkpoint)
    try:
        model = init_model(args.config, args.checkpoint, device=args.device)
        model.eval()
    except Exception as e:
        logger.error('Failed to initialize model: %s', e)
        sys.exit(1)

    candidate_paths = _load_candidate_paths(args)
    if len(candidate_paths) == 1:
        image, transform, crs, has_georef = _load_input_image(candidate_paths[0])
    else:
        loaded = []
        for p in candidate_paths:
            img, tfm, c, geo = _load_input_image(p)
            loaded.append(dict(path=p, image=img, transform=tfm, crs=c, has_georef=geo))
        rankings = rank_candidates([x['image'] for x in loaded])
        logger.info('Candidate ranking (best first): %s', [
            dict(path=loaded[r['index']]['path'], score=round(r['score'], 4))
            for r in rankings
        ])
        best = loaded[int(rankings[0]['index'])]
        if args.candidate_selection == 'weighted':
            image = weighted_fuse_images([x['image'] for x in loaded], rankings)
            transform, crs, has_georef = best['transform'], best['crs'], best['has_georef']
        else:
            image, transform, crs, has_georef = best['image'], best['transform'], best['crs'], best['has_georef']

    orig_h, orig_w, _ = image.shape
    depth_map = _load_depth_map(args.depth_map, target_hw=(orig_h, orig_w)) if args.depth_map else None
    sr_image = None
    sr_backend_effective = ''
    if args.sr_enable:
        sr_image, sr_backend_effective = generate_sr_image(
            image=image,
            scale=float(args.sr_scale),
            backend=str(args.sr_backend),
        )
        logger.info('SR branch enabled, backend=%s', sr_backend_effective)
    logger.info('Source Image: %sx%s | CRS: %s', orig_w, orig_h, crs)
    if not has_georef:
        logger.info('Non-geospatial input detected. GeoJSON coordinates will be in pixel space.')

    pad_h = (args.tile_size - (orig_h % args.tile_size)) % args.tile_size
    pad_w = (args.tile_size - (orig_w % args.tile_size)) % args.tile_size

    if pad_h > 0 or pad_w > 0:
        logger.info(
            'Applying reflection padding (H:+%s, W:+%s) to handle edge boundaries.',
            pad_h,
            pad_w)
        padded_img = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        padded_sr = None
        if sr_image is not None:
            padded_sr = np.pad(sr_image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded_img = image
        padded_sr = sr_image

    ph, pw, _ = padded_img.shape
    logger.info('Decomposing image into %sx%s tiles...', args.tile_size, args.tile_size)

    tiles = []
    offsets = []
    y_positions = _sliding_positions(ph, args.tile_size, args.stride)
    x_positions = _sliding_positions(pw, args.tile_size, args.stride)
    for y in y_positions:
        for x in x_positions:
            tiles.append(padded_img[y:y + args.tile_size, x:x + args.tile_size, :])
            offsets.append((y, x))

    all_raw_instances = []
    logger.info('Starting inference on %s tiles...', len(tiles))

    for i in tqdm(range(len(tiles)), desc='DeepRoof-Inference'):
        tile = tiles[i]
        y_off, x_off = offsets[i]
        try:
            result = apply_tta(
                model=model,
                image=tile,
                device=args.device,
                min_score=float(args.tta_min_score),
                merge_iou=float(args.tta_merge_iou),
                max_instances=int(args.max_instances),
            )
            instances = result.get('instances', [])
            if padded_sr is not None:
                tile_sr = padded_sr[y_off:y_off + args.tile_size, x_off:x_off + args.tile_size, :]
                sr_result = apply_tta(
                    model=model,
                    image=tile_sr,
                    device=args.device,
                    min_score=float(args.tta_min_score),
                    merge_iou=float(args.tta_merge_iou),
                    max_instances=int(args.max_instances),
                )
                sr_instances = sr_result.get('instances', [])
                if sr_instances:
                    instances = merge_tiles(
                        instances + sr_instances,
                        iou_threshold=float(args.tta_merge_iou),
                        method=str(args.sr_fuse_mode),
                    )
            for inst in instances:
                if not _passes_quality_gate(inst, args):
                    continue
                prepared = _prepare_instance_for_tile_merge(inst=inst, y_off=y_off, x_off=x_off)
                if prepared:
                    prepared['score'] = calibrate_probability(
                        float(prepared['score']),
                        float(args.temperature))
                    all_raw_instances.append(prepared)
        except Exception as e:
            logger.warning(
                'Tile %s at offset (%s, %s) failed processing. Reason: %s',
                i,
                y_off,
                x_off,
                e)

    if not all_raw_instances:
        logger.warning('No roof facets detected after quality filters.')
        return

    merged_instances = merge_tiles(
        all_raw_instances,
        iou_threshold=float(args.tile_merge_iou),
        method='score',
    )
    merged_instances.sort(key=lambda x: x['score'], reverse=True)
    if args.max_instances and args.max_instances > 0:
        merged_instances = merged_instances[: int(args.max_instances)]

    if args.sam2_enable and args.sam2_checkpoint:
        try:
            merged_instances = refine_instances_with_sam(
                image_rgb=image,
                instances=merged_instances,
                model_type=str(args.sam2_model_type),
                checkpoint=str(args.sam2_checkpoint),
            )
            logger.info('SAM refinement applied to merged instances.')
        except Exception as e:
            logger.warning('SAM refinement skipped due to runtime error: %s', e)

    graph = None
    structural_lines = None
    if args.graph_enable or args.polygon_snap_to_graph:
        roof_union = None
        if args.graph_use_model_edge:
            roof_union = _predict_whole_edge_map(model=model, image_rgb=image, device=args.device)
        if roof_union is None:
            roof_union = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for inst in merged_instances:
                m = np.asarray(inst.get('mask_crop', None)).astype(np.uint8) if inst.get('mask_crop', None) is not None else None
                if m is None:
                    continue
                y0, x0 = inst['offset']
                y1 = min(y0 + m.shape[0], orig_h)
                x1 = min(x0 + m.shape[1], orig_w)
                if y1 <= y0 or x1 <= x0:
                    continue
                roof_union[y0:y1, x0:x1] = np.maximum(roof_union[y0:y1, x0:x1], m[: y1 - y0, : x1 - x0])
        graph = extract_and_optimize_graph_from_mask(
            roof_mask=roof_union,
            hough_threshold=int(args.graph_hough_threshold),
            snap_distance=float(args.graph_snap_dist),
        )
        structural_lines = graph.get('segments', [])
        if args.graph_save_json:
            graph_path = Path(args.output).with_suffix('.graph.json')
            graph_path.write_text(json.dumps(graph, indent=2), encoding='utf-8')
            logger.info('Roof graph saved to %s', graph_path)

    final_polygons = []
    final_attributes = []

    for idx, inst in enumerate(merged_instances):
        if int(inst.get('label', -1)) <= 0 and not bool(args.keep_background):
            continue
        polys = regularize_building_polygons(
            inst['mask_crop'],
            epsilon_factor=0.015,
            min_area=max(20, int(args.min_area_px)),
            enforce_ortho=False,
            structural_lines=structural_lines if args.polygon_snap_to_graph else None,
            snap_dist=float(args.graph_snap_dist),
        )
        y_off, x_off = inst['offset']

        for poly in polys:
            global_poly = np.asarray(poly, dtype=np.float32).copy()
            if global_poly.ndim != 3 or global_poly.shape[-1] != 2:
                continue
            global_poly[:, 0, 0] += float(x_off)
            global_poly[:, 0, 1] += float(y_off)
            global_poly[:, 0, 0] = np.clip(global_poly[:, 0, 0], 0, orig_w - 1)
            global_poly[:, 0, 1] = np.clip(global_poly[:, 0, 1], 0, orig_h - 1)

            poly_points = global_poly.reshape(-1, 2)
            if not is_valid_polygon(poly_points, min_area=float(args.min_area_px)):
                continue

            poly_area = float(cv2.contourArea(to_cv2_poly(poly_points, round_coords=False)))
            geom_conf = _compute_geometry_confidence(
                score=float(inst['score']),
                normal=inst.get('normal', None),
                area_px=poly_area,
                min_area_px=int(args.min_area_px),
            )

            final_polygons.append(global_poly)
            attr = {
                'instance_id': idx,
                'confidence': round(float(inst['score']), 4),
                'geometry_confidence': round(float(geom_conf), 4),
                'class_id': int(inst['label']),
                'class_name': CLASS_NAMES.get(int(inst['label']), f'class_{int(inst["label"])}'),
                'area_px': round(poly_area, 2),
            }
            if graph is not None:
                attr['graph_nodes'] = int(len(graph.get('nodes', [])))
                attr['graph_edges'] = int(len(graph.get('edges', [])))
            if inst.get('normal', None) is not None:
                n = np.asarray(inst['normal'], dtype=np.float32)
                attr['normal'] = [float(v) for v in n]
                attr['slope_deg'] = round(float(get_slope(n)), 2)
                attr['azimuth_deg'] = round(float(get_azimuth(n)), 2)
                attr['geometry_source'] = 'query_normal'

            if depth_map is not None:
                poly_i32 = to_cv2_poly(poly_points, round_coords=True)
                if poly_i32 is not None:
                    facet_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    cv2.fillPoly(facet_mask, [poly_i32], 1)
                    pts_xyz = depth_mask_to_points(depth_map=depth_map, mask=facet_mask.astype(bool), xy_scale=1.0)
                    if pts_xyz.shape[0] >= max(int(args.plane_min_inliers), 3):
                        _, inliers = fit_plane_ransac(
                            points_xyz=pts_xyz,
                            iterations=int(args.plane_ransac_iters),
                            dist_threshold=float(args.plane_dist_thr),
                            min_inliers=int(args.plane_min_inliers),
                            seed=int(args.seed),
                        )
                        if inliers.size >= max(int(args.plane_min_inliers), 3):
                            plane = refine_plane_least_squares(pts_xyz[inliers])
                            if plane is not None:
                                n = plane_to_normal(plane)
                                attr['normal'] = [float(v) for v in n]
                                attr['slope_deg'] = round(float(get_slope(n)), 2)
                                attr['azimuth_deg'] = round(float(get_azimuth(n)), 2)
                                attr['plane_inliers'] = int(inliers.size)
                                attr['geometry_source'] = 'plane_fit_depth'
            qa = geometry_qa_flags(
                poly_points=poly_points,
                width=orig_w,
                height=orig_h,
                normal=attr.get('normal', None),
                slope_deg=attr.get('slope_deg', None),
                azimuth_deg=attr.get('azimuth_deg', None),
            )
            attr.update(qa)
            final_attributes.append(attr)

    logger.info('Exporting %s features to GeoJSON...', len(final_polygons))
    export_to_geojson(final_polygons, final_attributes, args.output, transform, crs)

    if args.save_viz:
        viz_out = str(Path(args.output).with_suffix('.png'))
        draw_visual_overlay(
            image=image,
            polygons=final_polygons,
            attributes=final_attributes,
            output_path=viz_out,
            fill_alpha=float(args.viz_fill_alpha),
        )

    if args.save_metadata:
        _save_metadata(
            args=args,
            output_path=args.output,
            image_shape=(orig_h, orig_w),
            num_tiles=len(tiles),
            num_instances=len(final_polygons),
            extras=dict(sr_backend=sr_backend_effective if args.sr_enable else ''),
        )

    logger.info('Inference job completed successfully.')


if __name__ == '__main__':
    run_production_inference()
