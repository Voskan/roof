import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import rasterio
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process OmniCity dataset into DeepRoof format')
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Path to OmniCity dataset root')
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for processed data')
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split')
    return parser.parse_args()


def calculate_surface_normal(height_map: np.ndarray) -> np.ndarray:
    """Compute unit surface normals from a height map via Sobel gradients."""
    dz_dx = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)

    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(height_map)], axis=-1)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    return (normals / np.maximum(norm, 1e-8)).astype(np.float32)


def _resolve_first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_annotation_file(data_root: Path, split: str) -> Optional[Path]:
    return _resolve_first_existing([
        data_root / 'annotations' / f'{split}.json',
        data_root / split / 'annotations' / f'{split}.json',
        data_root / f'{split}.json',
    ])


def _build_index(data_root: Path, suffixes: Iterable[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for suffix in suffixes:
        for path in data_root.rglob(f'*{suffix}'):
            stem = path.stem
            index.setdefault(stem, path)
    return index


def _iter_polygons(segmentation) -> Iterable[np.ndarray]:
    """Yield Nx2 polygons for COCO-style segmentation lists."""
    if not isinstance(segmentation, list):
        return

    if segmentation and isinstance(segmentation[0], (int, float)):
        poly = np.asarray(segmentation, dtype=np.float32).reshape(-1, 2)
        if poly.shape[0] >= 3:
            yield poly
        return

    for poly_data in segmentation:
        if not isinstance(poly_data, list):
            continue
        poly = np.asarray(poly_data, dtype=np.float32).reshape(-1, 2)
        if poly.shape[0] >= 3:
            yield poly


def _resolve_image_path(
    data_root: Path,
    split: str,
    image_index: Dict[str, Path],
    stem: str,
    file_name: str,
) -> Optional[Path]:
    file_name = file_name or ''
    candidates = [
        data_root / file_name,
        data_root / 'images' / file_name,
        data_root / split / 'images' / file_name,
        data_root / 'images' / f'{stem}.jpg',
        data_root / 'images' / f'{stem}.png',
        data_root / split / 'images' / f'{stem}.jpg',
        data_root / split / 'images' / f'{stem}.png',
    ]
    resolved = _resolve_first_existing(candidates)
    if resolved is not None:
        return resolved
    return image_index.get(stem)


def _resolve_height_path(
    data_root: Path,
    split: str,
    height_index: Dict[str, Path],
    stem: str,
) -> Optional[Path]:
    candidates = [
        data_root / 'height' / f'{stem}.tif',
        data_root / split / 'height' / f'{stem}.tif',
        data_root / 'heights' / f'{stem}.tif',
        data_root / split / 'heights' / f'{stem}.tif',
    ]
    resolved = _resolve_first_existing(candidates)
    if resolved is not None:
        return resolved
    return height_index.get(stem)


def _load_annotation_map(coco_data: dict) -> Dict[int, List[dict]]:
    ann_map: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco_data.get('annotations', []):
        image_id = ann.get('image_id')
        if image_id is None:
            continue
        ann_map[int(image_id)].append(ann)
    return ann_map


def _write_debug_image_from_height(height_map: np.ndarray, target_path: Path) -> None:
    vis = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(target_path), vis_rgb)


def process_dataset(data_root: Path, output_dir: Path, split: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
    (output_dir / 'normals').mkdir(parents=True, exist_ok=True)
    (output_dir / 'normals_vis').mkdir(parents=True, exist_ok=True)

    ann_file = _find_annotation_file(data_root, split)
    if ann_file is None:
        print(f'Annotation file for split="{split}" not found in {data_root}')
        return 0

    with open(ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    images = coco_data.get('images', [])
    if not images:
        print(f'No images found in annotation file: {ann_file}')
        return 0

    ann_map = _load_annotation_map(coco_data)
    image_index = _build_index(data_root, suffixes=['.jpg', '.jpeg', '.png'])
    height_index = _build_index(data_root, suffixes=['.tif', '.tiff'])

    processed_ids: List[str] = []

    for img_info in tqdm(images, desc=f'Processing {split}'):
        image_id = int(img_info.get('id', -1))
        file_name = str(img_info.get('file_name', ''))
        stem = Path(file_name).stem if file_name else str(image_id)
        if not stem:
            stem = str(image_id)

        height_path = _resolve_height_path(data_root, split, height_index, stem)
        if height_path is None:
            continue

        with rasterio.open(height_path) as src:
            height_map = src.read(1).astype(np.float32)

        h, w = height_map.shape
        instance_mask = np.zeros((h, w), dtype=np.uint16)

        image_annotations = ann_map.get(image_id, [])
        inst_id = 1
        for ann in image_annotations:
            segmentation = ann.get('segmentation')
            for poly in _iter_polygons(segmentation):
                poly_i32 = np.rint(poly).astype(np.int32)
                if poly_i32.shape[0] < 3:
                    continue
                cv2.fillPoly(instance_mask, [poly_i32], int(inst_id))
                inst_id += 1

        normals = calculate_surface_normal(height_map)
        normals_vis = ((normals + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        cv2.imwrite(str(output_dir / 'masks' / f'{stem}.png'), instance_mask)
        np.save(output_dir / 'normals' / f'{stem}.npy', normals)
        cv2.imwrite(
            str(output_dir / 'normals_vis' / f'{stem}.jpg'),
            cv2.cvtColor(normals_vis, cv2.COLOR_RGB2BGR))

        src_image_path = _resolve_image_path(
            data_root=data_root,
            split=split,
            image_index=image_index,
            stem=stem,
            file_name=file_name,
        )
        dst_image_path = output_dir / 'images' / f'{stem}.jpg'
        if src_image_path is not None:
            try:
                shutil.copy2(src_image_path, dst_image_path)
            except Exception:
                _write_debug_image_from_height(height_map, dst_image_path)
        else:
            _write_debug_image_from_height(height_map, dst_image_path)

        processed_ids.append(stem)

    split_file = output_dir / f'{split}.txt'
    with open(split_file, 'w', encoding='utf-8') as f:
        for image_id in processed_ids:
            f.write(f'{image_id}\n')

    if split == 'train':
        train_file = output_dir / 'train.txt'
        if train_file != split_file:
            shutil.copy2(split_file, train_file)

    print(f'Processed {len(processed_ids)} samples for split="{split}"')
    return len(processed_ids)


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    process_dataset(data_root=data_root, output_dir=output_dir, split=args.split)


if __name__ == '__main__':
    main()
