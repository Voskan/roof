import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine


def _write_height_tif(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=1,
        dtype='float32',
        transform=Affine.identity(),
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def test_process_omnicity_smoke(tmp_path: Path):
    data_root = tmp_path / 'omnicity_src'
    out_root = tmp_path / 'omnicity_out'

    (data_root / 'annotations').mkdir(parents=True, exist_ok=True)
    (data_root / 'images').mkdir(parents=True, exist_ok=True)
    (data_root / 'height').mkdir(parents=True, exist_ok=True)

    sample_ids = ['tile_001', 'tile_002', 'tile_003']
    images = []
    annotations = []

    for i, sid in enumerate(sample_ids, start=1):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 1] = 60 + i * 20
        cv2.imwrite(str(data_root / 'images' / f'{sid}.jpg'), img)

        height = np.zeros((64, 64), dtype=np.float32)
        height[16:48, 16:48] = float(i)
        _write_height_tif(data_root / 'height' / f'{sid}.tif', height)

        images.append({
            'id': i,
            'file_name': f'{sid}.jpg',
            'width': 64,
            'height': 64,
        })
        annotations.append({
            'id': i,
            'image_id': i,
            'category_id': 1,
            'segmentation': [[16, 16, 48, 16, 48, 48, 16, 48]],
            'area': 1024,
            'bbox': [16, 16, 32, 32],
            'iscrowd': 0,
        })

    ann = {'images': images, 'annotations': annotations, 'categories': [{'id': 1, 'name': 'roof'}]}
    with open(data_root / 'annotations' / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(ann, f)

    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'data' / 'process_omnicity.py'
    subprocess.check_call([
        sys.executable,
        str(script_path),
        '--data-root',
        str(data_root),
        '--output-dir',
        str(out_root),
        '--split',
        'train',
    ])

    train_file = out_root / 'train.txt'
    assert train_file.exists(), 'train.txt was not generated'

    lines = [line.strip() for line in train_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert set(lines) == set(sample_ids)

    for sid in sample_ids:
        img_path = out_root / 'images' / f'{sid}.jpg'
        mask_path = out_root / 'masks' / f'{sid}.png'
        normal_path = out_root / 'normals' / f'{sid}.npy'
        assert img_path.exists()
        assert mask_path.exists()
        assert normal_path.exists()

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        assert mask is not None
        assert int((mask > 0).sum()) > 0

        normals = np.load(normal_path)
        assert normals.shape == (64, 64, 3)
        assert np.isfinite(normals).all()
