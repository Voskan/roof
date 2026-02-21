import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_generate_pseudo_gt_from_dsm(tmp_path: Path):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 1] = 120
    dsm = np.zeros((48, 64), dtype=np.float32)
    yy, xx = np.mgrid[0:48, 0:64].astype(np.float32)
    dsm = 0.05 * xx + 0.02 * yy + 10.0

    img_path = tmp_path / 'img.png'
    dsm_path = tmp_path / 'dsm.npy'
    mask_path = tmp_path / 'mask.png'
    cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    np.save(str(dsm_path), dsm)
    cv2.imwrite(str(mask_path), np.full((48, 64), 255, dtype=np.uint8))

    pairs_csv = tmp_path / 'pairs.csv'
    with open(pairs_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'image_path', 'dsm_path', 'mask_path'])
        writer.writeheader()
        writer.writerow({
            'sample_id': 's1',
            'image_path': str(img_path),
            'dsm_path': str(dsm_path),
            'mask_path': str(mask_path),
        })

    out_dir = tmp_path / 'out'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'data' / 'generate_pseudo_gt_from_dsm.py'
    subprocess.check_call([
        sys.executable,
        str(script),
        '--pairs-csv',
        str(pairs_csv),
        '--output-dir',
        str(out_dir),
    ])

    assert (out_dir / 'depths' / 's1.npy').exists()
    assert (out_dir / 'normals' / 's1.npy').exists()
    assert (out_dir / 'manifest.json').exists()
    manifest = json.loads((out_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert len(manifest) == 1
