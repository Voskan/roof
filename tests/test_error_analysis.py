import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def test_error_analysis_smoke(tmp_path: Path):
    pred_dir = tmp_path / 'pred'
    gt_dir = tmp_path / 'gt'
    out_dir = tmp_path / 'out'
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # sample 1: mostly correct
    gt1 = np.zeros((32, 32), dtype=np.uint8)
    gt1[8:24, 8:24] = 1
    pr1 = gt1.copy()

    # sample 2: poor prediction
    gt2 = np.zeros((32, 32), dtype=np.uint8)
    gt2[10:22, 10:22] = 2
    pr2 = np.zeros((32, 32), dtype=np.uint8)
    pr2[0:10, 0:10] = 1

    cv2.imwrite(str(gt_dir / 's1.png'), gt1)
    cv2.imwrite(str(gt_dir / 's2.png'), gt2)
    cv2.imwrite(str(pred_dir / 's1.png'), pr1)
    cv2.imwrite(str(pred_dir / 's2.png'), pr2)

    script = Path(__file__).resolve().parents[1] / 'tools' / 'error_analysis.py'
    subprocess.check_call([
        sys.executable,
        str(script),
        '--pred-dir',
        str(pred_dir),
        '--gt-dir',
        str(gt_dir),
        '--out-dir',
        str(out_dir),
        '--hard-k',
        '1',
    ])

    report = out_dir / 'error_report.md'
    csv_path = out_dir / 'per_sample_metrics.csv'
    hard = out_dir / 'hard_examples.txt'
    assert report.exists()
    assert csv_path.exists()
    assert hard.exists()

    rows = list(csv.DictReader(csv_path.open('r', encoding='utf-8')))
    assert len(rows) == 2
    hard_ids = [line.strip() for line in hard.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len(hard_ids) == 1
