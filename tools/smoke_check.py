import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deeproof.utils.qa import geometry_qa_flags
from deeproof.utils.vectorization import regularize_building_polygons


def main():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:40, 20:50] = 1

    polys = regularize_building_polygons(mask, epsilon_factor=0.02, min_area=20)
    if not polys:
        raise RuntimeError('Smoke check failed: no polygons extracted')

    pts = polys[0].reshape(-1, 2)
    qa = geometry_qa_flags(
        poly_points=pts,
        width=64,
        height=64,
        normal=[0.0, 0.0, 1.0],
        slope_deg=0.0,
        azimuth_deg=90.0,
    )
    if not qa.get('qa_all_ok', False):
        raise RuntimeError(f'Smoke check failed: QA flags are invalid: {qa}')

    report = {
        'polygon_count': len(polys),
        'qa': qa,
        'status': 'ok',
    }
    out = Path('work_dirs') / 'smoke_check.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Smoke check passed. Report: {out}')


if __name__ == '__main__':
    main()
