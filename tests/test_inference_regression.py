import numpy as np

from deeproof.utils.post_processing import merge_tiles
from deeproof.utils.qa import geometry_qa_flags
from deeproof.utils.vectorization import regularize_building_polygons


def _make_instance(mask: np.ndarray, offset=(0, 0), score=0.9, label=1):
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return {
        'bbox': [x1 + offset[1], y1 + offset[0], x2 + offset[1], y2 + offset[0]],
        'mask_crop': mask[y1:y2, x1:x2],
        'offset': (offset[0] + y1, offset[1] + x1),
        'score': float(score),
        'label': int(label),
        'normal': np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }


def test_merge_tiles_regression_score_mode():
    m1 = np.zeros((64, 64), dtype=bool)
    m2 = np.zeros((64, 64), dtype=bool)
    m1[10:40, 10:40] = True
    m2[12:38, 12:38] = True
    instances = [_make_instance(m1, score=0.95), _make_instance(m2, score=0.80)]

    merged = merge_tiles(instances, iou_threshold=0.5, method='score')
    assert len(merged) == 1
    assert float(merged[0]['score']) == 0.95


def test_merge_tiles_regression_weighted_mode():
    m1 = np.zeros((64, 64), dtype=bool)
    m2 = np.zeros((64, 64), dtype=bool)
    m1[8:32, 8:32] = True
    m2[16:40, 16:40] = True
    instances = [_make_instance(m1, score=0.9), _make_instance(m2, score=0.85)]

    merged = merge_tiles(instances, iou_threshold=0.2, method='weighted')
    assert len(merged) == 1
    out = merged[0]
    assert 'mask_crop' in out
    area = int(np.asarray(out['mask_crop']).sum())
    assert area > 0


def test_polygon_and_geometry_qa_regression():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[20:90, 30:110] = 1
    polys = regularize_building_polygons(mask, epsilon_factor=0.01, min_area=50)
    assert len(polys) >= 1
    pts = polys[0].reshape(-1, 2)
    qa = geometry_qa_flags(
        poly_points=pts,
        width=128,
        height=128,
        normal=[0.0, 0.0, 1.0],
        slope_deg=0.0,
        azimuth_deg=180.0,
    )
    assert qa['qa_all_ok'] is True
