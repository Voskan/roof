from typing import Dict, Optional

import numpy as np

from deeproof.utils.vectorization import is_valid_polygon


def normal_is_unit(normal, atol: float = 0.05) -> bool:
    if normal is None:
        return False
    n = np.asarray(normal, dtype=np.float32).reshape(-1)
    if n.size != 3 or not np.isfinite(n).all():
        return False
    norm = float(np.linalg.norm(n))
    return abs(norm - 1.0) <= float(atol)


def pitch_in_range(slope_deg: Optional[float]) -> bool:
    if slope_deg is None:
        return False
    try:
        v = float(slope_deg)
    except Exception:
        return False
    return 0.0 <= v <= 90.0


def azimuth_in_range(azimuth_deg: Optional[float]) -> bool:
    if azimuth_deg is None:
        return False
    try:
        v = float(azimuth_deg)
    except Exception:
        return False
    return 0.0 <= v < 360.0


def polygon_in_bounds(poly_points: np.ndarray, width: int, height: int) -> bool:
    pts = np.asarray(poly_points).reshape(-1, 2)
    if not is_valid_polygon(pts, min_area=1.0):
        return False
    x_ok = np.logical_and(pts[:, 0] >= 0.0, pts[:, 0] <= float(width - 1)).all()
    y_ok = np.logical_and(pts[:, 1] >= 0.0, pts[:, 1] <= float(height - 1)).all()
    return bool(x_ok and y_ok)


def geometry_qa_flags(
    poly_points: np.ndarray,
    width: int,
    height: int,
    normal=None,
    slope_deg: Optional[float] = None,
    azimuth_deg: Optional[float] = None,
) -> Dict[str, bool]:
    flags = {
        'qa_polygon_valid': polygon_in_bounds(poly_points, width=width, height=height),
        'qa_normal_unit': normal_is_unit(normal),
        'qa_pitch_valid': pitch_in_range(slope_deg),
        'qa_azimuth_valid': azimuth_in_range(azimuth_deg),
    }
    flags['qa_all_ok'] = all(flags.values())
    return flags
