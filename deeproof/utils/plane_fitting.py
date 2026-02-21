from typing import Optional, Tuple

import numpy as np


def _plane_from_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[np.ndarray]:
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = normal / norm
    d = -float(np.dot(normal, p1))
    return np.asarray([normal[0], normal[1], normal[2], d], dtype=np.float32)


def _point_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    n = plane[:3]
    d = float(plane[3])
    return np.abs(points @ n + d)


def fit_plane_ransac(
    points_xyz: np.ndarray,
    iterations: int = 200,
    dist_threshold: float = 1.5,
    min_inliers: int = 50,
    seed: int = 42,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Fit plane ax + by + cz + d = 0 with RANSAC.
    Returns (plane, inlier_indices).
    """
    if points_xyz.shape[0] < 3:
        return None, np.zeros((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    best_plane = None
    best_inliers = np.zeros((0,), dtype=np.int64)
    n_points = points_xyz.shape[0]
    target_min = max(int(min_inliers), 3)

    for _ in range(max(int(iterations), 1)):
        idx = rng.choice(n_points, size=3, replace=False)
        p = _plane_from_points(points_xyz[idx[0]], points_xyz[idx[1]], points_xyz[idx[2]])
        if p is None:
            continue
        dist = _point_plane_distance(points_xyz, p)
        inliers = np.where(dist <= float(dist_threshold))[0]
        if inliers.size > best_inliers.size:
            best_inliers = inliers
            best_plane = p

    if best_plane is None or best_inliers.size < target_min:
        return None, np.zeros((0,), dtype=np.int64)
    return best_plane, best_inliers


def refine_plane_least_squares(points_xyz: np.ndarray) -> Optional[np.ndarray]:
    """
    Least-squares plane fit via SVD.
    """
    if points_xyz.shape[0] < 3:
        return None
    centroid = points_xyz.mean(axis=0, keepdims=True)
    centered = points_xyz - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = normal / norm
    d = -float(np.dot(normal, centroid[0]))
    return np.asarray([normal[0], normal[1], normal[2], d], dtype=np.float32)


def depth_mask_to_points(
    depth_map: np.ndarray,
    mask: np.ndarray,
    xy_scale: float = 1.0,
) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth_map[ys, xs].astype(np.float32)
    valid = np.isfinite(z)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    ys = ys[valid].astype(np.float32)
    xs = xs[valid].astype(np.float32)
    z = z[valid]
    return np.stack([xs * float(xy_scale), ys * float(xy_scale), z], axis=1).astype(np.float32)


def plane_to_normal(plane: np.ndarray) -> np.ndarray:
    n = plane[:3].astype(np.float32)
    norm = float(np.linalg.norm(n))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    n = n / norm
    if n[2] < 0:
        n = -n
    return n
