import numpy as np

from deeproof.utils.plane_fitting import (
    depth_mask_to_points,
    fit_plane_ransac,
    plane_to_normal,
    refine_plane_least_squares,
)


def test_plane_fitting_ransac_and_ls():
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = 0.1 * xx + 0.2 * yy + 5.0
    depth += np.random.default_rng(0).normal(0.0, 0.05, size=depth.shape).astype(np.float32)

    # add outliers
    depth[5:10, 5:10] += 15.0

    mask = np.zeros((h, w), dtype=bool)
    mask[8:56, 8:56] = True
    pts = depth_mask_to_points(depth, mask, xy_scale=1.0)

    _, inliers = fit_plane_ransac(
        points_xyz=pts,
        iterations=150,
        dist_threshold=0.3,
        min_inliers=200,
        seed=0,
    )
    assert inliers.size > 500

    plane = refine_plane_least_squares(pts[inliers])
    assert plane is not None
    n = plane_to_normal(plane)
    # expected plane: z - 0.1x - 0.2y - 5 = 0 -> normal ~ (-0.1, -0.2, 1)
    expected = np.asarray([-0.1, -0.2, 1.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    assert float(np.dot(n, expected)) > 0.98
