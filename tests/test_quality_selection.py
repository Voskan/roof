import numpy as np
import cv2

from deeproof.utils.quality_selection import rank_candidates, weighted_fuse_images


def test_rank_candidates_prefers_sharper_image():
    img_sharp = np.zeros((128, 128, 3), dtype=np.uint8)
    img_sharp[::8, :] = 255
    img_sharp[:, ::8] = 255
    img_blur = cv2.GaussianBlur(img_sharp, (9, 9), 2.0)

    rankings = rank_candidates([img_blur, img_sharp])
    assert rankings[0]['index'] == 1
    assert rankings[0]['score'] >= rankings[1]['score']


def test_weighted_fuse_images_shape_and_dtype():
    a = np.full((64, 64, 3), 10, dtype=np.uint8)
    b = np.full((64, 64, 3), 110, dtype=np.uint8)
    rankings = [
        {'index': 0, 'score': 0.2},
        {'index': 1, 'score': 0.8},
    ]
    fused = weighted_fuse_images([a, b], rankings)
    assert fused.shape == (64, 64, 3)
    assert fused.dtype == np.uint8
    assert fused.mean() > a.mean()
