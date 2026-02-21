from typing import Dict, List

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0].astype(np.float32)
    return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _sharpness_laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _contrast_std(gray: np.ndarray) -> float:
    return float(gray.std())


def _noise_estimate(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - blur
    return float(np.median(np.abs(residual)))


def _minmax(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.ones_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def rank_candidates(images: List[np.ndarray]) -> List[Dict]:
    if not images:
        return []
    rows = []
    for idx, image in enumerate(images):
        gray = _to_gray(image)
        rows.append(
            dict(
                index=idx,
                sharpness=_sharpness_laplacian_var(gray),
                contrast=_contrast_std(gray),
                noise=_noise_estimate(gray),
            ))

    sharp = np.asarray([r['sharpness'] for r in rows], dtype=np.float32)
    contrast = np.asarray([r['contrast'] for r in rows], dtype=np.float32)
    noise = np.asarray([r['noise'] for r in rows], dtype=np.float32)

    sharp_n = _minmax(sharp)
    contrast_n = _minmax(contrast)
    noise_n = _minmax(noise)

    # Higher sharpness/contrast is better; lower noise is better.
    score = 0.50 * sharp_n + 0.35 * contrast_n + 0.15 * (1.0 - noise_n)
    for i, row in enumerate(rows):
        row['score'] = float(score[i])
    rows.sort(key=lambda x: x['score'], reverse=True)
    return rows


def weighted_fuse_images(images: List[np.ndarray], rankings: List[Dict]) -> np.ndarray:
    if not images:
        raise ValueError('No images provided for weighted fusion.')
    if len(images) == 1:
        return images[0]

    base_h, base_w = images[0].shape[:2]
    score_by_idx = {int(r['index']): max(float(r['score']), 1e-6) for r in rankings}
    fused = np.zeros((base_h, base_w, 3), dtype=np.float32)
    weight_sum = 0.0
    for idx, image in enumerate(images):
        img = image
        if img.shape[:2] != (base_h, base_w):
            img = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
        w = score_by_idx.get(idx, 1e-6)
        fused += w * img.astype(np.float32)
        weight_sum += w
    fused /= max(weight_sum, 1e-6)
    return np.clip(fused, 0, 255).astype(np.uint8)
