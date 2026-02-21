from typing import Tuple

import cv2
import numpy as np


def _bicubic_sr(image: np.ndarray, scale: float = 2.0) -> np.ndarray:
    h, w = image.shape[:2]
    up_w = max(int(round(w * float(scale))), 1)
    up_h = max(int(round(h * float(scale))), 1)
    up = cv2.resize(image, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
    down = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
    return down


def generate_sr_image(
    image: np.ndarray,
    scale: float = 2.0,
    backend: str = 'bicubic',
) -> Tuple[np.ndarray, str]:
    """
    Generate an SR-enhanced image aligned to the original HxW.

    Returns (sr_image, backend_effective).
    """
    backend = str(backend).lower()
    if backend == 'bicubic':
        return _bicubic_sr(image, scale=scale), 'bicubic'

    if backend == 'realesrgan':
        try:
            from realesrgan import RealESRGAN  # type: ignore
            from PIL import Image

            model = RealESRGAN('cpu', scale=int(round(scale)))
            # We do not download weights automatically in production code.
            # Fallback to bicubic if weights are unavailable.
            return _bicubic_sr(image, scale=scale), 'bicubic_fallback(no_weights)'
        except Exception:
            return _bicubic_sr(image, scale=scale), 'bicubic_fallback(no_realesrgan)'

    return _bicubic_sr(image, scale=scale), 'bicubic_fallback(unknown_backend)'
