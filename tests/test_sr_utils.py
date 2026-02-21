import numpy as np

from deeproof.utils.sr import generate_sr_image


def test_generate_sr_image_bicubic():
    img = np.random.randint(0, 255, size=(64, 96, 3), dtype=np.uint8)
    sr, backend = generate_sr_image(img, scale=2.0, backend='bicubic')
    assert sr.shape == img.shape
    assert sr.dtype == np.uint8
    assert backend.startswith('bicubic')
