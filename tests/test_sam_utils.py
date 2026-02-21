import numpy as np

from deeproof.utils.sam_refine import build_sam_prompts


def test_build_sam_prompts_non_empty():
    m = np.zeros((64, 64), dtype=np.uint8)
    m[16:48, 20:44] = 1
    bbox, pos, neg = build_sam_prompts(m)
    assert bbox.shape == (4,)
    assert pos.ndim == 2 and pos.shape[1] == 2
    assert pos.shape[0] >= 1
    assert neg.ndim == 2 and neg.shape[1] == 2


def test_build_sam_prompts_empty_mask():
    m = np.zeros((32, 32), dtype=np.uint8)
    bbox, pos, neg = build_sam_prompts(m)
    assert pos.shape[0] == 0
    assert neg.shape[0] == 0
