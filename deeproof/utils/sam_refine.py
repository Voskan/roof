from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def build_sam_prompts(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build prompts from a binary mask:
    - bbox
    - positive points (inside)
    - negative points (ring around boundary)
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if ys.size == 0:
        return np.zeros((4,), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    bbox = np.asarray([x1, y1, x2, y2], dtype=np.float32)

    # positive points: centroid + up to 8 sampled points
    cx = float(xs.mean())
    cy = float(ys.mean())
    pos = [(cx, cy)]
    step = max(int(len(xs) // 8), 1)
    for i in range(0, len(xs), step):
        pos.append((float(xs[i]), float(ys[i])))
        if len(pos) >= 9:
            break

    # negative points: ring between dilated and eroded masks
    dil = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)
    ero = cv2.erode(m, np.ones((5, 5), np.uint8), iterations=1)
    ring = np.logical_and(dil > 0, ero == 0)
    rys, rxs = np.where(ring)
    neg = []
    if rys.size > 0:
        step_r = max(int(rys.size // 16), 1)
        for i in range(0, rys.size, step_r):
            neg.append((float(rxs[i]), float(rys[i])))
            if len(neg) >= 16:
                break

    return bbox, np.asarray(pos, dtype=np.float32), np.asarray(neg, dtype=np.float32)


def _try_build_predictor(model_type: str, checkpoint: str):
    # SAM2 path
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        predictor = SAM2ImagePredictor(build_sam2(model_type, checkpoint))
        return predictor, 'sam2'
    except Exception:
        pass

    # SAM1 path
    try:
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        predictor = SamPredictor(sam)
        return predictor, 'sam1'
    except Exception:
        return None, ''


def refine_instances_with_sam(
    image_rgb: np.ndarray,
    instances: List[Dict],
    model_type: str,
    checkpoint: str,
    min_improve_iou: float = 0.02,
) -> List[Dict]:
    """
    Optional SAM/SAM2 refinement.
    If SAM runtime is unavailable, returns original instances unchanged.
    """
    if not instances:
        return instances
    predictor, kind = _try_build_predictor(model_type=model_type, checkpoint=checkpoint)
    if predictor is None:
        return instances

    img = np.asarray(image_rgb).copy()
    predictor.set_image(img)

    refined = []
    for inst in instances:
        if 'mask_crop' in inst and inst.get('mask_crop') is not None:
            h = int(inst['bbox'][3] - inst['bbox'][1])
            w = int(inst['bbox'][2] - inst['bbox'][0])
            full_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            y_off, x_off = inst['offset']
            crop = (np.asarray(inst['mask_crop']) > 0).astype(np.uint8)
            y2 = min(y_off + crop.shape[0], full_mask.shape[0])
            x2 = min(x_off + crop.shape[1], full_mask.shape[1])
            full_mask[y_off:y2, x_off:x2] = crop[: y2 - y_off, : x2 - x_off]
        else:
            continue

        bbox, pos, neg = build_sam_prompts(full_mask)
        if pos.size == 0:
            refined.append(inst)
            continue
        point_coords = np.concatenate([pos, neg], axis=0)
        point_labels = np.concatenate([
            np.ones((len(pos),), dtype=np.int32),
            np.zeros((len(neg),), dtype=np.int32),
        ], axis=0)
        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox[None, :],
                multimask_output=False,
            )
        except TypeError:
            # SAM2 variant may have different kwargs
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox,
                multimask_output=False,
            )
        if masks is None or len(masks) == 0:
            refined.append(inst)
            continue
        new_mask = (masks[0] > 0).astype(np.uint8)
        old_mask = full_mask.astype(bool)
        inter = np.logical_and(old_mask, new_mask > 0).sum()
        union = np.logical_or(old_mask, new_mask > 0).sum()
        iou = float(inter / max(union, 1))
        if iou + float(min_improve_iou) < 0.25:
            refined.append(inst)
            continue

        ys, xs = np.where(new_mask > 0)
        if ys.size == 0:
            refined.append(inst)
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        out = dict(inst)
        out['bbox'] = [x1, y1, x2, y2]
        out['offset'] = (y1, x1)
        out['mask_crop'] = (new_mask[y1:y2, x1:x2] > 0)
        out['sam_refined'] = True
        out['sam_runtime'] = kind
        if scores is not None and len(scores) > 0:
            out['sam_score'] = float(scores[0])
        refined.append(out)
    return refined
