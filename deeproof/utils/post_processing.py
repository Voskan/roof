from typing import Dict, List, Tuple

import numpy as np


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _instance_bbox(instance: Dict) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = instance['bbox']
    return int(x1), int(y1), int(x2), int(y2)


def _to_global_canvas(instance: Dict, min_x: int, min_y: int, max_x: int, max_y: int) -> np.ndarray:
    h = int(max_y - min_y)
    w = int(max_x - min_x)
    canvas = np.zeros((h, w), dtype=bool)

    if 'mask_crop' in instance and instance.get('mask_crop') is not None:
        y_off, x_off = instance['offset']
        mask = np.asarray(instance['mask_crop'], dtype=bool)
        m_h, m_w = mask.shape[:2]
        y0 = int(y_off - min_y)
        x0 = int(x_off - min_x)

        y1 = max(0, y0)
        x1 = max(0, x0)
        y2 = min(h, y0 + m_h)
        x2 = min(w, x0 + m_w)
        if y2 <= y1 or x2 <= x1:
            return canvas
        crop_y1 = y1 - y0
        crop_x1 = x1 - x0
        crop_y2 = crop_y1 + (y2 - y1)
        crop_x2 = crop_x1 + (x2 - x1)
        canvas[y1:y2, x1:x2] = mask[crop_y1:crop_y2, crop_x1:crop_x2]
        return canvas

    if 'mask' in instance and instance.get('mask') is not None:
        mask = np.asarray(instance['mask'], dtype=bool)
        x1, y1, x2, y2 = _instance_bbox(instance)
        off_x = int(x1 - min_x)
        off_y = int(y1 - min_y)
        m_h, m_w = mask.shape[:2]
        yy2 = min(h, off_y + m_h)
        xx2 = min(w, off_x + m_w)
        if yy2 <= off_y or xx2 <= off_x:
            return canvas
        canvas[max(off_y, 0):yy2, max(off_x, 0):xx2] = mask[:yy2 - max(off_y, 0), :xx2 - max(off_x, 0)]
    return canvas


def _pairwise_iou(inst_a: Dict, inst_b: Dict) -> float:
    ax1, ay1, ax2, ay2 = _instance_bbox(inst_a)
    bx1, by1, bx2, by2 = _instance_bbox(inst_b)
    if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
        return 0.0

    min_x = min(ax1, bx1)
    min_y = min(ay1, by1)
    max_x = max(ax2, bx2)
    max_y = max(ay2, by2)
    canvas_a = _to_global_canvas(inst_a, min_x, min_y, max_x, max_y)
    canvas_b = _to_global_canvas(inst_b, min_x, min_y, max_x, max_y)
    return calculate_mask_iou(canvas_a, canvas_b)


def _cluster_by_iou(instances: List[Dict], iou_threshold: float) -> List[List[int]]:
    n = len(instances)
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _pairwise_iou(instances[i], instances[j]) > iou_threshold:
                neighbors[i].append(j)
                neighbors[j].append(i)

    visited = [False] * n
    components: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in neighbors[node]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(comp)
    return components


def _merge_component(component: List[Dict], method: str) -> Dict:
    if len(component) == 1 or method == 'score':
        return max(component, key=lambda x: float(x.get('score', 0.0)))

    x1 = min(int(inst['bbox'][0]) for inst in component)
    y1 = min(int(inst['bbox'][1]) for inst in component)
    x2 = max(int(inst['bbox'][2]) for inst in component)
    y2 = max(int(inst['bbox'][3]) for inst in component)
    h, w = int(y2 - y1), int(x2 - x1)

    masks = []
    scores = []
    for inst in component:
        masks.append(_to_global_canvas(inst, x1, y1, x2, y2).astype(np.float32))
        scores.append(max(float(inst.get('score', 0.0)), 1e-6))
    scores_np = np.asarray(scores, dtype=np.float32)

    if method == 'union':
        merged_mask = np.logical_or.reduce([m > 0.5 for m in masks])
    else:  # method == 'weighted'
        weighted_prob = np.zeros((h, w), dtype=np.float32)
        for score, mask in zip(scores_np, masks):
            weighted_prob += score * mask
        weighted_prob /= float(scores_np.sum())
        merged_mask = weighted_prob >= 0.5

    rows = np.any(merged_mask, axis=1)
    cols = np.any(merged_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return max(component, key=lambda x: float(x.get('score', 0.0)))

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    crop = merged_mask[ymin:ymax + 1, xmin:xmax + 1]

    out = {
        'bbox': [x1 + int(xmin), y1 + int(ymin), x1 + int(xmax + 1), y1 + int(ymax + 1)],
        'mask_crop': crop.astype(bool),
        'offset': (y1 + int(ymin), x1 + int(xmin)),
        'score': float(scores_np.max()),
        'label': int(component[0]['label']),
    }

    normals = [inst.get('normal') for inst in component if inst.get('normal') is not None]
    if normals:
        normal_weights = np.asarray(
            [max(float(inst.get('score', 0.0)), 1e-6) for inst in component if inst.get('normal') is not None],
            dtype=np.float32)
        fused = np.zeros((3,), dtype=np.float32)
        for w_i, n_i in zip(normal_weights, normals):
            fused += w_i * np.asarray(n_i, dtype=np.float32)
        norm = float(np.linalg.norm(fused))
        if norm > 1e-8:
            fused /= norm
        out['normal'] = fused.astype(np.float32)

    return out


def merge_tiles(
    instances: List[Dict],
    iou_threshold: float = 0.5,
    method: str = 'score',
) -> List[Dict]:
    """
    Merge overlapping tile instances with class-aware mask IoU graph clustering.

    method:
    - score: keep best-scoring instance per cluster
    - union: union masks inside cluster
    - weighted: score-weighted mask vote inside cluster
    """
    if not instances:
        return []
    if method not in {'score', 'union', 'weighted'}:
        raise ValueError(f'Unsupported merge method: {method}')

    by_label: Dict[int, List[Dict]] = {}
    for inst in instances:
        label = int(inst.get('label', -1))
        by_label.setdefault(label, []).append(inst)

    merged_all: List[Dict] = []
    for _, label_instances in by_label.items():
        components = _cluster_by_iou(label_instances, iou_threshold=float(iou_threshold))
        for comp in components:
            comp_instances = [label_instances[i] for i in comp]
            merged_all.append(_merge_component(comp_instances, method=method))

    merged_all.sort(key=lambda x: float(x.get('score', 0.0)), reverse=True)
    return merged_all
