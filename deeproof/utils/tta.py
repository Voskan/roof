import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmseg.structures import SegDataSample


def _instances_from_semantic(sem_map: torch.Tensor, min_area: int = 64) -> InstanceData:
    """Fallback conversion from semantic map to instance-style predictions."""
    if sem_map.ndim == 3:
        sem_map = sem_map.squeeze(0)
    sem_map = sem_map.long()
    device = sem_map.device
    h, w = int(sem_map.shape[-2]), int(sem_map.shape[-1])

    masks, labels, scores = [], [], []
    for cls_id in torch.unique(sem_map).tolist():
        cls_id = int(cls_id)
        if cls_id <= 0 or cls_id == 255:
            continue
        cls_mask = (sem_map == cls_id)
        if int(cls_mask.sum().item()) < min_area:
            continue
        comp_map = cls_mask.detach().cpu().numpy().astype(np.uint8)
        num_comp, comp_labels = cv2.connectedComponents(comp_map, connectivity=8)
        for comp_idx in range(1, int(num_comp)):
            comp = (comp_labels == comp_idx)
            if int(comp.sum()) < min_area:
                continue
            masks.append(torch.from_numpy(comp).to(device=device))
            labels.append(cls_id)
            # Keep semantic fallback confidence low; query-based instances
            # should dominate whenever available.
            scores.append(0.05)

    out = InstanceData()
    if masks:
        out.masks = torch.stack([m.bool() for m in masks], dim=0)
        out.labels = torch.tensor(labels, dtype=torch.long, device=device)
        out.scores = torch.tensor(scores, dtype=torch.float32, device=device)
    else:
        out.masks = torch.zeros((0, h, w), dtype=torch.bool, device=device)
        out.labels = torch.zeros((0,), dtype=torch.long, device=device)
        out.scores = torch.zeros((0,), dtype=torch.float32, device=device)
    return out


def _rotate_normal_vector(nx: float, ny: float, nz: float, degrees_ccw: float):
    theta = np.deg2rad(degrees_ccw)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    nx_new = nx * cos_t - ny * sin_t
    ny_new = nx * sin_t + ny * cos_t
    return nx_new, ny_new, nz


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _bbox_from_mask(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax + 1), int(ymax + 1)]


def _normalize_normal(normal: np.ndarray) -> np.ndarray:
    mag = float(np.linalg.norm(normal))
    if mag > 1e-8:
        return (normal / mag).astype(np.float32)
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def _cluster_instances(instances, iou_threshold: float):
    by_label = {}
    for inst in instances:
        by_label.setdefault(int(inst['label']), []).append(inst)

    merged = []
    for label, label_instances in by_label.items():
        remaining = sorted(label_instances, key=lambda x: x['score'], reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]

            changed = True
            while changed:
                changed = False
                keep = []
                for cand in remaining:
                    if any(_mask_iou(cand['mask'], m['mask']) >= iou_threshold for m in cluster):
                        cluster.append(cand)
                        changed = True
                    else:
                        keep.append(cand)
                remaining = keep

            merged.append(_fuse_cluster(cluster=cluster, label=label))

    merged = [m for m in merged if m is not None]
    merged.sort(key=lambda x: x['score'], reverse=True)
    return merged


def _fuse_cluster(cluster, label: int):
    if not cluster:
        return None

    masks = [c['mask'].astype(np.float32) for c in cluster]
    weights = np.asarray([max(float(c['score']), 1e-6) for c in cluster], dtype=np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        weights = np.ones_like(weights)
        weight_sum = float(weights.sum())

    weighted_prob = np.zeros_like(masks[0], dtype=np.float32)
    for w, m in zip(weights, masks):
        weighted_prob += w * m
    weighted_prob /= weight_sum

    fused_mask = weighted_prob >= 0.5
    if int(fused_mask.sum()) == 0:
        best = max(cluster, key=lambda x: x['score'])
        fused_mask = best['mask'].astype(bool)

    bbox = _bbox_from_mask(fused_mask)
    if bbox is None:
        return None

    score = float(max(c['score'] for c in cluster))
    out = {
        'bbox': bbox,
        'mask': fused_mask.astype(bool),
        'score': score,
        'label': int(label),
    }

    normals = [c.get('normal') for c in cluster if c.get('normal') is not None]
    if normals:
        fused_normal = np.zeros((3,), dtype=np.float32)
        normal_weights = np.asarray(
            [max(float(c['score']), 1e-6) for c in cluster if c.get('normal') is not None],
            dtype=np.float32)
        for w, n in zip(normal_weights, normals):
            fused_normal += w * np.asarray(n, dtype=np.float32)
        out['normal'] = _normalize_normal(fused_normal)

    return out


def apply_tta(
    model,
    image: np.ndarray,
    device='cuda',
    min_score: float = 0.1,
    merge_iou: float = 0.6,
    max_instances: int = 0,
    allow_semantic_fallback: bool = False,
) -> dict:
    """
    Apply TTA and aggregate predictions by mask-level fusion.

    Transforms:
    1) identity
    2) rot90
    3) rot180
    4) rot270
    5) hflip
    """
    model.eval()
    h, w, _ = image.shape
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

    transforms = ['id', 'rot90', 'rot180', 'rot270', 'hflip']
    batch_imgs = [
        img_tensor,
        torch.rot90(img_tensor, k=1, dims=[1, 2]),
        torch.rot90(img_tensor, k=2, dims=[1, 2]),
        torch.rot90(img_tensor, k=3, dims=[1, 2]),
        torch.flip(img_tensor, dims=[2]),
    ]
    batch_stack = torch.stack(batch_imgs)

    batch_samples = [
        SegDataSample(metainfo=dict(img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w)))
        for _ in transforms
    ]

    with torch.no_grad():
        batch_data = dict(inputs=[img for img in batch_stack], data_samples=batch_samples)
        results = model.test_step(batch_data)

    all_pred_instances = []

    for i, res in enumerate(results):
        transform = transforms[i]
        preds = getattr(res, 'pred_instances', None)
        if preds is None or len(preds) == 0:
            if not allow_semantic_fallback:
                continue
            sem = getattr(res, 'pred_sem_seg', None)
            sem_data = getattr(sem, 'data', None) if sem is not None else None
            if torch.is_tensor(sem_data):
                preds = _instances_from_semantic(sem_data)
            else:
                continue

        if len(preds) == 0:
            continue

        masks = preds.masks
        scores = preds.scores
        labels = preds.labels
        has_normals = hasattr(preds, 'normals')

        if transform == 'id':
            inv_masks = masks
        elif transform == 'rot90':
            inv_masks = torch.rot90(masks, k=3, dims=[1, 2])
        elif transform == 'rot180':
            inv_masks = torch.rot90(masks, k=2, dims=[1, 2])
        elif transform == 'rot270':
            inv_masks = torch.rot90(masks, k=1, dims=[1, 2])
        elif transform == 'hflip':
            inv_masks = torch.flip(masks, dims=[2])
        else:
            inv_masks = masks

        for k in range(len(scores)):
            score = float(scores[k].detach().cpu().item())
            if score < float(min_score):
                continue

            mask = inv_masks[k].detach().cpu().numpy().astype(bool)
            if int(mask.sum()) == 0:
                continue

            label = int(labels[k].detach().cpu().item())
            inst = {
                'mask': mask,
                'score': score,
                'label': label,
            }

            if has_normals:
                n = preds.normals[k].detach().cpu().numpy().astype(np.float32)
                nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
                if transform == 'id':
                    inv_nx, inv_ny, inv_nz = nx, ny, nz
                elif transform == 'rot90':
                    inv_nx, inv_ny, inv_nz = _rotate_normal_vector(nx, ny, nz, -90)
                elif transform == 'rot180':
                    inv_nx, inv_ny, inv_nz = _rotate_normal_vector(nx, ny, nz, 180)
                elif transform == 'rot270':
                    inv_nx, inv_ny, inv_nz = _rotate_normal_vector(nx, ny, nz, 90)
                elif transform == 'hflip':
                    inv_nx, inv_ny, inv_nz = -nx, ny, nz
                else:
                    inv_nx, inv_ny, inv_nz = nx, ny, nz
                inst['normal'] = _normalize_normal(np.array([inv_nx, inv_ny, inv_nz], dtype=np.float32))

            all_pred_instances.append(inst)

    if not all_pred_instances:
        return {'instances': [], 'count': 0}

    final_instances = _cluster_instances(all_pred_instances, iou_threshold=float(merge_iou))
    if max_instances and max_instances > 0:
        final_instances = final_instances[: int(max_instances)]

    return {'instances': final_instances, 'count': len(final_instances)}
