from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


def _get_field(sample, name: str):
    if hasattr(sample, name):
        return getattr(sample, name)
    if isinstance(sample, dict):
        return sample.get(name, None)
    return None


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    grad = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    return grad > 0


def _f1_boundary(pred: np.ndarray, gt: np.ndarray, tolerance: int = 1) -> float:
    pred_b = _boundary_mask(pred)
    gt_b = _boundary_mask(gt)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    gt_dil = cv2.dilate(gt_b.astype(np.uint8), kernel, iterations=1) > 0
    pred_dil = cv2.dilate(pred_b.astype(np.uint8), kernel, iterations=1) > 0

    precision = float((pred_b & gt_dil).sum()) / max(float(pred_b.sum()), 1.0)
    recall = float((gt_b & pred_dil).sum()) / max(float(gt_b.sum()), 1.0)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _to_numpy_sem(seg_obj) -> Optional[np.ndarray]:
    if seg_obj is None:
        return None
    data = _get_field(seg_obj, 'data')
    if data is None:
        return None
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    return np.asarray(data).squeeze()


def _masks_to_numpy(instance_obj) -> np.ndarray:
    masks = _get_field(instance_obj, 'masks')
    if masks is None:
        return np.zeros((0, 1, 1), dtype=bool)
    if hasattr(masks, 'to_tensor'):
        masks = masks.to_tensor(dtype=torch.bool, device='cpu')
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    return masks.astype(bool)


def _scores_to_numpy(instance_obj, default_count: int) -> np.ndarray:
    scores = _get_field(instance_obj, 'scores')
    if scores is None:
        return np.ones((default_count,), dtype=np.float32)
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size != default_count:
        return np.ones((default_count,), dtype=np.float32)
    return scores


def _pairwise_iou(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    n_pred = pred_masks.shape[0]
    n_gt = gt_masks.shape[0]
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float32)

    iou = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i in range(n_pred):
        p = pred_masks[i]
        for j in range(n_gt):
            g = gt_masks[j]
            inter = np.logical_and(p, g).sum()
            if inter == 0:
                continue
            union = np.logical_or(p, g).sum()
            if union > 0:
                iou[i, j] = float(inter) / float(union)
    return iou


def _greedy_ap(iou_mat: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    n_pred, n_gt = iou_mat.shape
    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0
    if n_pred == 0:
        return 0.0, 0.0
    if n_gt == 0:
        return 0.0, 1.0

    order = np.argsort(-scores)
    used_gt = set()
    tp = 0
    for p_idx in order:
        gt_idx = int(np.argmax(iou_mat[p_idx])) if n_gt > 0 else -1
        best = float(iou_mat[p_idx, gt_idx]) if gt_idx >= 0 else 0.0
        if best >= threshold and gt_idx not in used_gt:
            tp += 1
            used_gt.add(gt_idx)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = float(tp) / max(float(tp + fp), 1.0)
    recall = float(tp) / max(float(tp + fn), 1.0)
    return precision, recall


@METRICS.register_module(name='DeepRoofBoundaryMetric')
class DeepRoofBoundaryMetric(BaseMetric):
    default_prefix = 'boundary'

    def __init__(self, tolerance: int = 1, collect_device: str = 'cpu', prefix: str = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.tolerance = int(tolerance)

    def process(self, data_batch: dict, data_samples: List[dict]) -> None:
        for sample in data_samples:
            pred_sem = _to_numpy_sem(_get_field(sample, 'pred_sem_seg'))
            gt_sem = _to_numpy_sem(_get_field(sample, 'gt_sem_seg'))
            if pred_sem is None or gt_sem is None:
                continue

            pred_bin = (pred_sem > 0).astype(np.uint8)
            gt_bin = (gt_sem > 0).astype(np.uint8)
            f1 = _f1_boundary(pred_bin, gt_bin, tolerance=self.tolerance)
            self.results.append(float(f1))

    def compute_metrics(self, results: List[float]) -> Dict[str, float]:
        if not results:
            return {'BFScore': 0.0}
        return {'BFScore': float(np.mean(results) * 100.0)}


@METRICS.register_module(name='DeepRoofFacetMetric')
class DeepRoofFacetMetric(BaseMetric):
    default_prefix = 'facet'

    def __init__(
        self,
        overlap_threshold: float = 0.30,
        collect_device: str = 'cpu',
        prefix: str = None,
    ):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.overlap_threshold = float(overlap_threshold)

    def process(self, data_batch: dict, data_samples: List[dict]) -> None:
        for sample in data_samples:
            pred_inst = _get_field(sample, 'pred_instances')
            gt_inst = _get_field(sample, 'gt_instances')
            if pred_inst is None or gt_inst is None:
                continue

            pred_masks = _masks_to_numpy(pred_inst)
            gt_masks = _masks_to_numpy(gt_inst)
            pred_scores = _scores_to_numpy(pred_inst, default_count=pred_masks.shape[0])

            iou_mat = _pairwise_iou(pred_masks, gt_masks)

            p50, r50 = _greedy_ap(iou_mat, pred_scores, threshold=0.50)
            p75, r75 = _greedy_ap(iou_mat, pred_scores, threshold=0.75)

            over_seg = 0.0
            if gt_masks.shape[0] > 0 and pred_masks.shape[0] > 0:
                gt_overlap = (iou_mat >= self.overlap_threshold).sum(axis=0)
                over_seg = float((gt_overlap > 1).sum()) / float(gt_masks.shape[0])

            under_seg = 0.0
            if pred_masks.shape[0] > 0 and gt_masks.shape[0] > 0:
                pred_overlap = (iou_mat >= self.overlap_threshold).sum(axis=1)
                under_seg = float((pred_overlap > 1).sum()) / float(pred_masks.shape[0])

            topology_consistency = max(0.0, 1.0 - 0.5 * (over_seg + under_seg))

            self.results.append(
                dict(
                    ap50=p50,
                    ar50=r50,
                    ap75=p75,
                    ar75=r75,
                    over_seg=over_seg,
                    under_seg=under_seg,
                    topology_consistency=topology_consistency,
                )
            )

    def compute_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        if not results:
            return {
                'AP50': 0.0,
                'AR50': 0.0,
                'AP75': 0.0,
                'AR75': 0.0,
                'over_seg_rate': 0.0,
                'under_seg_rate': 0.0,
                'topology_consistency': 0.0,
            }

        def _mean(key: str) -> float:
            return float(np.mean([float(r[key]) for r in results]))

        return {
            'AP50': _mean('ap50') * 100.0,
            'AR50': _mean('ar50') * 100.0,
            'AP75': _mean('ap75') * 100.0,
            'AR75': _mean('ar75') * 100.0,
            'over_seg_rate': _mean('over_seg') * 100.0,
            'under_seg_rate': _mean('under_seg') * 100.0,
            'topology_consistency': _mean('topology_consistency') * 100.0,
        }
