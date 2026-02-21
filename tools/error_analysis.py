import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRoof error analysis and hard sample mining')
    parser.add_argument('--pred-dir', required=True, help='Directory with predicted segmentation masks')
    parser.add_argument('--gt-dir', required=True, help='Directory with GT segmentation masks')
    parser.add_argument('--pred-suffix', default='.png', help='Prediction file suffix')
    parser.add_argument('--gt-suffix', default='.png', help='GT file suffix')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of semantic classes')
    parser.add_argument('--sample-list', default='', help='Optional file with sample ids to evaluate')
    parser.add_argument('--geojson', default='', help='Optional prediction geojson for geometry outlier analysis')
    parser.add_argument('--out-dir', default='work_dirs/analysis', help='Output directory')
    parser.add_argument('--hard-k', type=int, default=200, help='How many hardest samples to export')
    return parser.parse_args()


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    grad = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    return grad > 0


def _boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 1) -> float:
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


def _collect_ids(pred_dir: Path, pred_suffix: str, sample_list: str) -> List[str]:
    if sample_list:
        ids = [line.strip() for line in Path(sample_list).read_text(encoding='utf-8').splitlines() if line.strip()]
        return sorted(set(ids))
    ids = []
    for path in pred_dir.rglob(f'*{pred_suffix}'):
        ids.append(path.name[:-len(pred_suffix)])
    return sorted(set(ids))


def _load_mask(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


def _compute_confusion(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    pred = np.clip(pred, 0, num_classes - 1)
    gt = np.clip(gt, 0, num_classes - 1)
    idx = gt.reshape(-1) * num_classes + pred.reshape(-1)
    hist = np.bincount(idx, minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes).astype(np.int64)


def _iou_per_class(confusion: np.ndarray) -> List[float]:
    ious = []
    for c in range(confusion.shape[0]):
        tp = float(confusion[c, c])
        fp = float(confusion[:, c].sum() - tp)
        fn = float(confusion[c, :].sum() - tp)
        den = tp + fp + fn
        ious.append(tp / den if den > 0 else 0.0)
    return ious


def _roof_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def _analyze_geometry_outliers(geojson_path: str) -> Dict[str, object]:
    if not geojson_path:
        return {}
    path = Path(geojson_path)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding='utf-8'))
    features = data.get('features', [])
    slopes = []
    bad_range = []
    bad_normal = []
    low_conf = []
    for f in features:
        prop = f.get('properties', {})
        slope = prop.get('slope_deg', None)
        az = prop.get('azimuth_deg', None)
        normal = prop.get('normal', None)
        conf = prop.get('geometry_confidence', prop.get('confidence', None))
        fid = prop.get('instance_id', prop.get('polygon_id', None))

        if slope is not None:
            slope = float(slope)
            slopes.append(slope)
            if slope < 0 or slope > 90:
                bad_range.append(fid)
        if az is not None:
            az = float(az)
            if az < 0 or az >= 360:
                bad_range.append(fid)
        if normal is not None and len(normal) == 3:
            norm = float(np.linalg.norm(np.asarray(normal, dtype=np.float32)))
            if abs(norm - 1.0) > 0.05:
                bad_normal.append(fid)
        if conf is not None and float(conf) < 0.30:
            low_conf.append(fid)

    robust_outliers = []
    if slopes:
        s = np.asarray(slopes, dtype=np.float32)
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        mad = max(mad, 1e-6)
        z = np.abs(s - med) / (1.4826 * mad)
        robust_outliers = [float(v) for v in s[z > 3.5]]

    return {
        'num_features': len(features),
        'invalid_range_count': len(bad_range),
        'invalid_normal_count': len(bad_normal),
        'low_confidence_count': len(low_conf),
        'robust_slope_outliers': robust_outliers[:20],
    }


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = _collect_ids(pred_dir, args.pred_suffix, args.sample_list)
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    rows = []

    for sid in sample_ids:
        pred_path = pred_dir / f'{sid}{args.pred_suffix}'
        gt_path = gt_dir / f'{sid}{args.gt_suffix}'
        pred = _load_mask(pred_path)
        gt = _load_mask(gt_path)
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = pred.astype(np.int64)

        confusion += _compute_confusion(pred, gt, args.num_classes)

        roof_iou = _roof_iou(pred, gt)
        bf = _boundary_f1((pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8), tolerance=1)
        hard_score = 0.5 * (1.0 - roof_iou) + 0.5 * (1.0 - bf)
        rows.append({
            'sample_id': sid,
            'roof_iou': roof_iou,
            'bfscore': bf,
            'hard_score': hard_score,
        })

    ious = _iou_per_class(confusion)
    miou = float(np.mean(ious)) if ious else 0.0
    mean_bf = float(np.mean([r['bfscore'] for r in rows])) if rows else 0.0
    mean_roof_iou = float(np.mean([r['roof_iou'] for r in rows])) if rows else 0.0

    rows_sorted = sorted(rows, key=lambda x: x['hard_score'], reverse=True)
    hard_ids = [r['sample_id'] for r in rows_sorted[: max(int(args.hard_k), 0)]]

    csv_path = out_dir / 'per_sample_metrics.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'roof_iou', 'bfscore', 'hard_score'])
        writer.writeheader()
        writer.writerows(rows_sorted)

    hard_path = out_dir / 'hard_examples.txt'
    hard_path.write_text('\n'.join(hard_ids) + ('\n' if hard_ids else ''), encoding='utf-8')

    geom = _analyze_geometry_outliers(args.geojson)

    report = []
    report.append('# DeepRoof Error Analysis Report')
    report.append('')
    report.append(f'- samples_evaluated: {len(rows)}')
    report.append(f'- mIoU: {miou * 100.0:.2f}')
    report.append(f'- roof IoU mean: {mean_roof_iou * 100.0:.2f}')
    report.append(f'- BFScore mean: {mean_bf * 100.0:.2f}')
    report.append('')
    report.append('## Per-class IoU')
    for i, iou in enumerate(ious):
        report.append(f'- class_{i}: {iou * 100.0:.2f}')
    report.append('')
    report.append('## Hardest Samples')
    for row in rows_sorted[: min(30, len(rows_sorted))]:
        report.append(
            f"- {row['sample_id']}: hard_score={row['hard_score']:.4f}, roof_iou={row['roof_iou']:.4f}, bf={row['bfscore']:.4f}"
        )
    report.append('')
    report.append('## Geometry Outliers')
    if geom:
        for k, v in geom.items():
            report.append(f'- {k}: {v}')
    else:
        report.append('- geometry analysis skipped')

    report_path = out_dir / 'error_report.md'
    report_path.write_text('\n'.join(report) + '\n', encoding='utf-8')

    print(f'Report: {report_path}')
    print(f'Hard examples: {hard_path}')
    print(f'Per-sample CSV: {csv_path}')


if __name__ == '__main__':
    main()
