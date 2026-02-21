import argparse
from contextlib import nullcontext
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deeproof.utils.tta import apply_tta
from deeproof.utils.post_processing import merge_tiles


MODE_PRESETS = {
    'speed': dict(
        use_tta=False,
        tta_min_score=0.35,
        tta_merge_iou=0.70,
        tile_merge_iou=0.60,
        max_instances=120,
    ),
    'balanced': dict(
        use_tta=True,
        tta_min_score=0.10,
        tta_merge_iou=0.60,
        tile_merge_iou=0.50,
        max_instances=0,
    ),
    'quality': dict(
        use_tta=True,
        tta_min_score=0.05,
        tta_merge_iou=0.55,
        tile_merge_iou=0.45,
        max_instances=0,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRoof performance profile')
    parser.add_argument('--config', default='', help='Model config')
    parser.add_argument('--checkpoint', default='', help='Model checkpoint')
    parser.add_argument('--input', default='', help='Input image path')
    parser.add_argument('--tile-size', type=int, default=1024, help='Tile size for synthetic benchmark')
    parser.add_argument('--runs', type=int, default=5, help='Number of timed runs')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup runs')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument(
        '--mode',
        choices=sorted(MODE_PRESETS.keys()),
        default='balanced',
        help='Quality-speed preset')
    parser.add_argument('--amp', action='store_true', help='Use torch autocast (AMP)')
    parser.add_argument(
        '--backend',
        choices=['torch', 'torch_tensorrt'],
        default='torch',
        help='Runtime backend (torch_tensorrt falls back to torch when unavailable)')
    parser.add_argument('--out', default='work_dirs/perf_profile.json', help='Output profile json')
    return parser.parse_args()


def _load_image(path: str, tile_size: int) -> np.ndarray:
    if path:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
    img = np.random.randint(0, 255, size=(tile_size, tile_size, 3), dtype=np.uint8)
    return img


def _init_model(config: str, checkpoint: str, device: str):
    from mmseg.apis import init_model  # local import to keep module import light
    return init_model(config, checkpoint, device=device)


def _init_runtime_backend(model, backend: str) -> str:
    if backend == 'torch':
        return 'torch'
    try:
        import torch_tensorrt  # noqa: F401
    except Exception:
        return 'torch_fallback(no_torch_tensorrt)'
    # Full TensorRT graph compilation for mmseg test_step depends on mmdeploy setup.
    return 'torch_fallback(mmdeploy_required_for_test_step)'


def _single_pass(model, image: np.ndarray):
    from mmseg.structures import SegDataSample
    h, w, _ = image.shape
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    batch_data = dict(
        inputs=[img_tensor],
        data_samples=[SegDataSample(metainfo=dict(img_shape=(h, w), ori_shape=(h, w), pad_shape=(h, w)))],
    )
    with torch.no_grad():
        results = model.test_step(batch_data)
    res = results[0]
    pred_instances = getattr(res, 'pred_instances', None)
    if pred_instances is None or len(pred_instances) == 0:
        return {'instances': [], 'count': 0}
    instances = []
    masks = pred_instances.masks.detach().cpu().numpy().astype(bool)
    scores = pred_instances.scores.detach().cpu().numpy()
    labels = pred_instances.labels.detach().cpu().numpy()
    for i in range(len(scores)):
        m = masks[i]
        ys, xs = np.where(m)
        if ys.size == 0 or xs.size == 0:
            continue
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        instances.append(
            dict(
                bbox=bbox,
                mask=m,
                score=float(scores[i]),
                label=int(labels[i]),
            ))
    return {'instances': instances, 'count': len(instances)}


def _profile_pipeline(
    model,
    image: np.ndarray,
    runs: int,
    warmup: int,
    device: str,
    amp: bool,
    preset: Dict,
):
    def _run_once() -> Dict[str, float]:
        t0 = time.perf_counter()
        prepared = np.ascontiguousarray(image)
        t1 = time.perf_counter()

        if torch.cuda.is_available() and 'cuda' in device:
            torch.cuda.reset_peak_memory_stats()

        amp_enabled = bool(amp and torch.cuda.is_available() and 'cuda' in device)
        amp_ctx = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if amp_enabled else nullcontext()
        )
        with amp_ctx:
            if preset['use_tta']:
                out = apply_tta(
                    model=model,
                    image=prepared,
                    device=device,
                    min_score=float(preset['tta_min_score']),
                    merge_iou=float(preset['tta_merge_iou']),
                    max_instances=int(preset['max_instances']),
                )
            else:
                out = _single_pass(model=model, image=prepared)
        if torch.cuda.is_available() and 'cuda' in device:
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        merged = merge_tiles(
            out.get('instances', []),
            iou_threshold=float(preset['tile_merge_iou']),
            method='score',
        )
        _ = merged
        if torch.cuda.is_available() and 'cuda' in device:
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        peak_mem_mb = 0.0
        if torch.cuda.is_available() and 'cuda' in device:
            peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

        return {
            'preprocess_sec': float(t1 - t0),
            'inference_sec': float(t2 - t1),
            'postprocess_sec': float(t3 - t2),
            'total_sec': float(t3 - t0),
            'peak_mem_mb': peak_mem_mb,
            'instances': int(len(out.get('instances', []))),
        }

    for _ in range(max(warmup, 0)):
        _run_once()

    records: List[Dict[str, float]] = []
    for _ in range(max(runs, 1)):
        records.append(_run_once())

    def _stats(key: str) -> Dict[str, float]:
        arr = np.asarray([r[key] for r in records], dtype=np.float64)
        return {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
        }

    return {
        'runs': int(len(records)),
        'preprocess_sec': _stats('preprocess_sec'),
        'inference_sec': _stats('inference_sec'),
        'postprocess_sec': _stats('postprocess_sec'),
        'total_sec': _stats('total_sec'),
        'peak_mem_mb': _stats('peak_mem_mb'),
        'instances': _stats('instances'),
    }


def main():
    args = parse_args()
    if not args.config or not args.checkpoint:
        raise ValueError('Both --config and --checkpoint are required for perf profiling.')

    preset = MODE_PRESETS[args.mode]
    t0 = time.perf_counter()
    model = _init_model(args.config, args.checkpoint, device=args.device)
    backend_effective = _init_runtime_backend(model=model, backend=args.backend)
    model.eval()
    if torch.cuda.is_available() and 'cuda' in args.device:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    image = _load_image(args.input, args.tile_size)
    stats = _profile_pipeline(
        model=model,
        image=image,
        runs=args.runs,
        warmup=args.warmup,
        device=args.device,
        amp=bool(args.amp),
        preset=preset,
    )
    stats['image_shape'] = list(image.shape)
    stats['device'] = args.device
    stats['mode'] = args.mode
    stats['preset'] = preset
    stats['amp'] = bool(args.amp)
    stats['backend_requested'] = args.backend
    stats['backend_effective'] = backend_effective
    stats['model_init_sec'] = float(t1 - t0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))
    print(f'Profile saved to: {out}')


if __name__ == '__main__':
    main()
