import argparse
import json
from pathlib import Path


DEFAULT_THRESHOLDS = {
    'mIoU': 85.0,
    'AP50': 90.0,
    'BFScore': 80.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRoof model registry and KPI gate')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--metrics-json', required=True, help='Metrics JSON path')
    parser.add_argument('--registry', default='work_dirs/model_registry.json', help='Registry JSON path')
    parser.add_argument('--kpi-thresholds', default='', help='Optional JSON string with KPI thresholds')
    return parser.parse_args()


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def _status_from_metrics(metrics: dict, thresholds: dict) -> str:
    for key, thr in thresholds.items():
        value = float(metrics.get(key, 0.0))
        if value < float(thr):
            return 'rejected'
    return 'promoted'


def main():
    args = parse_args()
    ckpt = str(Path(args.checkpoint))
    metrics = _load_json(Path(args.metrics_json))

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.kpi_thresholds:
        thresholds.update(json.loads(args.kpi_thresholds))

    status = _status_from_metrics(metrics, thresholds)
    entry = {
        'checkpoint': ckpt,
        'metrics': metrics,
        'thresholds': thresholds,
        'status': status,
    }

    reg_path = Path(args.registry)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    if reg_path.exists():
        registry = _load_json(reg_path)
        if not isinstance(registry, list):
            registry = []
    else:
        registry = []
    registry.append(entry)
    reg_path.write_text(json.dumps(registry, indent=2), encoding='utf-8')
    print(f'Registry updated: {reg_path}')
    print(f'Model status: {status}')


if __name__ == '__main__':
    main()
