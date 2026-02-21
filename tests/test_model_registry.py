import json
import subprocess
import sys
from pathlib import Path


def test_model_registry_kpi_gate(tmp_path: Path):
    metrics = {'mIoU': 86.0, 'AP50': 91.0, 'BFScore': 81.0}
    metrics_path = tmp_path / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics), encoding='utf-8')

    registry_path = tmp_path / 'registry.json'
    script = Path(__file__).resolve().parents[1] / 'tools' / 'model_registry.py'
    subprocess.check_call([
        sys.executable,
        str(script),
        '--checkpoint',
        str(tmp_path / 'iter_10000.pth'),
        '--metrics-json',
        str(metrics_path),
        '--registry',
        str(registry_path),
    ])

    data = json.loads(registry_path.read_text(encoding='utf-8'))
    assert isinstance(data, list) and len(data) == 1
    assert data[0]['status'] == 'promoted'
