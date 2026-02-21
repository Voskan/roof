import json
import subprocess
import sys
from pathlib import Path


def test_sync_docs_updates_and_checks(tmp_path: Path):
    readme = tmp_path / 'README.md'
    prd = tmp_path / 'prd.md'
    readme.write_text('# README\n', encoding='utf-8')
    prd.write_text('# PRD\n', encoding='utf-8')

    todo = tmp_path / 'todo.md'
    todo.write_text('### [x] T-001\n### [ ] T-002\n', encoding='utf-8')

    cfg = tmp_path / 'config.py'
    cfg.write_text(
        "num_classes = 3\nmodel = dict(type='DeepRoofMask2Former')\n",
        encoding='utf-8',
    )

    train = tmp_path / 'train.py'
    infer = tmp_path / 'inference.py'
    profile = tmp_path / 'perf_profile.py'
    registry = tmp_path / 'model_registry.py'
    for p in [train, infer, profile, registry]:
        p.write_text('# stub\n', encoding='utf-8')

    facts = {
        'project_name': 'DeepRoof-2026',
        'model': {
            'segmentor': 'DeepRoofMask2Former',
            'backbone': 'Swin Transformer V2-Large',
            'num_classes': 3,
        },
        'quick_start': {
            'train': 'python train.py',
            'inference': 'python inference.py',
            'profile': 'python perf_profile.py',
            'registry': 'python model_registry.py',
        },
        'kpi_targets': {
            'roof_miou': 0.9,
            'instance_ap50': 0.92,
            'bfscore': 0.85,
            'pitch_mae_deg': 3.0,
            'azimuth_mae_deg': 7.0,
        },
        'paths': {
            'production_config': str(cfg),
            'todo': str(todo),
            'train_tool': str(train),
            'inference_tool': str(infer),
            'profile_tool': str(profile),
            'registry_tool': str(registry),
        },
    }
    facts_path = tmp_path / 'facts.json'
    facts_path.write_text(json.dumps(facts), encoding='utf-8')

    script = Path(__file__).resolve().parents[1] / 'tools' / 'sync_docs.py'

    subprocess.check_call([
        sys.executable,
        str(script),
        '--facts',
        str(facts_path),
        '--targets',
        str(readme),
        str(prd),
    ])

    readme_text = readme.read_text(encoding='utf-8')
    assert 'AUTOGEN:PROJECT_FACTS:BEGIN' in readme_text
    assert 'TODO progress' in readme_text

    subprocess.check_call([
        sys.executable,
        str(script),
        '--facts',
        str(facts_path),
        '--targets',
        str(readme),
        str(prd),
        '--check',
    ])
