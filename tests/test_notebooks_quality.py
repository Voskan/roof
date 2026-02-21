import json
import re
from pathlib import Path


def _code_text(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    return '\n'.join(
        ''.join(c.get('source', []))
        for c in nb.get('cells', [])
        if c.get('cell_type') == 'code')


def test_train_notebook_reproducible():
    path = Path('notebooks/train_deeproof.ipynb')
    txt = _code_text(path)

    assert 'Runner.from_cfg' in txt
    assert 'apply_runtime_compat' in txt

    # Must not contain old environment surgery.
    assert 'nuclear_cuda_fix' not in txt
    assert '_upgrade_mask2former_cfg_inplace' not in txt
    assert re.search(r'pip\\s+install', txt) is None
    assert '/workspace' not in txt
    assert '/Users/' not in txt


def test_inference_notebook_production_aligned():
    path = Path('notebooks/checkpoint_inference_test.ipynb')
    txt = _code_text(path)

    assert 'inference.py' in txt
    assert "PROJECT_ROOT / 'tools'" in txt or 'PROJECT_ROOT / "tools"' in txt
    assert 'subprocess.check_call' in txt

    # Notebook should not re-implement decode_head pipeline.
    assert 'model.decode_head' not in txt
    assert 'all_cls_scores' not in txt
    assert '/workspace' not in txt
    assert '/Users/' not in txt


def test_demo_notebook_slope_column_compat():
    path = Path('notebooks/02_demo_result.ipynb')
    txt = _code_text(path)
    assert "slope_col = 'slope_deg'" in txt
