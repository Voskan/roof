import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


MARKER_BEGIN = '<!-- AUTOGEN:PROJECT_FACTS:BEGIN -->'
MARKER_END = '<!-- AUTOGEN:PROJECT_FACTS:END -->'


def parse_args():
    parser = argparse.ArgumentParser(description='Sync README/PRD facts from a single source-of-truth file.')
    parser.add_argument('--facts', default='docs/source_of_truth.json', help='Path to source-of-truth JSON')
    parser.add_argument('--targets', nargs='+', default=['README.md', 'prd.md'], help='Markdown files to sync')
    parser.add_argument('--check', action='store_true', help='Check mode: fail if drift is detected')
    parser.add_argument('--skip-validate', action='store_true', help='Skip path/config validation checks')
    return parser.parse_args()


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _extract_config_facts(config_path: Path) -> Dict:
    text = config_path.read_text(encoding='utf-8')
    segmentor = ''
    num_classes = -1
    m = re.search(r"model\s*=\s*dict\(\s*type='([^']+)'", text, flags=re.S)
    if m:
        segmentor = m.group(1)
    n = re.search(r"\bnum_classes\s*=\s*(\d+)", text)
    if n:
        num_classes = int(n.group(1))
    return {
        'segmentor': segmentor,
        'num_classes': num_classes,
    }


def _todo_progress(todo_path: Path) -> Tuple[int, int]:
    text = todo_path.read_text(encoding='utf-8')
    done = len(re.findall(r'### \[x\] T-\d+', text))
    open_ = len(re.findall(r'### \[ \] T-\d+', text))
    return done, open_


def validate_facts(facts: Dict):
    paths = facts.get('paths', {})
    required_path_keys = [
        'production_config',
        'todo',
        'train_tool',
        'inference_tool',
        'profile_tool',
        'registry_tool',
    ]
    for key in required_path_keys:
        rel = paths.get(key, '')
        if not rel:
            raise ValueError(f'Missing paths.{key} in source-of-truth')
        if not Path(rel).exists():
            raise ValueError(f'Path from source-of-truth does not exist: {rel}')

    cfg_facts = _extract_config_facts(Path(paths['production_config']))
    expected_segmentor = facts.get('model', {}).get('segmentor', '')
    expected_num_classes = int(facts.get('model', {}).get('num_classes', -1))
    if cfg_facts['segmentor'] != expected_segmentor:
        raise ValueError(
            f'Segmentor mismatch: config={cfg_facts["segmentor"]}, facts={expected_segmentor}')
    if cfg_facts['num_classes'] != expected_num_classes:
        raise ValueError(
            f'num_classes mismatch: config={cfg_facts["num_classes"]}, facts={expected_num_classes}')


def _render_block(facts: Dict) -> str:
    done, open_ = _todo_progress(Path(facts['paths']['todo']))
    quick = facts['quick_start']
    model = facts['model']
    kpi = facts['kpi_targets']
    lines = [
        MARKER_BEGIN,
        '## Synced Project Facts',
        '',
        f"- Project: `{facts['project_name']}`",
        f"- Segmentor: `{model['segmentor']}`",
        f"- Backbone: `{model['backbone']}`",
        f"- Classes: `{model['num_classes']}`",
        f"- TODO progress: `done={done}`, `open={open_}`",
        '',
        '| KPI | Target |',
        '| --- | --- |',
        f"| Roof mIoU | >= {kpi['roof_miou']:.2f} |",
        f"| Instance AP50 | >= {kpi['instance_ap50']:.2f} |",
        f"| BFScore | >= {kpi['bfscore']:.2f} |",
        f"| Pitch MAE (deg) | <= {kpi['pitch_mae_deg']:.1f} |",
        f"| Azimuth MAE (deg) | <= {kpi['azimuth_mae_deg']:.1f} |",
        '',
        '**Canonical Commands**',
        f"- Train: `{quick['train']}`",
        f"- Inference: `{quick['inference']}`",
        f"- Performance profile: `{quick['profile']}`",
        f"- Model registry gate: `{quick['registry']}`",
        MARKER_END,
        '',
    ]
    return '\n'.join(lines)


def upsert_autogen_block(text: str, block: str) -> str:
    begin = text.find(MARKER_BEGIN)
    end = text.find(MARKER_END)
    if begin != -1 and end != -1 and end > begin:
        end += len(MARKER_END)
        while end < len(text) and text[end] == '\n':
            end += 1
        return text[:begin] + block + text[end:]
    suffix = '' if text.endswith('\n') else '\n'
    return text + suffix + '\n' + block


def sync_targets(targets: List[Path], block: str, check: bool) -> int:
    changed: List[Path] = []
    for target in targets:
        old = target.read_text(encoding='utf-8')
        new = upsert_autogen_block(old, block)
        if old != new:
            changed.append(target)
            if not check:
                target.write_text(new, encoding='utf-8')
    if check and changed:
        print('Documentation drift detected:')
        for path in changed:
            print(f' - {path}')
        return 1
    return 0


def main() -> int:
    args = parse_args()
    facts_path = Path(args.facts)
    if not facts_path.exists():
        raise FileNotFoundError(f'Source-of-truth file not found: {facts_path}')

    facts = _read_json(facts_path)
    if not args.skip_validate:
        validate_facts(facts)
    block = _render_block(facts)
    targets = [Path(p) for p in args.targets]
    for target in targets:
        if not target.exists():
            raise FileNotFoundError(f'Target markdown file not found: {target}')

    code = sync_targets(targets, block, check=bool(args.check))
    if code == 0:
        action = 'Checked' if args.check else 'Synced'
        print(f'{action} documentation files successfully.')
    return code


if __name__ == '__main__':
    sys.exit(main())
