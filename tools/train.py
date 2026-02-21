
import argparse
import hashlib
import os
import os.path as osp
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

from deeproof.utils.runtime_compat import apply_runtime_compat

def parse_args():
    parser = argparse.ArgumentParser(description='Train DeepRoof-2026 Model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=False,
        help='Enable deterministic CUDA behavior (slower, more reproducible)')
    parser.add_argument(
        '--hard-examples-file',
        type=str,
        default='',
        help='Optional text file with hard sample ids for oversampling')
    parser.add_argument(
        '--hard-example-repeat',
        type=int,
        default=1,
        help='Repeat factor for hard samples (>=1)')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _set_global_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _runtime_versions() -> dict:
    versions = {'torch': torch.__version__}
    try:
        import mmcv
        versions['mmcv'] = mmcv.__version__
    except Exception:
        versions['mmcv'] = 'N/A'
    try:
        import mmseg
        versions['mmseg'] = mmseg.__version__
    except Exception:
        versions['mmseg'] = 'N/A'
    try:
        import mmdet
        versions['mmdet'] = mmdet.__version__
    except Exception:
        versions['mmdet'] = 'N/A'
    return versions


def _dump_resolved_cfg(cfg: Config):
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = work_dir / 'resolved_config.py'
    cfg_text = cfg.pretty_text
    resolved_path.write_text(cfg_text, encoding='utf-8')
    versions = _runtime_versions()
    runtime_path = work_dir / 'runtime_versions.txt'
    runtime_path.write_text(
        '\n'.join([f'{k}={v}' for k, v in versions.items()]),
        encoding='utf-8')

    def _sha256_file(path: Path) -> str:
        if not path.exists() or not path.is_file():
            return ''
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    git_sha = ''
    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        git_sha = ''

    data_hash = ''
    train_ds = cfg.get('train_dataloader', {}).get('dataset', {})
    data_root = train_ds.get('data_root', '')
    ann_file = train_ds.get('ann_file', '')
    if data_root and ann_file:
        ann_path = Path(data_root) / ann_file
        data_hash = _sha256_file(ann_path)

    manifest = {
        'git_sha': git_sha,
        'config_sha256': hashlib.sha256(cfg_text.encode('utf-8')).hexdigest(),
        'data_split_sha256': data_hash,
        'runtime_versions': versions,
    }
    try:
        import json
        (work_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    except Exception:
        pass


def _ensure_iou_metric_prefix_and_best_key(cfg: Config) -> str:
    """Ensure IoUMetric has explicit prefix and return save_best key."""
    default_key = 'mIoU'
    val_evaluator = cfg.get('val_evaluator', None)
    if val_evaluator is None:
        return default_key

    metric_cfgs = []
    if isinstance(val_evaluator, dict):
        metric_cfgs = [val_evaluator]
    elif isinstance(val_evaluator, (list, tuple)):
        metric_cfgs = [m for m in val_evaluator if isinstance(m, dict)]

    for metric in metric_cfgs:
        metric_type = metric.get('type', '')
        if metric_type != 'IoUMetric':
            continue
        # Suppress mmengine warning and keep metric name stable by default.
        if metric.get('prefix', None) is None:
            metric['prefix'] = ''
        prefix = metric.get('prefix', '')
        if not prefix:
            return default_key
        return f'{prefix}/mIoU'

    return default_key


def _safe_torch_load(path: Path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location='cpu')


def _find_last_checkpoint(work_dir: str | Path) -> Path | None:
    marker = Path(work_dir) / 'last_checkpoint'
    if not marker.exists():
        return None
    raw = marker.read_text(encoding='utf-8').strip()
    if not raw:
        return None
    ckpt_path = Path(raw)
    if not ckpt_path.is_absolute():
        ckpt_path = (Path(work_dir) / ckpt_path).resolve()
    if ckpt_path.exists():
        return ckpt_path
    return None


def _required_state_prefixes_from_cfg(cfg: Config) -> list[str]:
    prefixes: list[str] = []
    model = cfg.get('model', None)
    if not isinstance(model, dict):
        return prefixes
    if model.get('geometry_head', None) is not None:
        prefixes.append('geometry_head.')
    if model.get('dense_geometry_head', None) is not None:
        prefixes.append('dense_geometry_head.')
    if model.get('edge_head', None) is not None:
        prefixes.append('edge_head.')
    return prefixes


def _checkpoint_has_required_prefixes(ckpt_path: Path, required_prefixes: list[str]) -> bool:
    if not required_prefixes:
        return True
    try:
        ckpt_obj = _safe_torch_load(ckpt_path)
    except Exception:
        return False
    if not isinstance(ckpt_obj, dict):
        return False
    state_dict = ckpt_obj.get('state_dict', ckpt_obj)
    if not isinstance(state_dict, dict):
        return False
    keys = tuple(state_dict.keys())
    return all(any(k.startswith(prefix) for k in keys) for prefix in required_prefixes)


def _apply_safe_resume_fallback(cfg: Config, resume_requested: bool):
    if not resume_requested:
        return
    work_dir = cfg.get('work_dir', None)
    if not work_dir:
        return
    last_ckpt = _find_last_checkpoint(work_dir)
    if last_ckpt is None:
        return
    required_prefixes = _required_state_prefixes_from_cfg(cfg)
    if _checkpoint_has_required_prefixes(last_ckpt, required_prefixes):
        return
    # Incompatible architecture for full resume: load model weights only.
    cfg.resume = False
    cfg.load_from = str(last_ckpt)
    print(
        f'[safe-resume] Incompatible last checkpoint: {last_ckpt}. '
        'Switched to weights-only load_from to avoid optimizer param-group mismatch.'
    )


def main():
    args = parse_args()

    # Register all modules in mmseg into the registries
    # Do not init the default scope here because we might register custom modules
    register_all_modules(init_default_scope=False)
    
    # Import custom modules to ensure they are registered
    # (DeepRoofMask2Former, GeometryHead, DeepRoofLosses)
    import deeproof.models.backbones.swin_v2_compat
    import deeproof.models.deeproof_model
    import deeproof.models.heads.mask2former_head
    import deeproof.models.heads.geometry_head
    import deeproof.models.heads.dense_normal_head
    import deeproof.models.heads.edge_head
    import deeproof.models.losses
    import deeproof.datasets.roof_dataset

    _set_global_seed(seed=args.seed, deterministic=args.deterministic)

    # Load config
    cfg = Config.fromfile(args.config)
    
    # Update config via CLI
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
        
    # Initial setup
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Default work_dir based on config filename
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Enable AMP
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # Resume training
    if args.resume:
        cfg.resume = True

    # Reproducibility and compatibility.
    cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)
    if args.hard_examples_file:
        cfg.train_dataloader.dataset.hard_examples_file = args.hard_examples_file
        cfg.train_dataloader.dataset.hard_example_repeat = max(int(args.hard_example_repeat), 1)
    apply_runtime_compat(cfg)
    _apply_safe_resume_fallback(cfg, resume_requested=args.resume)
    
    save_best_key = _ensure_iou_metric_prefix_and_best_key(cfg)

    # Checkpoint Configuration (Best IoU)
    # Ensure default_hooks.checkpoint exists and configure it
    # We want to save the best model based on mIoU
    if 'default_hooks' in cfg and 'checkpoint' in cfg.default_hooks:
        cfg.default_hooks.checkpoint.save_best = save_best_key
        cfg.default_hooks.checkpoint.rule = 'greater'
        cfg.default_hooks.checkpoint.max_keep_ckpts = 3
    else:
        # If not in config, add it manually
        cfg.setdefault('default_hooks', {})
        cfg.default_hooks['checkpoint'] = dict(
            type='CheckpointHook',
            interval=1,
            save_best=save_best_key,
            rule='greater',
            max_keep_ckpts=3
        )

    # Visualization / Logging
    # Ensure Tensorboard is enabled
    if 'visualizer' in cfg:
        # Add TensorboardVisBackend if not present
        if 'vis_backends' not in cfg.visualizer:
             cfg.visualizer.vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
        else:
             # Check if Tensorboard is already there
             has_tb = any(b['type'] == 'TensorboardVisBackend' for b in cfg.visualizer.vis_backends)
             if not has_tb:
                 cfg.visualizer.vis_backends.append(dict(type='TensorboardVisBackend'))
    else:
         # Default visualizer setup
         cfg.visualizer = dict(
            type='SegLocalVisualizer',
            vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
            name='visualizer')

    _dump_resolved_cfg(cfg)

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()

if __name__ == '__main__':
    main()
