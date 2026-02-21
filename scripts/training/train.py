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
from mmseg.registry import RUNNERS

from deeproof.utils.runtime_compat import apply_runtime_compat

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='train config file path', required=True)
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Check the kb up for more details.')
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
        help='Enable deterministic CUDA behavior')
    parser.add_argument(
        '--hard-examples-file',
        type=str,
        default='',
        help='Optional text file with hard sample ids')
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


def _dump_resolved_cfg(cfg: Config):
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = cfg.pretty_text
    (work_dir / 'resolved_config.py').write_text(cfg_text, encoding='utf-8')

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
    }
    try:
        import json
        (work_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    except Exception:
        pass


def main():
    args = parse_args()
    _set_global_seed(seed=args.seed, deterministic=args.deterministic)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'OptimWrapper':
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
        else:
            print(f'Cannot set amp for {optim_wrapper}')

    # resume training
    cfg.resume = args.resume
    cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)
    if args.hard_examples_file:
        cfg.train_dataloader.dataset.hard_examples_file = args.hard_examples_file
        cfg.train_dataloader.dataset.hard_example_repeat = max(int(args.hard_example_repeat), 1)
    apply_runtime_compat(cfg)
    _dump_resolved_cfg(cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()

if __name__ == '__main__':
    main()
