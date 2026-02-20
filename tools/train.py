
import argparse
import logging
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.registry import RUNNERS
from mmseg.utils import register_all_modules

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
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

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
    import deeproof.models.losses
    import deeproof.datasets.roof_dataset

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

    # MMEngine compatibility: enforce loop config triads.
    if cfg.get('val_dataloader') is not None and cfg.get('val_evaluator') is not None and cfg.get('val_cfg') is None:
        cfg.val_cfg = dict(type='ValLoop')
    if cfg.get('test_dataloader') is not None and cfg.get('test_evaluator') is not None and cfg.get('test_cfg') is None:
        cfg.test_cfg = dict(type='TestLoop')
    
    # Checkpoint Configuration (Best IoU)
    # Ensure default_hooks.checkpoint exists and configure it
    # We want to save the best model based on mIoU
    if 'default_hooks' in cfg and 'checkpoint' in cfg.default_hooks:
        cfg.default_hooks.checkpoint.save_best = 'mIoU'
        cfg.default_hooks.checkpoint.rule = 'greater'
        cfg.default_hooks.checkpoint.max_keep_ckpts = 3
    else:
        # If not in config, add it manually
        cfg.setdefault('default_hooks', {})
        cfg.default_hooks['checkpoint'] = dict(
            type='CheckpointHook',
            interval=1,
            save_best='mIoU',
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

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()

if __name__ == '__main__':
    main()
