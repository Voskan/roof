import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

from deeproof.utils.runtime_compat import apply_runtime_compat

# 1. Environment Setup
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Register all mmseg and deeproof modules
register_all_modules(init_default_scope=False)

# Import custom modules for registration
import deeproof.models.backbones.swin_v2_compat
import deeproof.models.deeproof_model
import deeproof.models.heads.mask2former_head
import deeproof.models.heads.geometry_head
import deeproof.models.losses
import deeproof.datasets.roof_dataset

# 2. Configuration
CONFIG_PATH = str(project_root / 'configs/deeproof_production_swin_L.py')
WORK_DIR = str(project_root / 'work_dirs/deeproof_absolute_ideal_v1')

cfg = Config.fromfile(CONFIG_PATH)
cfg.default_scope = 'mmseg'
cfg.work_dir = WORK_DIR

# 3. Path Overrides (Ensure data is found)
cfg.data_root = str(project_root / 'data/OmniCity/')
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_root = cfg.data_root

# MMEngine compatibility: enforce loop config triads.
if cfg.get('val_dataloader') is not None and cfg.get('val_evaluator') is not None and cfg.get('val_cfg') is None:
    cfg.val_cfg = dict(type='ValLoop')
if cfg.get('test_dataloader') is not None and cfg.get('test_evaluator') is not None and cfg.get('test_cfg') is None:
    cfg.test_cfg = dict(type='TestLoop')

apply_runtime_compat(cfg)

# 4. Checkpoint Loading
# Start from the best available baseline to preserve learned features.
# New query slots (300 total) will be initialized while preserving the first 100.
baseline_ckpt = str(project_root / 'work_dirs/swin_l_scratch_v1/iter_40000.pth')
if os.path.exists(baseline_ckpt):
    cfg.load_from = baseline_ckpt
    print(f"[*] Loading baseline checkpoint: {baseline_ckpt}")
else:
    print("[!] Warning: Baseline checkpoint not found. Starting training from scratch.")

# 5. Kickoff
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg.randomness = dict(seed=seed, deterministic=False)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"[*] Starting Absolute Ideal SOTA Training on: {device_name}")
    print(f"[*] Config:   {CONFIG_PATH}")
    print(f"[*] Work Dir: {WORK_DIR}")
    
    runner = Runner.from_cfg(cfg)
    runner.train()
